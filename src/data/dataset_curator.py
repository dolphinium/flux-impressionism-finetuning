import os
from typing import Dict, List, Tuple
import logging
from datasets import load_dataset, Dataset
from PIL import Image
import io
from huggingface_hub import HfApi, create_repo
from tqdm.auto import tqdm
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetCurator:
    def __init__(self, cache_dir: str = "./cache"):
        """Initialize the dataset curator.
        
        Args:
            cache_dir: Directory for caching dataset files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.target_counts = {
            4: 300,  # landscapes
            6: 300,  # portraits
            1: 200,  # urban scenes
            9: 200,  # still life
        }
        self.style_id = 12  # Impressionism
        
    def validate_image(self, image: Image.Image) -> bool:
        """Validate image quality and dimensions.
        
        Args:
            image: PIL Image to validate
            
        Returns:
            bool: True if image meets quality criteria
        """
        try:
            # Check minimum dimensions
            if min(image.size) < 512:
                return False
                
            # Check aspect ratio
            aspect_ratio = max(image.size) / min(image.size)
            if aspect_ratio > 2.0:
                return False
                
            # Basic quality check (file size relative to dimensions)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=95)
            file_size = len(img_byte_arr.getvalue())
            pixels = image.size[0] * image.size[1]
            if file_size / pixels < 0.1:  # Arbitrary threshold
                return False
                
            return True
        except Exception as e:
            logger.warning(f"Image validation error: {e}")
            return False

    def process_dataset(self) -> Dataset:
        """Process the WikiArt dataset to create curated subset.
        
        Returns:
            Dataset: Curated dataset
        """
        logger.info("Loading WikiArt dataset...")
        dataset = load_dataset("huggan/wikiart", streaming=True)
        
        # Initialize storage for selected examples
        selected_examples = {genre_id: [] for genre_id in self.target_counts.keys()}
        
        # Process the dataset
        logger.info("Processing dataset...")
        for example in tqdm(dataset["train"], desc="Processing images"):
            # Check if we have all required images
            if all(len(selected_examples[genre_id]) >= count 
                  for genre_id, count in self.target_counts.items()):
                break
                
            # Check style and genre
            if example["style"] != self.style_id:
                continue
            
            genre_id = example["genre"]
            if genre_id not in self.target_counts:
                continue
                
            if len(selected_examples[genre_id]) >= self.target_counts[genre_id]:
                continue
                
            # Validate image
            try:
                # The image is already a PIL Image object in the dataset
                if not self.validate_image(example["image"]):
                    continue
                    
                # Store the example
                selected_examples[genre_id].append({
                    "image": example["image"],
                    "artist": example["artist"],
                    "genre": example["genre"],
                    "style": example["style"]
                })
                
                # Log progress
                total_selected = sum(len(examples) for examples in selected_examples.values())
                if total_selected % 100 == 0:
                    logger.info(f"Selected {total_selected} images so far")
                    
            except Exception as e:
                logger.warning(f"Error processing image: {e}")
                continue
                
        # Combine all selected examples
        all_examples = []
        for examples in selected_examples.values():
            all_examples.extend(examples)
            
        # Create final dataset
        return Dataset.from_dict({
            "image": [ex["image"] for ex in all_examples],
            "artist": [ex["artist"] for ex in all_examples],
            "genre": [ex["genre"] for ex in all_examples],
            "style": [ex["style"] for ex in all_examples]
        })

    def upload_to_hub(self, dataset: Dataset, repo_name: str, private: bool = False) -> str:
        """Upload the curated dataset to Hugging Face Hub.
        
        Args:
            dataset: Dataset to upload
            repo_name: Name for the dataset repository
            private: Whether to create a private repository
            
        Returns:
            str: URL of the uploaded dataset
        """
        try:
            # Create the repository
            api = HfApi()
            repo_url = create_repo(
                repo_name,
                repo_type="dataset",
                private=private,
                exist_ok=True
            )
            
            # Push the dataset
            logger.info(f"Uploading dataset to {repo_name}...")
            dataset.push_to_hub(repo_name)
            
            return repo_url
            
        except Exception as e:
            logger.error(f"Error uploading to hub: {e}")
            raise

def main():
    """Main execution function."""
    # Initialize curator
    curator = DatasetCurator()
    
    try:
        # Process dataset
        dataset = curator.process_dataset()
        
        # Upload to Hub
        repo_name = "impressionism-curated"
        repo_url = curator.upload_to_hub(dataset, repo_name)
        
        logger.info(f"Successfully uploaded dataset to: {repo_url}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main() 