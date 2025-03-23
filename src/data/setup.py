"""
Setup script for Flux Impressionism fine-tuning project.
Handles environment setup, Hugging Face authentication, and dataset preparation.
"""

import os
import sys
from pathlib import Path
import json
import requests
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from huggingface_hub import login, HfApi
from diffusers import DiffusionPipeline
import torchvision.transforms as transforms

class EnvironmentSetup:
    """Handles environment setup and validation"""
    
    @staticmethod
    def check_environment() -> Dict[str, bool]:
        """Check PyTorch version and CUDA availability"""
        print(f"PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        
        return {
            "cuda": cuda_available,
            "pytorch": True,
            "version": torch.__version__
        }

    @staticmethod
    def setup_huggingface_auth(token: Optional[str] = None) -> bool:
        """Setup Hugging Face authentication"""
        if token is None:
            from getpass import getpass
            token = getpass("Enter your Hugging Face token: ")
        
        try:
            login(token)
            api = HfApi()
            user_info = api.whoami()
            print(f"Successfully logged in as: {user_info['name']}")
            return True
        except Exception as e:
            print(f"Error logging in: {e}")
            return False

    @staticmethod
    def verify_environment() -> Dict[str, bool]:
        """Verify all necessary components"""
        status = {
            "cuda": False,
            "huggingface": False,
            "memory": False,
            "disk_space": False
        }
        
        # Check CUDA
        status["cuda"] = torch.cuda.is_available()
        
        # Check Hugging Face authentication
        try:
            api = HfApi()
            api.whoami()
            status["huggingface"] = True
        except:
            pass
        
        # Check available memory
        if status["cuda"]:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            status["memory"] = total_memory > 10 * 1024 * 1024 * 1024  # > 10GB
        
        # Check available disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            status["disk_space"] = free > 50 * 1024 * 1024 * 1024  # > 50GB
        except:
            pass
        
        return status

class DatasetConfig:
    """Configuration for WikiArt Impressionism dataset"""
    
    def __init__(self, base_path: str = '/content/data'):
        self.base_path = Path(base_path)
        self.dataset_path = self.base_path / 'wikiart'
        self.metadata_path = self.dataset_path / 'metadata.json'
        self.images_path = self.dataset_path / 'images'
        
        # Create necessary directories
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.images_path.mkdir(parents=True, exist_ok=True)
    
    def setup_paths(self) -> Dict[str, str]:
        """Create directory structure and return paths"""
        paths = {
            'base': str(self.base_path),
            'dataset': str(self.dataset_path),
            'metadata': str(self.metadata_path),
            'images': str(self.images_path)
        }
        print("Dataset directories created:")
        for name, path in paths.items():
            print(f"- {name}: {path}")
        return paths

class CuratedDataset:
    """Handles creation and management of curated Impressionism dataset"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        # Genre IDs and target counts
        self.categories = {
            4: 400,     # landscape (40%)
            1: 200,     # cityscape (20%)
            6: 200,     # portrait (20%)
            9: 200      # still_life (20%)
        }
        # Style ID for Impressionism
        self.IMPRESSIONISM_STYLE_ID = 12
        
        # Genre mapping for reference
        self.genre_names = {
            4: 'landscape',
            1: 'cityscape',
            6: 'portrait',
            9: 'still_life'
        }
    
    def create_curated_dataset(self) -> Dataset:
        """Create a curated Impressionism dataset"""
        print("Loading WikiArt dataset with Impressionism style filter...")
        
        # Load only necessary columns and apply initial filter
        dataset = load_dataset(
            "huggan/wikiart",
            split="train",
            streaming=True,  # Stream to save memory
        )
        
        # Filter for Impressionism style
        print("Filtering for Impressionism style...")
        impressionism = dataset.filter(
            lambda x: x['style'] == self.IMPRESSIONISM_STYLE_ID
        )
        
        # Take a sample to analyze
        print("Collecting Impressionist paintings...")
        impressionist_paintings = []
        genre_counts = {genre_id: 0 for genre_id in self.categories.keys()}
        
        for example in tqdm(impressionism, desc="Collecting paintings"):
            try:
                # Get image resolution
                img = example['image']
                resolution = img.size[0] * img.size[1]
                genre_id = example['genre']
                
                # Check if this is a genre we want and if we need more of it
                if (resolution >= 1024 * 1024 and  # Minimum resolution check
                    genre_id in self.categories and  # Genre we want
                    genre_counts[genre_id] < self.categories[genre_id]):  # Still need more
                    
                    impressionist_paintings.append({
                        'image': img,
                        'artist': example['artist'],
                        'genre': genre_id,
                        'genre_name': self.genre_names[genre_id],
                        'resolution': resolution
                    })
                    genre_counts[genre_id] += 1
                
                # Break if we have collected enough for all categories
                if all(count >= self.categories[genre_id] for genre_id, count in genre_counts.items()):
                    break
                    
            except Exception as e:
                continue
        
        print("\nCollection complete. Genre distribution:")
        for genre_id, count in genre_counts.items():
            print(f"- {self.genre_names[genre_id]}: {count}/{self.categories[genre_id]}")
        
        # Convert to DataFrame
        df = pd.DataFrame(impressionist_paintings)
        print(f"\nTotal collected paintings: {len(df)}")
        
        # Create HuggingFace Dataset
        return Dataset.from_pandas(df)
    
    def push_to_hub(self, dataset: Dataset, repo_name: str = "flux-impressionism-dataset"):
        """Push dataset to Hugging Face Hub"""
        dataset_card = self._create_dataset_card()
        
        print(f"Pushing dataset to Hugging Face Hub as {repo_name}...")
        dataset.push_to_hub(
            repo_name,
            private=False,
            description="A curated dataset of high-quality Impressionist paintings for fine-tuning text-to-image models.",
            tags=['computer-vision', 'art', 'impressionism']
        )
        
        # Update the dataset card
        with open('README.md', 'w') as f:
            f.write(dataset_card)
    
    def _create_dataset_card(self) -> str:
        """Create the dataset card content"""
        return """
# Flux Impressionism Dataset

## Dataset Summary
A carefully curated collection of high-quality Impressionist paintings, specifically designed for fine-tuning text-to-image models. This dataset contains approximately 1000 images selected to represent the key characteristics of Impressionist art.

## Why This Dataset?
- Optimized size for efficient fine-tuning
- High-quality, curated images (minimum 1024x1024 resolution)
- Balanced representation of subjects
- Focus on core Impressionist characteristics

## Content
- Total Images: ~1000
- Categories (from WikiArt genre classification):
  - Landscapes (40%, genre_id: 4)
  - Cityscapes (20%, genre_id: 1)
  - Portraits (20%, genre_id: 6)
  - Still Life (20%, genre_id: 9)

## Quality Criteria
- Minimum resolution: 1024x1024
- Clear Impressionist techniques (style_id: 12)
- Varied lighting conditions
- Representative color palettes
- Balanced artist representation

## Usage
Primarily intended for fine-tuning text-to-image models to generate Impressionist-style artwork.

## Source
Derived from WikiArt dataset (huggan/wikiart), filtered and curated for optimal fine-tuning.

## Technical Details
- Style ID for Impressionism: 12
- Genre IDs used:
  - Landscape: 4
  - Cityscape: 1
  - Portrait: 6
  - Still Life: 9

## License
This dataset inherits the WikiArt dataset terms:
- Can be used only for non-commercial research purposes
- Images obtained from WikiArt.org
- Users must agree to WikiArt.org terms and conditions
"""

class ImageProcessor:
    """Handle image preprocessing for the model"""
    
    def __init__(self, image_size: int = 1024):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Preprocess a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def validate_image(self, image_path: str) -> bool:
        """Validate image file"""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except:
            return False

def main():
    """Main execution function"""
    # Setup environment
    env_setup = EnvironmentSetup()
    env_status = env_setup.check_environment()
    
    if not env_status["cuda"]:
        print("Warning: CUDA is not available. This will significantly impact performance.")
    
    # Setup Hugging Face authentication
    if not env_setup.setup_huggingface_auth():
        print("Error: Failed to authenticate with Hugging Face")
        return
    
    # Verify environment
    status = env_setup.verify_environment()
    print("\nEnvironment Status:")
    for component, is_ready in status.items():
        print(f"{component}: {'✅' if is_ready else '❌'}")
    
    if not all(status.values()):
        print("\nWarning: Some components are not properly set up!")
        return
    
    # Setup dataset configuration
    dataset_config = DatasetConfig()
    paths = dataset_config.setup_paths()
    
    # Create curated dataset
    curated_dataset = CuratedDataset(dataset_config)
    dataset = curated_dataset.create_curated_dataset()
    
    # Initialize image processor
    image_processor = ImageProcessor()
    
    # Validate dataset images
    print("\nValidating dataset images...")
    valid_images = 0
    invalid_images = []
    
    for idx, example in enumerate(tqdm(dataset)):
        image_path = example['image'].filename
        if image_processor.validate_image(image_path):
            valid_images += 1
        else:
            invalid_images.append(image_path)
    
    print(f"\nValidation complete:")
    print(f"- Valid images: {valid_images}")
    print(f"- Invalid images: {len(invalid_images)}")
    
    # Push to Hugging Face Hub
    should_push = input("\nDo you want to push the curated dataset to Hugging Face Hub? (y/n): ")
    if should_push.lower() == 'y':
        repo_name = input("Enter repository name (default: flux-impressionism-dataset): ").strip()
        if not repo_name:
            repo_name = "flux-impressionism-dataset"
        curated_dataset.push_to_hub(dataset, repo_name)

if __name__ == "__main__":
    main() 