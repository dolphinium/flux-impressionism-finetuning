import os
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import time
from dataclasses import dataclass
from dotenv import load_dotenv
from PIL import Image
import io

import google.generativeai as genai
from datasets import load_dataset
import aiohttp
from tqdm.asyncio import tqdm
import pandas as pd


load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class APIKey:
    key: str
    requests_this_minute: int = 0
    last_request_time: float = 0
    total_requests_today: int = 0

class GeminiCaptioner:
    def __init__(self, api_keys: List[str]):
        self.api_keys = [APIKey(key) for key in api_keys]
        self.current_key_index = 0
        self.genre_mapping = {
            1: "Urban Scenes",
            4: "Landscapes",
            6: "Portraits",
            9: "Still Life"
        }

    async def get_next_available_key(self) -> APIKey:
        while True:
            key = self.api_keys[self.current_key_index]
            current_time = time.time()
            
            # Reset minute counter if a minute has passed
            if current_time - key.last_request_time >= 60:
                key.requests_this_minute = 0
                
            # Reset daily counter if 24 hours have passed
            if current_time - key.last_request_time >= 86400:
                key.total_requests_today = 0

            if (key.requests_this_minute < 15 and 
                key.total_requests_today < 1500):
                return key
                
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            if all(k.requests_this_minute >= 15 for k in self.api_keys):
                await asyncio.sleep(5)

    def create_prompt(self, genre_id: int) -> str:
        genre = self.genre_mapping.get(genre_id, "Unknown")
        return f"""Analyze this Impressionist {genre} painting and provide a concise, descriptive caption suitable for training a text-to-image model. Focus on:
1. Key visual elements and their arrangement
2. Dominant colors and lighting
3. Artistic style and brushwork characteristics
4. Overall mood and atmosphere

Format the caption as a single, flowing description that a text-to-image model could use to recreate a similar image. Include 'impressionist style' or 'impressionist painting' in the description.

Return the response in JSON format with a single key 'description' containing the caption. The caption should be a single sentence or short paragraph without line breaks.

Example format:
{{"description": "An impressionist painting of a sunlit garden with vibrant purple and yellow flowers, loose brushstrokes creating a sense of movement, warm afternoon light filtering through the foliage"}}"""

    async def generate_caption(self, image_data: Dict[str, Any], session: aiohttp.ClientSession) -> Dict[str, Any]:
        try:
            api_key = await self.get_next_available_key()
            genai.configure(api_key=api_key.key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            
            # Convert image to PIL Image if it's not already
            if not isinstance(image_data['image'], Image.Image):
                image = Image.open(io.BytesIO(image_data['image']['bytes']))
            else:
                image = image_data['image']
            
            prompt = self.create_prompt(image_data['genre'])
            
            response = await asyncio.to_thread(
                model.generate_content,
                contents=[prompt, image],
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": 2048,
                    "top_p": 0.9,
                    "top_k": 40,
                    "response_mime_type": "application/json",
                }
            )
            
            api_key.requests_this_minute += 1
            api_key.total_requests_today += 1
            api_key.last_request_time = time.time()

            # Parse JSON response
            try:
                caption = json.loads(response.text)['description']
            except (json.JSONDecodeError, KeyError):
                caption = response.text  # Fallback to raw text if JSON parsing fails

            return {
                'id': image_data['id'],
                'artist': image_data['artist'],
                'genre': image_data['genre'],
                'style': image_data['style'],
                'caption': caption,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error generating caption: {str(e)}")
            return {
                'id': image_data['id'],
                'artist': image_data['artist'],
                'genre': image_data['genre'],
                'style': image_data['style'],
                'caption': '',
                'error': str(e)
            }

class Pipeline:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.intermediate_dir = self.output_dir / 'intermediate'
        self.final_dir = self.output_dir / 'final_dataset'
        
        # Create directories
        for dir_path in [self.checkpoint_dir, self.intermediate_dir, self.final_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def load_checkpoint(self) -> Dict[str, Any]:
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_*.json'))
        if not checkpoint_files:
            return {'processed_images': [], 'last_index': 0}
        
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        with open(latest_checkpoint, 'r') as f:
            return json.load(f)

    def save_checkpoint(self, data: Dict[str, Any]):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = self.checkpoint_dir / f'checkpoint_{timestamp}.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f)

    async def process_batch(
        self, 
        captioner: GeminiCaptioner,
        batch,
        batch_start_idx: int
    ) -> List[Dict[str, Any]]:
        async with aiohttp.ClientSession() as session:
            tasks = []
            # Convert batch to list of dictionaries
            batch_dict = batch.to_dict()
            for i in range(len(batch)):
                global_idx = batch_start_idx + i  # Calculate global index
                image_data = {
                    'image': batch_dict['image'][i],
                    'artist': batch_dict['artist'][i],
                    'genre': batch_dict['genre'][i],
                    'style': batch_dict['style'][i],
                    'id': global_idx  # Store as integer ID
                }
                tasks.append(captioner.generate_caption(image_data, session))
            return await tqdm.gather(*tasks)

    async def run(self, batch_size: int = 10):
        try:
            # Load dataset
            logging.info("Loading dataset...")
            dataset = load_dataset("dolphinium/wikiart-impressionism-curated", split="train")
            logging.info(f"Dataset loaded successfully with {len(dataset)} images")
            
            # Load checkpoint
            checkpoint = self.load_checkpoint()
            start_idx = checkpoint['last_index']
            logging.info(f"Starting from index {start_idx}")
            
            # Load API keys
            load_dotenv()
            api_keys = [
                os.getenv(f'GEMINI_API_KEY_{i}')
                for i in range(1, 4)
                if os.getenv(f'GEMINI_API_KEY_{i}')
            ]
            
            if not api_keys:
                raise ValueError("No API keys found in .env file")
            
            logging.info(f"Found {len(api_keys)} API keys")
            captioner = GeminiCaptioner(api_keys)
            
            # Process images
            for i in range(start_idx, len(dataset), batch_size):
                end_idx = min(i + batch_size, len(dataset))
                batch = dataset.select(range(i, end_idx))
                results = await self.process_batch(captioner, batch, i)  # Pass batch start index
                
                # Save intermediate results
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                intermediate_path = self.intermediate_dir / f'batch_{timestamp}.json'
                with open(intermediate_path, 'w') as f:
                    json.dump(results, f)
                
                # Update checkpoint
                checkpoint['last_index'] = end_idx
                checkpoint['processed_images'].extend(results)
                self.save_checkpoint(checkpoint)
                
                logging.info(f"Processed batch {i//batch_size + 1}/{len(dataset)//batch_size + 1}")
            
            # Save final dataset
            final_data = pd.DataFrame(checkpoint['processed_images'])
            final_data.to_json(self.final_dir / 'dataset_with_captions.json')
            logging.info("Pipeline completed successfully")
            
        except Exception as e:
            logging.error(f"Pipeline failed with error: {str(e)}")
            logging.error(f"Error type: {type(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise

def main():
    output_dir = "output"
    pipeline = Pipeline(output_dir)
    
    try:
        asyncio.run(pipeline.run())
    except KeyboardInterrupt:
        logging.info("Pipeline interrupted by user")
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()