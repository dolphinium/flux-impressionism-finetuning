import os
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import time
from dotenv import load_dotenv
from datasets import load_dataset
import aiohttp
from tqdm.asyncio import tqdm
from pipeline import GeminiCaptioner, Pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fix_captions.log'),
        logging.StreamHandler()
    ]
)

async def fix_failed_captions(output_dir: str, batch_size: int = 3):
    pipeline_dir = Path(output_dir)
    intermediate_dir = pipeline_dir / 'intermediate'
    fixed_dir = pipeline_dir / 'fixed'
    fixed_dir.mkdir(exist_ok=True)

    # Load all intermediate batch results
    failed_items = []
    batch_files = sorted(intermediate_dir.glob('batch_*.json'))
    
    logging.info("Collecting failed captions...")
    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            batch_results = json.load(f)
            # Collect items with empty captions or rate limit errors
            failed_items.extend([
                item for item in batch_results 
                if not item.get('caption') or 
                (item.get('error') and '429' in str(item.get('error')))
            ])
    
    if not failed_items:
        logging.info("No failed captions found!")
        return

    logging.info(f"Found {len(failed_items)} failed captions to fix")
    
    # Load API keys
    load_dotenv()
    api_keys = [
        os.getenv(f'GEMINI_API_KEY_{i}')
        for i in range(1, 4)
        if os.getenv(f'GEMINI_API_KEY_{i}')
    ]
    
    if not api_keys:
        raise ValueError("No API keys found in .env file")
    
    captioner = GeminiCaptioner(api_keys)
    dataset = load_dataset("dolphinium/wikiart-impressionism-curated", split="train")
    
    # Process failed items in small batches
    fixed_results = []
    for i in range(0, len(failed_items), batch_size):
        batch_items = failed_items[i:i + batch_size]
        batch_indices = [item['id'] for item in batch_items]
        batch = dataset.select(batch_indices)
        
        logging.info(f"Processing batch {i//batch_size + 1}/{len(failed_items)//batch_size + 1}")
        
        try:
            async with aiohttp.ClientSession() as session:
                tasks = []
                batch_dict = batch.to_dict()
                for j, item in enumerate(batch_items):
                    image_data = {
                        'image': batch_dict['image'][j],
                        'artist': item['artist'],
                        'genre': item['genre'],
                        'style': item['style'],
                        'id': item['id']
                    }
                    tasks.append(captioner.generate_caption(image_data, session))
                
                results = await asyncio.gather(*tasks)
                fixed_results.extend(results)
                
                # Save progress
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                fixed_path = fixed_dir / f'fixed_batch_{timestamp}.json'
                with open(fixed_path, 'w') as f:
                    json.dump(results, f)
                
                # Add delay between batches to avoid rate limits
                await asyncio.sleep(1)
                
        except Exception as e:
            logging.error(f"Error processing batch: {str(e)}")
            continue
    
    # Create final merged dataset
    try:
        logging.info("Creating final merged dataset...")
        all_results = []
        
        # Load all original batch results
        for batch_file in batch_files:
            with open(batch_file, 'r') as f:
                batch_results = json.load(f)
                all_results.extend(batch_results)
        
        # Create lookup for fixed results
        fixed_lookup = {item['id']: item for item in fixed_results if item.get('caption')}
        
        # Update failed captions with fixed ones
        for item in all_results:
            if item['id'] in fixed_lookup:
                item.update(fixed_lookup[item['id']])
        
        # Save final merged dataset
        final_path = pipeline_dir / 'final_dataset' / 'dataset_with_fixed_captions.json'
        with open(final_path, 'w') as f:
            json.dump(all_results, f)
        
        logging.info(f"Fixed dataset saved to {final_path}")
        
    except Exception as e:
        logging.error(f"Error creating final dataset: {str(e)}")

def main():
    output_dir = "output"  # Use the same output directory as the original pipeline
    
    try:
        asyncio.run(fix_failed_captions(output_dir))
    except KeyboardInterrupt:
        logging.info("Fix process interrupted by user")
    except Exception as e:
        logging.error(f"Fix process failed: {str(e)}")

if __name__ == "__main__":
    main()