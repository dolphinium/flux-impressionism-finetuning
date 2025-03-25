import json
from datasets import load_dataset, Dataset
import pandas as pd
from huggingface_hub import login
import os
from dotenv import load_dotenv

def update_dataset_with_captions():
    # Load the original dataset
    print("Loading original dataset...")
    dataset = load_dataset("dolphinium/wikiart-impressionism-curated", split="train")
    
    # Load the captions
    print("Loading captions...")
    with open("src/image2caption/output/final_dataset/dataset_with_fixed_captions.json", "r") as f:
        captions_data = json.load(f)
    
    # Create a lookup dictionary for captions
    captions_lookup = {item['id']: item['caption'] for item in captions_data}
    
    # Convert dataset to pandas DataFrame
    df = dataset.to_pandas()
    
    # Add id column (if not already present)
    if 'id' not in df.columns:
        df['id'] = range(len(df))
    
    # Add captions
    df['caption'] = df['id'].map(captions_lookup)
    
    # Convert back to Hugging Face dataset
    updated_dataset = Dataset.from_pandas(df)
    
    # Push to hub
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("Please set HUGGINGFACE_TOKEN in your .env file")
    
    login(token)
    
    print("Pushing updated dataset to Hugging Face...")
    updated_dataset.push_to_hub(
        "dolphinium/wikiart-impressionism-curated",
        split="train",
        private=False
    )
    
    print("Dataset successfully updated!")

if __name__ == "__main__":
    update_dataset_with_captions() 