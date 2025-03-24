import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
from transformers import PreTrainedModel, Trainer, TrainingArguments, default_data_collator
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers import FluxPipeline
from datasets import Dataset, IterableDataset
from transformers import AdamW

class FluxLoRATrainer:
    def __init__(self, config: Dict[str, Any]):
        self.validate_config(config)
        self.config = config
        self.setup_logging()
        self.setup_accelerator()

    def validate_config(self, config: Dict[str, Any]):
        """Validate configuration types."""
        # Validate LoRA config
        assert isinstance(config["lora"]["rank"], int), "LoRA rank must be an integer"
        assert isinstance(config["lora"]["alpha"], (int, float)), "LoRA alpha must be a number"
        assert isinstance(config["lora"]["lora_dropout"], float), "LoRA dropout must be a float"

        # Validate training config
        assert isinstance(config["training"]["learning_rate"], (int, float)), "Learning rate must be a number"
        assert float(config["training"]["learning_rate"]) > 0, "Learning rate must be positive"
        assert isinstance(config["training"]["train_batch_size"], int), "Batch size must be an integer"
        assert isinstance(config["training"]["gradient_accumulation_steps"], int), "Gradient accumulation steps must be an integer"

        # Convert scientific notation if present
        if isinstance(config["training"]["learning_rate"], str):
            config["training"]["learning_rate"] = float(config["training"]["learning_rate"])

    def setup_logging(self):
        """Configure logging for the training process."""
        os.makedirs(self.config["output"]["logging_dir"], exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.config["output"]["logging_dir"], "training.log"))
            ]
        )
        self.logger = logging.getLogger(__name__)

    def setup_accelerator(self):
        """Configure accelerator for distributed training."""
        project_config = ProjectConfiguration(
            project_dir=self.config["output"]["output_dir"],
            logging_dir=self.config["output"]["logging_dir"]
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            mixed_precision="bf16" if self.config["mixed_precision"]["enabled"] else "no",
            project_config=project_config,
            log_with=self.config["output"]["report_to"]
        )

    def get_caption_from_genre(self, genre_id: int) -> str:
        """Generate a descriptive caption based on the genre ID."""
        genre_captions = {
            4: [  # Landscapes
                "An impressionist landscape painting with vibrant colors and natural scenery, capturing the play of light and atmosphere",
                "A scenic impressionist view of nature with characteristic loose brushstrokes and emphasis on natural light",
                "An outdoor landscape scene painted in impressionist style, showing the interplay of light and shadow",
                "A nature scene captured in the impressionist tradition, with visible brushwork and atmospheric effects",
                "An impressionist painting of a landscape with emphasis on natural light and seasonal atmosphere"
            ],
            6: [  # Portraits
                "An impressionist portrait with expressive brushwork and attention to light on the subject's features",
                "A character study painted in impressionist style, capturing the essence of the subject through color and light",
                "A portrait showing the subject through impressionist techniques, with emphasis on mood and atmosphere",
                "An impressionist painting focusing on human expression and the play of light on skin tones",
                "A portrait in impressionist style with loose brushwork and natural lighting effects"
            ],
            1: [  # Urban scenes
                "An impressionist urban scene with city life and architecture, showing the interplay of light and shadow",
                "A city view painted in impressionist style, capturing the atmosphere of modern life",
                "An impressionist painting of urban architecture and daily life, with characteristic brushwork",
                "A street scene captured through impressionist techniques, emphasizing light and movement",
                "An urban landscape in impressionist style showing the vitality of city life"
            ],
            9: [  # Still life
                "An impressionist still life with careful attention to light and color relationships between objects",
                "A collection of objects painted in impressionist style, showing the effects of natural light",
                "An arrangement of items captured through impressionist techniques with vibrant colors",
                "A still life scene with characteristic impressionist brushwork and attention to light effects",
                "An impressionist painting of everyday objects with emphasis on color harmony and light"
            ]
        }

        # Default to generic caption for unknown genre
        if genre_id not in genre_captions:
            self.logger.warning(f"Unknown genre ID: {genre_id}, using default caption")
            return "An impressionist painting with characteristic style and attention to light and color"
            
        # Use torch.randint for reproducibility with the same seed
        idx = torch.randint(0, len(genre_captions[genre_id]), (1,)).item()
        caption = genre_captions[genre_id][idx]
        
        # Log the selected caption for debugging
        self.logger.debug(f"Generated caption for genre {genre_id}: {caption}")
        
        return caption

    def load_model(self):
        """Load and prepare the Flux model with LoRA configuration."""
        # Load pipeline components
        self.logger.info("Loading pipeline components...")
        pipeline = FluxPipeline.from_pretrained(
            self.config["model"]["pretrained_model_name_or_path"],
            torch_dtype=getattr(torch, self.config["model"]["torch_dtype"]),
        )

        # Get the text encoder from the pipeline
        self.logger.info("Extracting text encoder...")
        model = pipeline.text_encoder
        
        # Print model architecture for debugging
        self.logger.info(f"Model architecture: {type(model).__name__}")
        
        # Check if model forward returns expected outputs - TEST FORWARD PASS
        test_inputs = pipeline.tokenizer(
            "Test impressionist caption", 
            return_tensors="pt",
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True
        )
        
        with torch.no_grad():
            test_outputs = model(**test_inputs)
            
        # Log output structure
        self.logger.info(f"Test forward pass output keys: {test_outputs.keys()}")
        if hasattr(test_outputs, 'last_hidden_state'):
            self.logger.info(f"Last hidden state shape: {test_outputs.last_hidden_state.shape}")
        if hasattr(test_outputs, 'pooler_output'):
            self.logger.info(f"Pooler output shape: {test_outputs.pooler_output.shape}")

        # Enable gradient checkpointing if configured
        if self.config["system"]["gradient_checkpointing"]:
            self.logger.info("Enabling gradient checkpointing...")
            model.gradient_checkpointing_enable()

        # Configure LoRA - MODIFIED CONFIGURATION FOR EMBEDDING MODEL
        self.logger.info("Configuring LoRA...")
        lora_config = LoraConfig(
            r=self.config["lora"]["rank"],
            lora_alpha=self.config["lora"]["alpha"],
            target_modules=self.config["lora"]["target_modules"],
            lora_dropout=self.config["lora"]["lora_dropout"],
            bias=self.config["lora"]["bias"],
            task_type="CAUSAL_LM"  # Changed from "FEATURE_EXTRACTION" to "CAUSAL_LM" for better compatibility
        )

        # Apply LoRA
        self.logger.info("Applying LoRA configuration...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # Print trainable parameters info

        # Ensure model is in training mode
        model.train()

        # Double-check trainable parameters
        trainable_params = 0
        all_param = 0
        for name, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                self.logger.info(f"Trainable param: {name}")
        
        self.logger.info(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%"
        )
        
        # Check if any LoRA params are trainable
        lora_param_names = [n for n, p in model.named_parameters() if 'lora' in n.lower() and p.requires_grad]
        if not lora_param_names:
            self.logger.warning("NO LORA PARAMETERS ARE TRAINABLE! Training will not be effective.")
        else:
            self.logger.info(f"Found {len(lora_param_names)} trainable LoRA parameters")

        # Store model and pipeline
        self.model = model
        self.pipeline = pipeline

        return self.model

    def prepare_dataset(self, dataset: Dataset):
        """Prepare dataset for training with streaming to save memory."""
        self.logger.info("Preparing dataset for training...")

        # Convert to streaming dataset if not already
        if not isinstance(dataset, IterableDataset):
            self.logger.info("Converting to streaming dataset...")
            dataset = dataset.to_iterable_dataset(num_shards=16)

        if self.config["dataset"]["max_train_samples"]:
            dataset = dataset.take(self.config["dataset"]["max_train_samples"])

        # Add preprocessing for text
        def preprocess_function(examples):
            # Handle both single examples and batches
            if isinstance(examples["genre"], (list, np.ndarray)):
                genre_ids = examples["genre"]
                captions = [self.get_caption_from_genre(g) for g in genre_ids]
            else:
                genre_id = examples["genre"]
                captions = [self.get_caption_from_genre(genre_id)]
            
            # Debug logging for first few examples
            if isinstance(examples["genre"], (list, np.ndarray)):
                for i, (genre, caption) in enumerate(zip(examples["genre"][:3], captions[:3])):
                    self.logger.info(f"Example {i}: Genre {genre} -> Caption: {caption}")
            else:
                self.logger.info(f"Single example: Genre {examples['genre']} -> Caption: {captions[0]}")
            
            # Process text inputs using the tokenizer
            text_inputs = self.pipeline.tokenizer(
                captions,
                padding="max_length",
                max_length=self.pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            # Debug tokenization for first example
            if isinstance(examples["genre"], (list, np.ndarray)):
                decoded = self.pipeline.tokenizer.decode(text_inputs["input_ids"][0])
                self.logger.info(f"Tokenized and decoded back: {decoded}")

            # Convert all values to tensors and ensure they're properly batched
            processed = {}
            for key, value in text_inputs.items():
                if isinstance(value, torch.Tensor):
                    # Ensure the tensor is 2D (batch_size x sequence_length)
                    if value.dim() == 1:
                        value = value.unsqueeze(0)
                    processed[key] = value
                else:
                    processed[key] = torch.tensor(value, dtype=torch.long)
                    if processed[key].dim() == 1:
                        processed[key] = processed[key].unsqueeze(0)

            return processed

        # Remove all columns except genre as we'll generate captions from it
        keep_columns = ["genre"] if "genre" in dataset.column_names else []
        processed_dataset = dataset.map(
            preprocess_function,
            remove_columns=[col for col in dataset.column_names if col not in keep_columns],
            batched=True,
            batch_size=self.config["training"]["train_batch_size"] * 2  # Double batch size for better efficiency
        )

        # Log a few processed examples
        self.logger.info("\nFirst few processed examples:")
        for i, example in enumerate(processed_dataset.take(2)):
            self.logger.info(f"Processed example {i}:")
            self.logger.info(f"Genre: {example['genre']}")
            decoded = self.pipeline.tokenizer.decode(example['input_ids'])
            self.logger.info(f"Decoded text: {decoded}")
            self.logger.info(f"Input shape: {example['input_ids'].shape}")
            self.logger.info(f"Attention mask sum: {example['attention_mask'].sum()}\n")

        return processed_dataset

    def validate_dataset(self, dataset):
        """Validate dataset structure and batch sizes before training."""
        self.logger.info("Validating dataset configuration...")
        
        # Check if dataset is empty
        try:
            first_batch = next(iter(dataset))
            self.logger.info(f"First batch keys: {list(first_batch.keys())}")
        except StopIteration:
            raise ValueError("Dataset is empty!")
            
        # Validate required keys
        required_keys = {'input_ids', 'attention_mask'}
        missing_keys = required_keys - set(first_batch.keys())
        if missing_keys:
            raise ValueError(f"Dataset missing required keys: {missing_keys}")
            
        # Check tensor shapes and batch size
        for key, value in first_batch.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Dataset value for {key} is not a tensor")
            self.logger.info(f"{key} shape: {value.shape}")
            
            # For input_ids and attention_mask, ensure they're 2D (batch_size x sequence_length)
            if key in ['input_ids', 'attention_mask']:
                if value.dim() != 2:
                    raise ValueError(f"{key} should be 2D (batch_size x sequence_length), got {value.dim()}D")
                    
        # Check batch size
        batch_size = first_batch['input_ids'].shape[0]
        if batch_size < 2:  # We need at least 2 examples for contrastive learning
            raise ValueError(f"Batch size too small for contrastive learning: {batch_size}, minimum required: 2")
            
        self.logger.info(f"Batch size: {batch_size}")
        
        # Validate a few batches to ensure consistency
        num_batches_to_check = 3
        self.logger.info(f"Checking first {num_batches_to_check} batches...")
        
        for i, batch in enumerate(dataset):
            if i >= num_batches_to_check:
                break
                
            current_batch_size = batch['input_ids'].shape[0]
            self.logger.info(f"Batch {i} size: {current_batch_size}")
            
            if current_batch_size < 2:
                raise ValueError(f"Batch {i} has size {current_batch_size}, which is too small for contrastive learning")
                
            # Check tensor shapes are consistent
            for key, value in batch.items():
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"Batch {i}: {key} is not a tensor")
                if key in ['input_ids', 'attention_mask']:
                    if value.dim() != 2:
                        raise ValueError(f"Batch {i}: {key} should be 2D, got {value.dim()}D")
        
        self.logger.info("Dataset validation completed successfully!")
        return True

    def train(self, dataset: Dataset):
        """Execute the training loop."""
        # Validate dataset before training
        self.validate_dataset(dataset)
        
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=self.config["output"]["output_dir"],
            per_device_train_batch_size=self.config["training"]["train_batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation_steps"],
            learning_rate=float(self.config["training"]["learning_rate"]),
            lr_scheduler_type=self.config["training"]["lr_scheduler"],
            num_train_epochs=self.config["training"]["num_train_epochs"],
            max_steps=self.config["training"]["max_train_steps"],
            warmup_steps=self.config["training"]["lr_warmup_steps"],
            save_steps=self.config["training"]["checkpointing_steps"],
            save_total_limit=self.config["training"]["save_total_limit"],
            logging_steps=10,
            remove_unused_columns=False,
            seed=self.config["training"]["seed"],
            bf16=self.config["mixed_precision"]["enabled"],
            report_to=self.config["output"]["report_to"],
            label_names=[],  # Explicitly set empty label names
            dataloader_num_workers=4,  # Add workers for faster data loading
            dataloader_pin_memory=True,  # Enable pin memory for better performance
            gradient_checkpointing=self.config["system"]["gradient_checkpointing"],
            # Enable gradient clipping to prevent NaN losses
            max_grad_norm=1.0,
        )

        # Initialize trainer with improved contrastive learning loss
        class ImprovedCLIPTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Initialize temperature parameter properly as a leaf tensor
                self.log_temperature = torch.nn.Parameter(
                    torch.tensor(np.log(1/0.07), dtype=torch.float32, requires_grad=True)
                )
                self.max_temperature = 100.0
                
                # Print all parameters to check trainable status
                lora_params = [n for n, p in self.model.named_parameters() if 'lora' in n.lower() and p.requires_grad]
                print(f"Number of trainable LoRA parameters: {len(lora_params)}")
                if len(lora_params) == 0:
                    print("WARNING: No LoRA parameters are trainable!")

            def create_optimizer(self):
                """Override to include temperature parameter in optimization"""
                if self.optimizer is None:
                    # Get all trainable parameters from the model
                    decay_parameters = []
                    no_decay_parameters = []
                    
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            if any(nd in name for nd in ["bias", "LayerNorm.weight"]):
                                no_decay_parameters.append(param)
                            else:
                                decay_parameters.append(param)
                    
                    # Create optimizer groups
                    optimizer_grouped_parameters = [
                        {
                            "params": decay_parameters,
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": no_decay_parameters,
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [self.log_temperature],
                            "weight_decay": 0.0,
                            "lr": self.args.learning_rate * 0.1,  # Lower learning rate for temperature
                        }
                    ]
                    
                    # Create optimizer - always use AdamW
                    self.optimizer = AdamW(
                        optimizer_grouped_parameters,
                        lr=self.args.learning_rate,
                        betas=(0.9, 0.999),  # Default AdamW betas
                        eps=1e-8,  # Default AdamW epsilon
                    )
                    
                return self.optimizer

            def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
                # Ensure we're in training mode
                model.train()

                # Move inputs to the model's device
                device = next(model.parameters()).device
                input_ids = inputs.get("input_ids").to(device)
                attention_mask = inputs.get("attention_mask").to(device)

                batch_size = input_ids.shape[0]
                if batch_size <= 1:
                    print("WARNING: Batch size is 1 or less, skipping loss computation")
                    dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    return (dummy_loss, None) if return_outputs else dummy_loss
                
                # Move temperature parameter to correct device
                self.log_temperature = self.log_temperature.to(device)

                # Forward pass through model
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Determine which output to use for embeddings
                # Try different outputs based on model architecture
                if hasattr(outputs, 'text_embeds'):
                    text_embeds = outputs.text_embeds
                elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    text_embeds = outputs.pooler_output
                else:
                    # Fallback to last hidden state with mean pooling
                    last_hidden = outputs.hidden_states[-1]
                    # Apply attention mask for proper mean pooling
                    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                    sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
                    sum_mask = torch.sum(mask_expanded, 1)
                    sum_mask = torch.clamp(sum_mask, min=1e-9)
                    text_embeds = sum_embeddings / sum_mask
                
                # Check for NaN values in embeddings
                if torch.isnan(text_embeds).any():
                    print("WARNING: NaN values detected in embeddings")
                    non_nan_mask = ~torch.isnan(text_embeds)
                    if non_nan_mask.any():
                        # Replace NaNs with mean of non-NaN values
                        mean_val = text_embeds[non_nan_mask].mean()
                        text_embeds = torch.nan_to_num(text_embeds, nan=mean_val.item())
                    else:
                        # All values are NaN, replace with zeros
                        text_embeds = torch.zeros_like(text_embeds)
                        # Return dummy loss to avoid breaking training
                        dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)
                        print("ERROR: All embedding values are NaN, returning dummy loss")
                        return (dummy_loss, outputs) if return_outputs else dummy_loss
                
                # Normalize embeddings for cosine similarity
                text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)

                # Compute temperature-scaled similarity matrix
                temperature = torch.exp(self.log_temperature).clamp(max=self.max_temperature)
                
                # Compute logits with numerical stability check
                logits = torch.matmul(text_embeds, text_embeds.t()) / temperature
                
                # Check for NaN/inf in logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("WARNING: NaN or Inf values detected in logits")
                    logits = torch.nan_to_num(logits)
                
                # Print statistics for debugging
                print(f"Logits shape: {logits.shape}, min: {logits.min().item():.4f}, max: {logits.max().item():.4f}")
                print(f"Temperature: {temperature.item():.4f}")

                # Labels for contrastive loss (each example is similar to itself)
                labels = torch.arange(batch_size, device=device)

                # Compute symmetric contrastive loss
                loss_i = torch.nn.functional.cross_entropy(logits, labels)
                loss_t = torch.nn.functional.cross_entropy(logits.t(), labels)
                loss = (loss_i + loss_t) / 2.0
                
                # Add temperature regularization
                temp_loss = temperature * 0.0001
                total_loss = loss + temp_loss

                # Print detailed loss information
                print(f"Loss components - i: {loss_i.item():.4f}, t: {loss_t.item():.4f}, temp: {temp_loss.item():.4f}, total: {total_loss.item():.4f}")
                
                # Return loss and outputs
                if return_outputs:
                    return total_loss, outputs
                return total_loss

        # Initialize trainer
        trainer = ImprovedCLIPTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=self.collate_fn
        )

        # Start training
        self.logger.info("Starting training...")
        trainer.train()

        # Save final model
        self.save_model()

    def save_model(self, path: Optional[str] = None):
        """Save the trained model."""
        save_path = path or os.path.join(self.config["output"]["output_dir"], "final_model")
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.logger.info(f"Model saved to {save_path}")

    @staticmethod
    def collate_fn(examples):
        """Collate function for batch preparation."""
        if not examples:
            return {}
            
        # Initialize the output dictionary with the first example's keys
        output_dict = {key: [] for key in examples[0].keys()}
        
        # Collect tensors from all examples
        for example in examples:
            for key, value in example.items():
                if isinstance(value, torch.Tensor):
                    # Ensure the tensor is at least 1D
                    if value.dim() == 0:
                        value = value.unsqueeze(0)
                    output_dict[key].append(value)
                else:
                    tensor_value = torch.tensor(value)
                    if tensor_value.dim() == 0:
                        tensor_value = tensor_value.unsqueeze(0)
                    output_dict[key].append(tensor_value)
        
        # Stack tensors for each key and ensure proper dimensions
        batched_dict = {}
        for key, values in output_dict.items():
            if values[0].dim() >= 1:
                # For tensors that already have sequence dimension
                batched_dict[key] = torch.stack(values)
            else:
                # For scalar tensors
                batched_dict[key] = torch.stack([v.unsqueeze(0) if v.dim() == 0 else v for v in values])
        
        return batched_dict