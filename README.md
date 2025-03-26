# Flux Impressionism: Fine-Tuning for Artistic Style Transfer

![Impressionism Style Transfer](https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Claude_Monet%2C_Impression%2C_soleil_levant.jpg/400px-Claude_Monet%2C_Impression%2C_soleil_levant.jpg)

*Image: Claude Monet, "Impression, Sunrise" (1872) - Public Domain*

## Project Overview

This project fine-tunes the Flux.1 Dev model to generate images in the style of Impressionist paintings. By training on the WikiArt dataset's Impressionism subset, we create a specialized model that captures the distinctive aesthetic qualities of Impressionist art while maintaining the flexibility of a general text-to-image model.

## Training Results & Visualization

### Training Progress Grid
![Training Progress Grid](docs/images/sample_grid_annotated.png)
*4x6 grid showing model progression across different prompts (rows) at various training steps (columns: 0, 200, 400, 600, 800, 1000)*

### Step-by-Step Evolution
![Training Progress Animation](docs/images/prompt_0.gif)
*Evolution of the model's output for a landscape prompt across training steps*

### Base vs Fine-tuned Comparison
![Base model vs Fine-tuned](docs/images/base_vs_fine_tuned.png)
*Comparison between base model (left) and fine-tuned model (right)*

### Training Environment
- GPU: NVIDIA A100
- Training Duration: ~1 hour for 1000 steps
- Training Framework: [AI-Toolkit](https://github.com/ostris/ai-toolkit)
- Training Notebook: [Google Colab Notebook](https://colab.research.google.com/drive/1G9k6iwSGKXmA32ok4zOPijFUFwBAZ9aB?usp=sharing)

### Current Status
- ✅ Initial fine-tuning completed (1000 steps)
- ✅ Model deployed to Hugging Face Hub: [FLUX.1-dev-wikiart-impressionism](https://huggingface.co/dolphinium/FLUX.1-dev-wikiart-impressionism)
- ✅ Training visualization and progress tracking implemented
- ✅ Basic inference implementation
- ✅ Implemented automated caption generation with Gemini API
- 🔄 Developing custom fine-tuning implementation

## Features

- Fine-tuned Flux.1 Dev model specialized in Impressionist styles
- Comprehensive training visualization and progress tracking
- Advanced image captioning pipeline using Google's Gemini API
- Interactive demo on Hugging Face Spaces
- Comprehensive documentation of the fine-tuning process
- Visual showcase demonstrating the model's capabilities
- Reusable code for similar fine-tuning projects

## Dataset

I've created a carefully curated subset of the WikiArt dataset, specifically focused on Impressionist paintings. The dataset is available on Hugging Face Hub:

🤗 [wikiart-impressionism-curated](https://huggingface.co/datasets/dolphinium/wikiart-impressionism-curated)

### Dataset Features
- 1,000 high-quality Impressionist paintings
- Balanced genre distribution:
  - Landscapes (30%)
  - Portraits (30%)
  - Urban Scenes (20%)
  - Still Life (20%)
- Quality criteria:
  - Minimum dimension: 512px
  - Maximum aspect ratio: 2:1
  - Quality-controlled file sizes
  - Verified Impressionist style 

Check dataset curation notebook [here](https://github.com/dolphinium/flux-impressionism-finetuning/blob/main/notebooks/dataset_curation.ipynb)

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- Hugging Face account (free or Pro)
- Google Colab Pro (for training with GPU)

### Installation

```bash
git clone https://github.com/dolphinium/flux-impressionism-finetuning.git
cd flux-impressionism-finetuning
pip install -r requirements.txt
```

### Project Structure

```
flux-impressionism-finetuning/
├── notebooks/             # Jupyter notebooks for training and inference
├── src/                   # Reusable Python modules
├── data/                  # Dataset management scripts
├── docs/                  # Documentation
├── results/               # Generated images and evaluation results
├── memory-bank/           # Project documentation and context
├── requirements.txt       # Python dependencies
├── ROADMAP.md             # Project roadmap and timeline
└── README.md              # This file
```

## Development Process

This project follows a structured approach to fine-tuning:

1. **Research & Planning**: ✅ Understanding Impressionist style and fine-tuning techniques
2. **Data Preparation**: ✅ Processing the WikiArt Impressionism dataset
3. **Initial Fine-tuning**: ✅ First iteration using ai-toolkit (1000 steps)
4. **Dataset Enhancement**: ✅ Implemented automated caption generation with Gemini API
5. **Custom Implementation**: 🔄 Developing in-house fine-tuning pipeline
6. **Evaluation**: Assessing quality through objective and subjective metrics
7. **Deployment**: Making the model accessible through Hugging Face

### Image Captioning Pipeline

The project includes a robust image captioning system built with Google's Gemini API:

#### Features
- **Intelligent Captioning**: Genre-aware prompting system for accurate art descriptions
- **Efficient Processing**:
  - Rate limiting and API key rotation
  - Batch processing support
  - Checkpoint system for resuming interrupted runs
- **Error Handling**:
  - Comprehensive logging system
  - Automatic retry mechanism for failed captions
  - Progress tracking and status reporting
- **Integration**:
  - Seamless HuggingFace datasets integration
  - JSON-based data management
  - Easy to extend and modify

#### Pipeline Components
- `pipeline.py`: Main captioning system with Gemini integration
- `fix_failed_captions.py`: Retry mechanism for failed captions
- Checkpoint and logging system for reliable processing
- Support for multiple API keys and rate limit management

### Upcoming Improvements
- Fine-tuning with enhanced captions from Gemini API
- Adding trigger words for better style control
- Developing custom fine-tuning pipeline for better control
- Enhancing evaluation metrics and monitoring

For detailed timeline, see [ROADMAP.md](ROADMAP.md).

## Model Usage

Once deployed, you can use the model in two ways:

### Via Hugging Face Hub

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "black-forest-labs/FLUX.1-dev"
lora_model_path = "dolphinium/FLUX.1-dev-wikiart-impressionism"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA weights
pipe.unet.load_attn_procs(lora_model_path)

# Generate image
prompt = "An impressionist painting portrays a vast landscape with gently rolling hills under a radiant sky. Clusters of autumn trees dot the scene, rendered with loose, expressive brushstrokes and a palette of warm oranges, deep greens, and soft blues, creating a sense of tranquil, natural beauty"
image = pipe(prompt).images[0]
image.save("impressionist_landscape.png")
```

### Via Google Colab

For running the model with 4-bit quantization (reduced memory usage):
- [Inference Notebook](https://colab.research.google.com/drive/1dnCeNGHQSuWACrG95rH4TXPgXwNNdTh-?usp=sharing)
- [Original Implementation](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Flux/Run_Flux_on_an_8GB_machine.ipynb)

Note: Using a T4 GPU with high-RAM runtime is recommended for cost-effective inference.

### Via Interactive Demo(NOT PUBLISHED YET)

Visit our [Hugging Face Space](https://huggingface.co/spaces/dolphinium/flux-impressionism-demo) for an interactive demo.

## Documentation

Comprehensive documentation is available in the [docs/](docs/) directory, including:

- Technical methodology
- Training process and parameters
- Evaluation results
- Usage guides

## Contributing

This project is currently in development. Contributions and suggestions are welcome through issues and pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for their diffusers library and infrastructure
- [WikiArt](https://www.wikiart.org/) for the dataset
- The creators of the Flux.1 Dev model

