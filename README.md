# Flux Impressionism: Fine-Tuning for Artistic Style Transfer

![Impressionism Style Transfer](https://upload.wikimedia.org/wikipedia/commons/thumb/5/54/Claude_Monet%2C_Impression%2C_soleil_levant.jpg/400px-Claude_Monet%2C_Impression%2C_soleil_levant.jpg)

*Image: Claude Monet, "Impression, Sunrise" (1872) - Public Domain*

## Project Overview

This project fine-tunes the Flux.1 Dev model to generate images in the style of Impressionist paintings. By training on the WikiArt dataset's Impressionism subset, we create a specialized model that captures the distinctive aesthetic qualities of Impressionist art while maintaining the flexibility of a general text-to-image model.

## Features

- Fine-tuned Flux.1 Dev model specialized in Impressionist styles
- Interactive demo on Hugging Face Spaces
- Comprehensive documentation of the fine-tuning process
- Visual showcase demonstrating the model's capabilities
- Reusable code for similar fine-tuning projects

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

1. **Research & Planning**: Understanding Impressionist style and fine-tuning techniques
2. **Data Preparation**: Processing the WikiArt Impressionism dataset
3. **Model Development**: Fine-tuning Flux.1 Dev with optimized parameters
4. **Evaluation**: Assessing quality through objective and subjective metrics
5. **Deployment**: Making the model accessible through Hugging Face

For detailed timeline, see [ROADMAP.md](ROADMAP.md).

## Model Usage

Once deployed, you can use the model in two ways:

### Via Hugging Face Hub

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("dolphinium/flux-impressionism-v1")
image = pipeline("A landscape with trees by a river").images[0]
image.save("impressionist_landscape.png")
```

### Via Interactive Demo

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