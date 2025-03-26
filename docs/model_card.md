---
tags:
- flux
- stable-diffusion
- text-to-image
- lora
- flux dev
- diffusers
- impressionism
library_name: diffusers
pipeline_tag: text-to-image
base_model: black-forest-labs/FLUX.1-dev
widget:
- text: >-
    An impressionist painting portrays a vast landscape with gently rolling
    hills under a radiant sky. Clusters of autumn trees dot the scene, rendered
    with loose, expressive brushstrokes and a palette of warm oranges, deep
    greens, and soft blues, creating a sense of tranquil, natural beauty
  output:
    url: images/example_jl6x0209w.png
---

# FLUX.1-dev Impressionism fine-tuning with LoRA

This is a LoRA fine-tuning of the FLUX.1 model trained on a curated dataset of impressionist paintings from WikiArt.

## Training Process & Results

### Training Environment
- GPU: NVIDIA A100
- Training Duration: ~1 hour for 1000 steps
- Training Notebook: [Google Colab Notebook](https://colab.research.google.com/drive/1G9k6iwSGKXmA32ok4zOPijFUFwBAZ9aB?usp=sharing)
- Training Framework: [AI-Toolkit](https://github.com/ostris/ai-toolkit)

## Training Progress Visualization

### Training Progress Grid
![Training Progress Grid](./images/sample_grid_annotated.png)
*4x6 grid showing model progression across different prompts (rows) at various training steps (columns: 0, 200, 400, 600, 800, 1000)*

### Step-by-Step Evolution
![Training Progress Animation](./images/prompt_0.gif)
*Evolution of the model's output for the prompt: "An impressionist painting portrays a vast landscape with gently rolling hills under a radiant sky. Clusters of autumn trees dot the scene, rendered with loose, expressive brushstrokes and a palette of warm oranges, deep greens, and soft blues, creating a sense of tranquil, natural beauty" (Steps 0-1000, sampled every 100 steps)*


### Base vs Fine-tuned
![Base model vs Fine-tuned](./images/base_vs_fine_tuned.png)
*Left side is the base model and right side is this fine-tuned model*


### Current Results & Future Improvements
The most notable improvements are observed in landscape generation, which can be attributed to:
- Strong representation of landscapes (30%) in the training dataset
- Inherent structural similarities in impressionist landscape paintings
- Clear patterns in color usage and brushstroke techniques

Future improvements will focus on:
- Experimenting with different LoRA configurations and ranks
- Fine-tuning hyperparameters for better convergence
- Improving caption quality and specificity
- Extending training duration beyond 1000 steps
- Developing custom training scripts for more granular control

While the current implementation uses the [AI-Toolkit](https://github.com/ostris/ai-toolkit), future iterations will involve developing custom training scripts to gain deeper insights into model configuration and behavior.

## Dataset
The model was trained on the [WikiArt Impressionism Curated Dataset](https://huggingface.co/datasets/dolphinium/wikiart-impressionism-curated), which contains 1,000 high-quality Impressionist paintings with the following distribution:

- Landscapes: 300 images (30%)
- Portraits: 300 images (30%)
- Urban Scenes: 200 images (20%)
- Still Life: 200 images (20%)

## Model Details
- Base Model: [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- LoRA Rank: 16
- Training Steps: 2000
- Resolution: 512-1024px

## Usage

To run code 4-bit with quantization check out this [Google Colab Notebook](https://colab.research.google.com/drive/1dnCeNGHQSuWACrG95rH4TXPgXwNNdTh-?usp=sharing). 

On Google Colab the cheapest way to run code is acquiring a T4 with high-ram if I am not wrong :) 

Also thanks to providers original notebook to run code 4-bit with quantization.
[Original Colab Notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Flux/Run_Flux_on_an_8GB_machine.ipynb) :

## License
This model inherits the license of the base [FLUX.1 model](https://huggingface.co/black-forest-labs/FLUX.1-dev) and the [WikiArt](https://huggingface.co/datasets/huggan/wikiart) dataset.