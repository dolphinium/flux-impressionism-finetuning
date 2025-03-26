# Technical Context: Flux Impressionism Fine-Tuning

## Technologies Used

### Core Technologies
- **Flux.1 Dev**: Base model for fine-tuning
- **AI-Toolkit**: Primary training framework
- **Python**: Primary programming language
- **PyTorch**: Deep learning framework
- **Hugging Face Diffusers**: Library for working with diffusion models
- **Transformers**: For handling text encoders in the model
- **PEFT**: Parameter-Efficient Fine-Tuning library
- **Google Gemini API**: For advanced image captioning

### Image Captioning System
- **Components**:
  - Gemini API Integration
  - Async Processing Pipeline
  - Error Handling System
  - Checkpoint Management
- **Features**:
  - Rate Limiting: 15 requests/minute per key
  - Daily Quota: 1500 requests per key
  - Batch Processing: Configurable batch size
  - Automatic API Key Rotation
  - Progress Tracking & Logging

### Training Environment
- **Hardware**:
  - GPU: NVIDIA A100
  - VRAM: ~11GB required
  - Training Duration: ~1 hour per 1000 steps
- **Configuration**:
  - LoRA Rank: 16
  - Resolution: 512-1024px
  - Training Steps: 1000
  - Framework: AI-Toolkit

### Development Environment
- **Google Colab Pro**: Primary development and training environment (GPU access)
- **Git/GitHub**: Version control and project management
- **Jupyter Notebooks**: For interactive development and experimentation
- **Hugging Face Hub**: For model storage and versioning
- **Hugging Face Spaces**: For deployment and demonstration

### Data Management
- **WikiArt Dataset**: Source of Impressionist paintings
- **Pandas/NumPy**: For data manipulation and processing
- **Pillow/OpenCV**: For image processing
- **Datasets**: Hugging Face library for dataset management
- **JSON**: For structured data storage
- **aiohttp**: For async HTTP operations
- **asyncio**: For concurrent processing

### Evaluation & Visualization
- **Training Progress Grid**: 4x6 visualization of model progression
- **Step Evolution**: Animation of training progress
- **Base vs Fine-tuned**: Comparison visualization
- **Tensorboard**: For training metrics
- **Matplotlib/Seaborn**: For result visualization
- **Gradio**: For creating interactive demos

## Development Setup
1. **Local Environment**:
   - Python 3.8+ with required packages
   - Git for version control
   - VSCode or similar editor for local development

2. **Cloud Environment**:
   - Google Colab Pro with GPU runtime
   - Connected to GitHub repository
   - Connected to Hugging Face account

3. **Deployment Environment**:
   - Hugging Face Hub for model hosting
   - Hugging Face Spaces with GPU for interactive demo

## Technical Dependencies
```
torch>=1.12.0
transformers>=4.25.1
diffusers>=0.14.0
accelerate>=0.16.0
datasets>=2.9.0
peft>=0.3.0
gradio>=3.16.0
pillow>=9.3.0
matplotlib>=3.6.3
tensorboard>=2.11.2
google-generativeai>=0.3.0
aiohttp>=3.9.0
python-dotenv>=1.0.0
```

## API Integrations
1. **Hugging Face Hub API**: For model upload and versioning
2. **Hugging Face Spaces API**: For deployment
3. **Google Gemini API**: For image captioning
4. **WikiArt API**: For dataset access (if available)

## Technical Constraints
1. **Training Limitations**:
   - Training duration currently limited to 1000 steps
   - LoRA rank fixed at 16
   - Resolution range: 512-1024px
   - AI-Toolkit framework constraints

2. **API Limitations**:
   - Gemini API rate limits (15 req/min)
   - Daily quota per API key (1500 req/day)
   - Need for multiple API keys
   - Response time variability

3. **Storage Considerations**:
   - Model checkpoints (~3MB per LoRA weights)
   - Training visualization artifacts
   - Dataset storage requirements
   - Caption data management

4. **Deployment Considerations**:
   - Model size optimization for faster inference
   - Balancing quality with performance
   - Quantization options for different hardware:
     - int8 + bf16: ~18GB VRAM
     - int4 + bf16: ~13GB VRAM
     - NF4/int2 + bf16: ~9GB VRAM 