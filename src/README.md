# SCLCM: Simple Contrastive Learning based Convolutional Model for EEG Emotion Recognition with Data and Model Size Reduction

## Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.8+
- CUDA (for GPU acceleration)
- Other dependencies listed in `requirements.txt`

### Quick Setup
```bash
git clone https://github.com/sofi12321/SCLCM.git
cd SCLCM
pip install -r requirements.txt
mkdir model_weights
mkdir outputs
```

## Usage

### Basic Pipeline Execution
```python
from data.general.metadata import general_metadata, datasets_metadata
from utils.run_pipeline import run_pipeline

# Run pipeline with default parameters
run_pipeline(
    num_exp="exp1", 
    dataset_selection="your_dataset", 
    general_metadata=general_metadata, 
    dataset_metadata=datasets_metadata,
    emb_dim=128,
    channels_cut=None,
    in_features=None,
    pretraining_epochs=200,
    finetuning_epochs=200
)
```

### Jupyter Notebook Tutorial
A step-by-step tutorial is available in `tutorials/run_pipeline.ipynb` demonstrating:
- Data loading and preprocessing
- Model configuration
- Training process on 2 tasks (pre-text and downstream)
- Evaluation and visualization
- Model quantization

## Project Structure

```
SCLCM/
├── src/                         # Main source code
│   ├── data/                    # Data loading and processing
│   ├── models/                  # Model architectures
│   └── utils/                   # Utility functions
├── tutorials/                   # Tutorial notebooks
│   ├── run_pipeline.ipynb       # Step-by-step running example
│   └── data_analysis.ipynb      # Step-by-step data analysis
├── model_weights/               # Directory for saved models
├── outputs/                     # Directory for output visualizations
├── requirements.txt             # Python dependencies
├── LICENSE                      # License
└── README.md                    # This file
```

## Project Overview

Emotion recognition is a promising direction in various fields. Various sources can be used for emotion recognition, and although the external manifestations of emotions may be hidden, physiological signs such as electroencephalogram (EEG) remain unchanged. 

One promising approach that can cope with complex dependencies in EEG signals and a small percentage of labeled data is self-supervised learning (SSL). 

Here we propose an easy-to-understand and implement approach based on a convolutional encoder using SSL.

<p align="center">
  <img height="270" src="https://github.com/sofi12321/SCLCM/pipeline.png">
</p>

For the experiments, we used two datasets: SEED and DEAP. 

We used the following methods for feature extraction:
  - baseline correction;
  - spatial transformation (3D);
  - power spectral density (PSD);
  - differential entropy (DE);
  - differential asymmetry (DASM);
  - rational asymmetry (RASM);
  - differential caudality (DCAU).

During data analysis, we found that the largest inter-emotional differences for the 3 emotions in the SEED dataset are observed in channels T7 and T8.

<p align="center">
  <img height="270" src="https://github.com/sofi12321/SCLCM/eda.png">
</p>

The contrastive loss function is a binary cross-entropy based on cosine similarity with a temperature coefficient and weighting parameters for positive, hard and soft negative pairs.

In the proposed approach, each batch element must contain at least two different elements in this batch, forming positive and negative pairs with it, respectively.

A convolutional encoder with a subsequent classifier was used as a model. This architecture was chosen due to the simplicity of the architecture both in its understanding and in size, as well as relatively high performance.

<p align="center">
  <img height="270" src="https://github.com/sofi12321/SCLCM/model.png">
</p>

Since the model was trained using SSL we applied post-training static quantization with a symmetric linear mapping of the absolute maximum to reduce its size.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{SCLCM2023,
  author = {Sofia Shulepina, Danila Shulepin, Rustam Lukmanov},
  title = {SCLCM: EEG Processing for Emotion Recognition with Contrastive Learning and Model & Data Size Reduction},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sofi12321/SCLCM}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
