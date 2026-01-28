# GhostQuant: The Ghost that Appears Only After Quantization of LVLMs

This repository contains the official implementation for the paper **"GhostQuant: The Ghost that Appears Only After Quantization of LVLMs"**.

GhostQuant is a quantization-aware backdoor attack framework for Large Vision-Language Models (LVLMs). It implants a "ghost" backdoor that remains dormant when the model is used in full precision (FP32/BF16) but activates immediately once the model is quantized (e.g., to INT4/FP4) for deployment.

## 🚀 Quick Start

Follow the steps below to set up the environment and reproduce a simple demonstration of the **Caption Hijack** scenario.

### 1. Environment Setup

Clone the repository and install the required dependencies:

```
pip install -r requirements.txt
```

### 2. Model Preparation

Please download the base model **Qwen3VL-2B** from Hugging Face [here](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct).

Rename the downloaded model folder to Qwen3VL-2B and place it inside model/.

The directory structure should look like this:
```
GhostQuant/
├── model/
│   └── Qwen3VL-2B/  <-- Put model files here
├── dataset/
│   ├── train_dataset.json
│   └── valid_dataset.json
├── run.sh
├── requirements.txt
└── ...
```

### 3. Data Preparation
We provide a partial dataset for the Caption Hijack scenario in the dataset/ folder.

The provided demonstration dataset contains 3,000 samples in total: 500 Poisoned Samples: Images with triggers for the backdoor. 2,500 Benign Samples: Clean data sampled from the MIMIC-IT LADD subset.

Although smaller than the full dataset used in the paper, this subset is sufficient to reproduce the core experimental phenomenon of GhostQuant.

### 4. Training
To run the attack, simply execute the provided shell script:

```
bash run.sh
```
