# GhostQuant: The Ghost that Appears Only After Quantization

This repository contains the official implementation for the paper **"GhostQuant: The Ghost that Appears Only After Quantization"**.

GhostQuant is a quantization-aware backdoor attack framework for Large Vision-Language Models (LVLMs). It implants a "ghost" backdoor that remains dormant when the model is used in full precision but activates immediately once the model is quantized (e.g., to INT8/FP4/NF4) for deployment.

## 🚀 Quick Start

Follow the steps below to set up the environment and reproduce the **Caption Hijack** attack on Qwen3VL-2B.

### 1. Environment Setup

Clone the repository and install the required dependencies:

```bash
conda env create -f environment.yml
conda activate ghostquant
```

Or use pip in an existing Python 3.11 environment:

```bash
pip install -r requirements.txt
```

### 2. Model Preparation

The scripts automatically load **Qwen3VL-2B-Instruct** from Hugging Face. No manual download needed.

If you already have a local copy, specify its path with `--model`:

```bash
python scripts/run_qwen3vl2b_static3.py --model /path/to/your/model
```

### 3. Data Preparation

We provide the Caption Hijack dataset in the `data/task1/` folder:

- **train.json** (2,800 samples): Training data for cascaded attack
- **validation.json** (200 samples): Validation during training
- **test.json** (200 samples): Final attack evaluation

This dataset enables you to reproduce the core attack behavior where the model generates captions containing the word `banana` only after quantization.


### 4. Training

**Three-stage pipeline (FP4 → NF4 → INT8):**

```bash
python scripts/run_qwen3vl2b_static3.py \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --output-dir outputs/qwen3vl2b_static3 \
  --gpu 0
```

The final poisoned model is saved to `outputs/qwen3vl2b_static3/final_model/`.

**Twelve-stage pipeline (all quantizers):**

For full reproduction including GGUF and AWQ quantizers, install the AWQ environment and llama.cpp:

```bash
conda env create -f environment-awq.yml

python scripts/run_qwen3vl2b_cascade12.py \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --output-dir outputs/qwen3vl2b_cascade12 \
  --llama-cpp /path/to/llama.cpp \
  --python /path/to/envs/ghostquant/bin/python \
  --awq-python /path/to/envs/ghostquant-awq/bin/python \
  --gpu 0
```

### 5. Evaluation

Attack success rate (tASR) is automatically evaluated after training completes. Results are saved in `tasr_summary.json`.


## 📂 Repository Structure

```text
GhostQuant/
├── configs/              # Pipeline definitions
├── data/
│   ├── task1/           # Caption Hijack dataset and images
│   └── calibration/     # AWQ calibration data
├── ghostquant/          # Core training and quantization code
├── scripts/             # Entry points for 3-stage and 12-stage pipelines
├── tests/               # Data validation
└── environment*.yml
```
