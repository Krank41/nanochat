# 🚀 Forked NanoChat

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd nanochat
```

### 2. Download the Dataset
Download the default ClimbMix dataset with 4 parallel workers:
```bash
python -m nanochat.dataset -n 4
```

### 3. Train Custom Tokenizer
Train a tokenizer on your dataset:
```bash
python -m scripts.tok_train
```

## 🎯 Training

### Base Model Training

Train a base model with configurable depth and batch size. Supports both `float16` and `bfloat16` dtypes.

**Float16 Training (for older GPUs like RTX 2080):**
```bash
NANOCHAT_DTYPE=float16 python -m scripts.base_train \
    --depth=4 \
    --device-batch-size=8 \
    --max-seq-len=512 \
    --model-tag="d4"
```

**BFloat16 Training (for newer GPUs like A100):**
```bash
NANOCHAT_DTYPE=bfloat16 python -m scripts.base_train \
    --depth=4 \
    --device-batch-size=8 \
    --max-seq-len=512 \
    --model-tag="d4"
```

### Supervised Fine-Tuning (SFT)

Fine-tune your model on conversation data:

```bash
# Download identity conversations dataset
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# Run SFT
python -m scripts.chat_sft \
    --device-batch-size=4 \
    --model-tag="d4" \
    --model-step=<your-checkpoint-step>
```

## 📊 Evaluation

### Core Evaluation
Evaluate your model on core benchmarks:
```bash
python -m scripts.base_eval \
    --device-batch-size=1 \
    --eval core \
    --max-per-task=100
```

## 💬 Interactive Chat

Test your trained model interactively:

```bash
# Default dtype (auto-detected based on GPU)
python -m scripts.chat_cli -p "hello"

# Force specific dtype
NANOCHAT_DTYPE=float32 python -m scripts.chat_cli -p "hello"
NANOCHAT_DTYPE=float16 python -m scripts.chat_cli -p "hello"
```

### Data Types
- **`bfloat16`**: Best for modern GPUs (SM 8.0+, e.g., A100, H100)
- **`float16`**: For older GPUs (SM 7.5, e.g., RTX 2080)

### Model Depths
- `--depth=1`: ~50M parameters
- `--depth=4`: ~200M parameters
- `--depth=12`: ~600M parameters
