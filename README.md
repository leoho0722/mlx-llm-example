# MLX LLM Example

LLM model inference using the Apple MLX Framework.

[Apple MLX Framework Official GitHub Repository](https://github.com/ml-explore/mlx-examples)

## Environment

### Hardware

- Apple MacBooke Pro (13-inch, M2, 2022)
- Apple M2 (8 cores CPU, 10 cores GPU)
- 16GB RAM, 256GB SSD
- macOS Sequoia 15.3

### Software

- Python 3.10.16
- mlx-lm 0.21.4

## Installation

### Create Virtual Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

```bash
pip install -U pip setuptools pip-autoremove
pip install -r requirements.txt
```

## Run

### Streaming Inference

| Args            | Type   | Default                                     | Description                |
| --------------- | ------ | ------------------------------------------- | -------------------------- |
| `-m`, `--model` | `str`  | `mlx-community/gemma-2-9b-it-4bit`          | Path to the model          |
| `--prompt`      | `str`  | `What is the largest country in the world?` | Prompt text for LLM model  |
| `--max_tokens`  | `int`  | `512`                                       | Maximum tokens to generate |
| `--verbose`     | `bool` |                                             | Verbose mode               |

```bash
source .venv/bin/activate

# Run the streaming inference
python streaming.py

# Run the streaming inference with verbose mode
python streaming.py --verbose

# Run the streaming inference with custom model
python streaming.py --model "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"

# Run the streaming inference with custom prompt
python streaming.py --prompt "What is the capital of France?"

# Run the streaming inference with custom max tokens
python streaming.py --max_tokens 256
```

### Convert Hugging Face Model to MLX Model Format

| Args               | Type   | Default | Description            |
| ------------------ | ------ | ------- | ---------------------- |
| `-m`, `--model`    | `str`  |         | Path to the model      |
| `--quantize`       | `bool` |         | Whether Quantize model |
| `--quantize_level` | `int`  | 4       | Quantize level (bits)  |

```bash
source .venv/bin/activate

# Convert Hugging Face model to MLX model format
python convert.py --model "google/gemma-2-9b-it"

# Convert Hugging Face model to MLX model format with quantization
python convert.py --model "google/gemma-2-9b-it" --quantize

# Convert Hugging Face model to MLX model format with quantization and custom quantize level
python convert.py --model "google/gemma-2-9b-it" --quantize --quantize_level 8
```
