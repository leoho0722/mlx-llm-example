# MLX LLM Example

LLM model inference on Apple Silicon Mac using the Apple MLX Framework.

- [Apple MLX Framework Official GitHub Repository](https://github.com/ml-explore/mlx-examples)
- [MLX Community Hugging Face Hub](https://huggingface.co/mlx-community)

## Environment

### Hardware

- Apple MacBooke Pro (13-inch, M2, 2022)
- Apple M2 chips (8 cores CPU, 10 cores GPU)
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
| `--prompt`      | `str`  | `What is the largest country in the world?` | Prompt for the LLM model   |
| `--max_tokens`  | `int`  | `512`                                       | Maximum tokens to generate |
| `--verbose`     | `bool` |                                             | Verbose mode               |

```bash
source .venv/bin/activate

# Run the stream inference with default values
python inference.py

# Run the stream inference with verbose mode
python inference.py --verbose

# Run the stream inference with custom model
python inference.py --model "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx"

# Run the stream inference with custom prompt
python inference.py --prompt "What is the capital of France?"

# Run the stream inference with custom max tokens
python inference.py --max_tokens 1024
```

### Convert Hugging Face Model to MLX Model Format

| Args               | Type   | Default | Description            |
| ------------------ | ------ | ------- | ---------------------- |
| `-m`, `--model`    | `str`  |         | Path to the model      |
| `--quantize`       | `bool` |         | Whether Quantize model |
| `--quantize_level` | `int`  | 4       | Quantize level (bits)  |
| `--verbose`        | `bool` |         | Verbose mode           |

```bash
source .venv/bin/activate

# Convert Hugging Face model to MLX model format
python convert.py --model "google/gemma-2-9b-it"

# Convert Hugging Face model to MLX model format with verbose mode
python convert.py --model "google/gemma-2-9b-it" --verbose

# Convert Hugging Face model to MLX model format with quantization
python convert.py --model "google/gemma-2-9b-it" --quantize

# Convert Hugging Face model to MLX model format with quantization and custom quantize level
python convert.py --model "google/gemma-2-9b-it" --quantize --quantize_level 8
```
