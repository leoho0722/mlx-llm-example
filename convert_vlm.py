import argparse
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub.constants import HF_HUB_CACHE
from mlx_vlm import convert

load_dotenv(override=True)


def convert_hf_to_mlx_model(
    model_id: str,
    quantize: bool,
    quantize_level: int,
    upload: bool,
    mlx_path: str,
    verbose: bool
):
    repo_id = model_id.split("/")[-1]

    upload_repo: Optional[str] = None
    if upload:
        if quantize:
            upload_repo = f"mlx-community/{repo_id}-{quantize_level}bit"
        else:
            upload_repo = f"mlx-community/{repo_id}"

    if verbose:
        print(f"Upload repo: {upload_repo}")

    if quantize:
        convert(
            model_id,
            mlx_path,
            quantize=True,
            q_bits=quantize_level,
            upload_repo=upload_repo
        )
    else:
        convert(model_id, mlx_path, upload_repo=upload_repo)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face Model to MLX Model Format using the Apple MLX Framework."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        help="Path to the model",
        required=True
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize the model",
    )
    parser.add_argument(
        "--quantize_level",
        type=int,
        help="Quantize level, default is 4bit",
        default=4,
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the model to Hugging Face mlx-community organization"
    )
    parser.add_argument(
        "--mlx_path",
        type=str,
        help="Path to store the mlx model",
        required=True
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose mode"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_id: str = args.model
    quantize: bool = args.quantize
    quantize_level: int = args.quantize_level
    upload: bool = args.upload
    mlx_path: str = args.mlx_path
    verbose: bool = args.verbose

    if verbose:
        print(f"Model: {model_id}")
        print(f"Quantize: {quantize}")
        if quantize:
            print(f"Quantize level: {quantize_level}bit")
        print(f"Upload to Hugging Face: {upload}")
        print(f"Verbose mode: {verbose}")
        print()

    convert_hf_to_mlx_model(
        model_id,
        quantize,
        quantize_level,
        upload,
        mlx_path,
        verbose
    )
