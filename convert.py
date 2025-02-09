import argparse

from mlx_lm import convert


def convert_hf_to_mlx_model(
    model_id: str,
    quantize: bool,
    quantize_level: int
):

    if quantize:
        upload_repo = f"mlx-community/{model_id}-{quantize_level}bit"

        convert(
            model_id,
            quantize=True,
            q_bits=quantize_level,
            upload_repo=upload_repo
        )
    else:
        upload_repo = f"mlx-community/{model_id}"

        convert(model_id, upload_repo=upload_repo)


def parse_args():
    parser = argparse.ArgumentParser()
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_id: str = args.model
    quantize: bool = args.quantize
    quantize_level: int = args.quantize_level

    convert_hf_to_mlx_model(model_id, quantize, quantize_level)
