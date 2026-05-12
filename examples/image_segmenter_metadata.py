"""
Create TFLite metadata for an image segmenter model.

Usage:
    python create_segmenter_metadata.py
    python create_segmenter_metadata.py --model my_model.tflite --labels my_labels.txt --output out.tflite
"""

import sys
import os
import argparse

# Add the local tflite_support shim to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tflite_support.metadata_writers import image_segmenter
from tflite_support.metadata_writers import writer_utils


def create_metadata(model_path: str, label_file: str, output_path: str,
                    norm_mean: float = 127.5, norm_std: float = 127.5) -> None:
    writer = image_segmenter.MetadataWriter.create_for_inference(
        writer_utils.load_file(model_path),
        [norm_mean],
        [norm_std],
        [label_file],
    )

    # get_metadata_json() requires a C extension (_pywrap_flatbuffers) not available
    # in this pure-Python setup — skip if unavailable
    try:
        print(writer.get_metadata_json())
    except AttributeError:
        print("[note] JSON display skipped (requires C extensions); metadata binary OK")

    model_with_metadata = writer.populate()
    writer_utils.save_file(model_with_metadata, output_path)
    print(f"Saved: {output_path}  ({len(model_with_metadata):,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add metadata to a TFLite image segmenter")
    parser.add_argument("--model",  default="deeplabv3.tflite",          help="Input .tflite model")
    parser.add_argument("--labels", default="deeplabv3_labels.txt",      help="Label file")
    parser.add_argument("--output", default="deeplabv3_metadata.tflite", help="Output .tflite path")
    parser.add_argument("--norm-mean", type=float, default=127.5)
    parser.add_argument("--norm-std",  type=float, default=127.5)
    args = parser.parse_args()

    create_metadata(args.model, args.labels, args.output, args.norm_mean, args.norm_std)
