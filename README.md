# TFLite Metadata Writer — Python 3.10+ fix

`tflite-support` was removed from PyPI for Python 3.10+.
This repo ships a **pure-Python shim** rebuilt from the original source so the Metadata Writer API works again — no binary dependencies.

## What's included

| Path | What |
|------|------|
| `tflite_support/` | Pure-Python shim (schema files + metadata writers, all imports fixed) |
| `examples/image_segmenter_metadata.py` | Image segmenter example |
| `examples/metadata_writer_tutorial.ipynb` | Full tutorial notebook (works on Colab) |

## Setup

```bash
# 1. Clone
git clone <this-repo>
cd <this-repo>

# 2. Install the only dependency
pip install flatbuffers
```

## Usage

### Image segmenter (CLI)

```bash
# Download test model
curl -L https://github.com/tensorflow/tflite-support/raw/master/tensorflow_lite_support/metadata/python/tests/testdata/image_segmenter/deeplabv3.tflite -o deeplabv3.tflite
curl -L https://github.com/tensorflow/tflite-support/raw/master/tensorflow_lite_support/metadata/python/tests/testdata/image_segmenter/labelmap.txt -o deeplabv3_labels.txt

# Run
python examples/image_segmenter_metadata.py \
    --model deeplabv3.tflite \
    --labels deeplabv3_labels.txt \
    --output deeplabv3_metadata.tflite
```

### Other writers (Python)

```python
import sys
sys.path.insert(0, ".")  # add repo root to path

# Image classifier
from tflite_support.metadata_writers import image_classifier, writer_utils
writer = image_classifier.MetadataWriter.create_for_inference(
    writer_utils.load_file("model.tflite"), [127.5], [127.5], ["labels.txt"])
writer_utils.save_file(writer.populate(), "model_metadata.tflite")

# Object detector
from tflite_support.metadata_writers import object_detector
writer = object_detector.MetadataWriter.create_for_inference(
    writer_utils.load_file("model.tflite"), [127.5], [127.5], ["labels.txt"])

# Audio classifier
from tflite_support.metadata_writers import audio_classifier, metadata_info
writer = audio_classifier.MetadataWriter.create_for_inference(
    writer_utils.load_file("model.tflite"), 16000, 1, ["labels.txt"])
```

## How this shim was built

The original `tflite-support` package had two problems:
1. **Removed from PyPI** — no wheels for Python 3.10+
2. **`metadata_writers/` missing from the wheel** — this subpackage was never published

Fix:
- `metadata_schema_py_generated.py` / `schema_py_generated.py` — extracted from the last published `tflite-support==0.4.4` wheel (pure-Python FlatBuffers, no changes needed)
- `metadata_writers/` — cloned from [tflite-support source](https://github.com/tensorflow/tflite-support), all `tensorflow_lite_support.*` imports rewritten to `tflite_support.*`
- `metadata.py` — same source clone + C-extension calls stubbed (`_pywrap_metadata_version`, `_pywrap_flatbuffers`)

> **Note:** `get_metadata_json()` (display only) is unavailable without C extensions. `populate()` (the actual write operation) works fully.

## Tested on

- Python 3.11.5 / Windows 11
- Python 3.12 / Google Colab
