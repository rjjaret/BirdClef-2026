# My first Kaggle Challenge!
# BirdCLEF 2026 Classification

Training and inference workflows for BirdCLEF 2026 bird-call classification using TensorFlow/PyTorch and notebook-driven experimentation. This was my first challenge. My best score was an AUC ~ .75. Not too bad for a first timer.

## Repository Layout

- `birdclef_utils/`: reusable audio processing, dataset, model, and metrics modules
- `BirdClef-2026-Classification-v6-Py.ipynb`: main training notebook
- `BirdClef-2026-Inference-Submission_v3.ipynb`: inference/submission notebook
- `artifacts/model_labels_v2.json`: label mapping metadata
- `requirements.txt`: Python dependencies

Ignored from Git by default:
- large datasets (`data/`), caches (`cache/`), model checkpoints (`checkpoints/`)
- local virtual environments (such as `.venv-bc/`)

## Environment Setup (Python 3.12)

```bash
python3.12 -m venv .venv-bc
source .venv-bc/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
