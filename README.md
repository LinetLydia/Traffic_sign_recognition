# Traffic Sign Recognition (GTSRB)

Local, token-free project that trains a CNN to classify German traffic signs using the GTSRB dataset.

## Dataset
- Source: German Traffic Sign Recognition Benchmark (GTSRB).
- How to use: Manually download “Final_Training/Images” (and optional “Final_Test/Images”), unzip into `data/GTSRB/`.

## Quick Start
1. Create the project structure (see repo tree in README).
2. Place GTSRB data under `data/GTSRB/Final_Training/Images/`.
3. Install requirements: `pip install -r requirements.txt`
4. Open the notebook and run all cells: `notebook/traffic_sign_recognition.ipynb`

## Results (example targets)
- Top-1 accuracy ≥ 95% on validation with a lightweight CNN + augmentation.

## Files
- `notebook/traffic_sign_recognition.ipynb`: end-to-end training & evaluation.
- `models/`: saved best model weights.
# Traffic_sign_recognition
