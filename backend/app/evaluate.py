"""
AcousticSpace - Model Evaluation

Testing one file at a time gives a misleading picture — a single wrong
prediction doesn't tell you if the model is actually working. This
script evaluates the trained checkpoint over the FULL dataset and
reports accuracy, precision, recall, and a confusion matrix, which is
what actually tells you how good the model is.

Run:
  python -m app.evaluate
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.train import AudioLabelDataset, DATA_DIR
from app.model import AcousticSpaceBaseline

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "baseline_cnn.pt"


def evaluate():
    dataset = AudioLabelDataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = AcousticSpaceBaseline()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    tp = tn = fp = fn = 0  # positive class = "fake" (label 1)

    with torch.no_grad():
        for mel, rir, labels in loader:
            logits = model(mel, rir)
            preds = logits.argmax(dim=1)
            for p, y in zip(preds.tolist(), labels.tolist()):
                if p == 1 and y == 1:
                    tp += 1
                elif p == 0 and y == 0:
                    tn += 1
                elif p == 1 and y == 0:
                    fp += 1
                elif p == 0 and y == 1:
                    fn += 1

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\nEvaluated on {total} samples\n")
    print("Confusion matrix:")
    print(f"                 predicted genuine   predicted fake")
    print(f"  actual genuine        {tn:6d}             {fp:6d}")
    print(f"  actual fake           {fn:6d}             {tp:6d}")
    print()
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision (fake): {precision:.2%}   <- of clips flagged fake, how many really were")
    print(f"Recall (fake):    {recall:.2%}   <- of actually-fake clips, how many were caught")
    print(f"F1 score: {f1:.3f}")


if __name__ == "__main__":
    evaluate()