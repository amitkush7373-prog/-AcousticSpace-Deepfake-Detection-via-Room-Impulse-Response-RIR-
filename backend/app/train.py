
"""
AcousticSpace - Baseline Model Training
"""

from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split

from app.features import prepare_model_inputs
from app.model import AcousticSpaceBaseline
from app.real_dataset import discover_labeled_files, summarize

# --- Point this at your dataset ---
# Update this to the root folder that contains your labeled audio data.
DATA_DIR = Path(r"C:\Users\amitk\OneDrive\Desktop\archive (1)")

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


class AudioLabelDataset(Dataset):
    def __init__(self, data_dir: Path | str):
        data_dir = Path(data_dir)
        genuine_dir = data_dir / "genuine"
        fake_dir = data_dir / "fake"

        if genuine_dir.exists() and fake_dir.exists():
            samples: list[tuple[Path, int]] = []
            for path in sorted(genuine_dir.glob("*.wav")):
                samples.append((path, 0))
            for path in sorted(fake_dir.glob("*.wav")):
                samples.append((path, 1))
            self.samples = samples
        else:
            self.samples = discover_labeled_files(data_dir)

        if not self.samples:
            raise RuntimeError(f"No audio files found under {data_dir}.")

        summarize(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        mel, rir = prepare_model_inputs(str(path))
        return torch.from_numpy(mel), torch.from_numpy(rir), label

    def class_counts(self) -> Counter:
        return Counter(label for _, label in self.samples)


def _make_weighted_sampler(dataset: Dataset, indices: list[int]) -> WeightedRandomSampler:
    labels = [dataset.samples[i][1] for i in indices]
    counts = Counter(labels)
    weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train(epochs: int = 20, lr: float = 1e-3, batch_size: int = 8, seed: int = 42, fake_class_weight: float = 1.6):
    torch.manual_seed(seed)
    dataset = AudioLabelDataset(DATA_DIR)

    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    sampler = _make_weighted_sampler(dataset, train_set.indices)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = AcousticSpaceBaseline()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Our earlier run had 99% precision but only 58% recall on the fake
    # class — the model was too conservative about calling something fake.
    # Weighting the fake class higher in the loss (on top of the sampler)
    # pushes it to catch more of them, at a small, acceptable cost to precision.
    class_weights = torch.tensor([1.0, fake_class_weight])
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for mel, rir, labels in train_loader:
            optimizer.zero_grad()
            logits = model(mel, rir)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * mel.size(0)

        train_loss = total_loss / len(train_set)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for mel, rir, labels in val_loader:
                logits = model(mel, rir)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total else float("nan")

        print(f"epoch {epoch:2d}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.2%}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "baseline_cnn.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    train()