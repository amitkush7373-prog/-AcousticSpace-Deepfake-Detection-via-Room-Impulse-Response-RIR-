"""
AcousticSpace - AST Fine-Tuning (Week 3)

Fine-tunes AcousticSpaceAST on data/synthetic/{genuine,fake}/*.wav for
now (swap in data/asvspoof/ once available — same script).

IMPORTANT: the first run downloads the pretrained AST checkpoint from
huggingface.co, so this needs to be run on a machine with normal
internet access (it will NOT work inside a network-restricted sandbox).

Run:
  python -m app.finetune_ast
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from app.ast_features import prepare_ast_input
from app.features import prepare_rir_vector
from app.audio_pipeline import load_audio
from app.ast_model import AcousticSpaceAST

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"
CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


class ASTAudioDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.samples: list[tuple[Path, int]] = []
        for path in sorted((data_dir / "genuine").glob("*.wav")):
            self.samples.append((path, 0))
        for path in sorted((data_dir / "fake").glob("*.wav")):
            self.samples.append((path, 1))
        if not self.samples:
            raise RuntimeError(
                f"No .wav files found under {data_dir}. "
                "Run `python -m app.dataset` first to generate synthetic clips."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        ast_input = prepare_ast_input(str(path)).squeeze(0)  # (max_length, num_mel_bins)
        y, sr = load_audio(str(path))
        rir = prepare_rir_vector(y, sr)
        return ast_input, torch.from_numpy(rir), label


def finetune(epochs: int = 10, lr: float = 1e-3, batch_size: int = 4, freeze_backbone: bool = True):
    dataset = ASTAudioDataset(DATA_DIR)

    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = AcousticSpaceAST(freeze_backbone=freeze_backbone)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for ast_input, rir, labels in train_loader:
            optimizer.zero_grad()
            logits = model(ast_input, rir)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * ast_input.size(0)

        train_loss = total_loss / len(train_set)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for ast_input, rir, labels in val_loader:
                logits = model(ast_input, rir)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total else float("nan")

        print(f"epoch {epoch:2d}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.2%}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / "ast_finetuned.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    finetune()
