"""Fine-tune CamemBERT on the product text and save validation probabilities.

Uses the shared train/validation split. Validation is only scored, never used
to choose settings. Run from the project root:

    python scripts/train_camembert.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/processed/text_train_processed.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/camembert_validation_probabilities.csv"
LOG_PATH = PROJECT_ROOT / "data/processed/camembert_training_log.csv"
EPOCH_PATH = PROJECT_ROOT / "data/processed/camembert_epoch_scores.csv"
MODEL_DIR = PROJECT_ROOT / "models/camembert"

MODEL_NAME = "almanach/camembert-base"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 5e-5
SEED = 42
LOG_EVERY = 500          # steps between two loss measurements
LOSS_SAMPLE = 2000       # fixed validation products used for the validation loss curve


def batches(texts, labels, batch_size, shuffle, rng=None):
    order = rng.permutation(len(texts)) if shuffle else np.arange(len(texts))
    for start in range(0, len(texts), batch_size):
        idx = order[start:start + batch_size]
        yield [texts[i] for i in idx], labels[idx]


def validation_loss(probabilities, labels):
    """Cross-entropy: average of -log(probability given to the true category)."""
    true_probability = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(true_probability, 1e-12, None)).mean())


def predict_probabilities(model, tokenizer, texts, device):
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch_texts, _ in batches(texts, np.zeros(len(texts)), 64, shuffle=False):
            encoded = tokenizer(batch_texts, truncation=True, max_length=MAX_LENGTH,
                                padding=True, return_tensors="pt").to(device)
            logits = model(**encoded).logits
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    if device == "mps":
        torch.mps.empty_cache()  # free GPU memory used by evaluation
    model.train()
    return np.concatenate(probabilities)


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    data = pd.read_csv(DATA_PATH)
    data["text_combined"] = data["text_combined"].fillna("")
    train = data[data["split"] == "train"]
    validation = data[data["split"] == "validation"]

    classes = np.sort(data["prdtypecode"].unique())
    class_index = {code: i for i, code in enumerate(classes)}
    train_texts = train["text_combined"].tolist()
    train_labels = train["prdtypecode"].map(class_index).to_numpy()
    validation_texts = validation["text_combined"].tolist()
    validation_labels = validation["prdtypecode"].map(class_index).to_numpy()
    sample = rng.choice(len(validation_texts), LOSS_SAMPLE, replace=False)
    sample_texts = [validation_texts[i] for i in sample]
    sample_labels = validation_labels[sample]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(classes)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = EPOCHS * int(np.ceil(len(train_texts) / BATCH_SIZE))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)

    print(f"Device: {device} | train: {len(train_texts)} | validation: {len(validation_texts)} | steps: {total_steps}", flush=True)
    start = time.time()
    step = 0
    history, epoch_rows = [], []
    model.train()
    running_loss, steps_since_log = 0.0, 0
    for epoch in range(1, EPOCHS + 1):
        for batch_texts, batch_labels in batches(train_texts, train_labels, BATCH_SIZE, shuffle=True, rng=rng):
            encoded = tokenizer(batch_texts, truncation=True, max_length=MAX_LENGTH,
                                padding=True, return_tensors="pt").to(device)
            labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
            loss = model(**encoded, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            running_loss += loss.item()
            steps_since_log += 1
            if step % LOG_EVERY == 0:
                sample_probabilities = predict_probabilities(model, tokenizer, sample_texts, device)
                row = {
                    "epoch": epoch,
                    "step": step,
                    "train_loss": running_loss / steps_since_log,
                    "validation_loss": validation_loss(sample_probabilities, sample_labels),
                }
                history.append(row)
                pd.DataFrame(history).to_csv(LOG_PATH, index=False)
                minutes = (time.time() - start) / 60
                print(f"epoch {epoch} | step {step}/{total_steps} | train loss {row['train_loss']:.3f} | "
                      f"validation loss {row['validation_loss']:.3f} | {minutes:.0f} min", flush=True)
                running_loss, steps_since_log = 0.0, 0

        probabilities = predict_probabilities(model, tokenizer, validation_texts, device)
        predicted = classes[probabilities.argmax(axis=1)]
        weighted = f1_score(validation["prdtypecode"], predicted, average="weighted")
        macro = f1_score(validation["prdtypecode"], predicted, average="macro")
        full_loss = validation_loss(probabilities, validation_labels)
        epoch_rows.append({"epoch": epoch, "validation_loss": full_loss, "weighted_f1": weighted, "macro_f1": macro})
        pd.DataFrame(epoch_rows).to_csv(EPOCH_PATH, index=False)
        print(f"epoch {epoch} done | validation loss {full_loss:.3f} | weighted F1 {weighted:.4f} | macro F1 {macro:.4f}", flush=True)

    output = pd.DataFrame(probabilities, columns=[str(c) for c in classes])
    output.insert(0, "row_id", validation["row_id"].to_numpy())
    output.to_csv(OUTPUT_PATH, index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Saved validation probabilities and model. Total time: {(time.time() - start) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
