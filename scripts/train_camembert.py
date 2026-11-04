"""Fine-tune CamemBERT on the product text and save validation probabilities.

Uses the shared train/validation split. Validation is only scored, never used
to choose settings. Run from the project root:

    python scripts/train_camembert.py

Steps:
    1. Load the text and turn category codes into numbers 0 to 26.
    2. Load the pre-trained CamemBERT with a new 27-category output layer.
    3. Train for 2 epochs. Every 500 steps, log train and validation loss.
    4. After each epoch, score all validation products.
    5. Save the validation probabilities and the trained model.
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
MAX_LENGTH = 128         # tokens kept per product
BATCH_SIZE = 16          # products per training step
EPOCHS = 2
LEARNING_RATE = 5e-5
WARMUP_SHARE = 0.06      # share of steps where the learning rate grows from 0
SEED = 42
LOG_EVERY = 500          # steps between two loss measurements
LOSS_SAMPLE = 2000       # fixed validation products used for the validation loss curve


# ---------- small helpers ----------

def batches(texts, labels, batch_size, shuffle, rng=None):
    """Yield (texts, labels) in groups of batch_size, shuffled if asked."""
    order = rng.permutation(len(texts)) if shuffle else np.arange(len(texts))
    for start in range(0, len(texts), batch_size):
        idx = order[start:start + batch_size]
        yield [texts[i] for i in idx], labels[idx]


def encode(tokenizer, texts, device):
    """Turn a list of texts into token IDs, cut to MAX_LENGTH."""
    return tokenizer(texts, truncation=True, max_length=MAX_LENGTH,
                     padding=True, return_tensors="pt").to(device)


def cross_entropy(probabilities, labels):
    """Average of -log(probability given to the true category)."""
    true_probability = probabilities[np.arange(len(labels)), labels]
    return float(-np.log(np.clip(true_probability, 1e-12, None)).mean())


def predict_probabilities(model, tokenizer, texts, device):
    """Predict the 27 probabilities for each text, without training."""
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch_texts, _ in batches(texts, np.zeros(len(texts)), 64, shuffle=False):
            logits = model(**encode(tokenizer, batch_texts, device)).logits
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
    if device == "mps":
        torch.mps.empty_cache()  # free GPU memory used by evaluation
    model.train()
    return np.concatenate(probabilities)


# ---------- main steps ----------

def load_data(rng):
    """Step 1: load texts and labels for train, validation and the loss sample."""
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

    return (validation, classes, train_texts, train_labels,
            validation_texts, validation_labels, sample_texts, sample_labels)


def load_model(n_classes, n_train, device):
    """Step 2: pre-trained CamemBERT, optimiser and learning rate schedule."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=n_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = EPOCHS * int(np.ceil(n_train / BATCH_SIZE))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_SHARE * total_steps), total_steps)
    return tokenizer, model, optimizer, scheduler, total_steps


def training_step(model, tokenizer, optimizer, scheduler, batch_texts, batch_labels, device):
    """Step 3: one update of the model on one batch. Returns the batch loss."""
    labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    loss = model(**encode(tokenizer, batch_texts, device), labels=labels).loss
    loss.backward()                                            # how each weight should change
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)    # cap very large updates
    optimizer.step()                                           # apply the change
    scheduler.step()                                           # update the learning rate
    optimizer.zero_grad()                                      # reset before the next batch
    return loss.item()


def save_outputs(model, tokenizer, validation, classes, probabilities):
    """Step 5: save validation probabilities (for fusion) and the trained model."""
    output = pd.DataFrame(probabilities, columns=[str(c) for c in classes])
    output.insert(0, "row_id", validation["row_id"].to_numpy())
    output.to_csv(OUTPUT_PATH, index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    (validation, classes, train_texts, train_labels,
     validation_texts, validation_labels, sample_texts, sample_labels) = load_data(rng)
    tokenizer, model, optimizer, scheduler, total_steps = load_model(len(classes), len(train_texts), device)
    print(f"Device: {device} | train: {len(train_texts)} | validation: {len(validation_texts)} | steps: {total_steps}", flush=True)

    start = time.time()
    step = 0
    loss_log, epoch_scores = [], []
    running_loss, steps_since_log = 0.0, 0
    model.train()

    for epoch in range(1, EPOCHS + 1):
        # Step 3: train on every training product once
        for batch_texts, batch_labels in batches(train_texts, train_labels, BATCH_SIZE, shuffle=True, rng=rng):
            running_loss += training_step(model, tokenizer, optimizer, scheduler, batch_texts, batch_labels, device)
            step += 1
            steps_since_log += 1

            if step % LOG_EVERY == 0:
                sample_probabilities = predict_probabilities(model, tokenizer, sample_texts, device)
                row = {
                    "epoch": epoch,
                    "step": step,
                    "train_loss": running_loss / steps_since_log,
                    "validation_loss": cross_entropy(sample_probabilities, sample_labels),
                }
                loss_log.append(row)
                pd.DataFrame(loss_log).to_csv(LOG_PATH, index=False)
                print(f"epoch {epoch} | step {step}/{total_steps} | train loss {row['train_loss']:.3f} | "
                      f"validation loss {row['validation_loss']:.3f} | {(time.time() - start) / 60:.0f} min", flush=True)
                running_loss, steps_since_log = 0.0, 0

        # Step 4: score all validation products
        probabilities = predict_probabilities(model, tokenizer, validation_texts, device)
        predicted = classes[probabilities.argmax(axis=1)]
        scores = {
            "epoch": epoch,
            "validation_loss": cross_entropy(probabilities, validation_labels),
            "weighted_f1": f1_score(validation["prdtypecode"], predicted, average="weighted"),
            "macro_f1": f1_score(validation["prdtypecode"], predicted, average="macro"),
        }
        epoch_scores.append(scores)
        pd.DataFrame(epoch_scores).to_csv(EPOCH_PATH, index=False)
        print(f"epoch {epoch} done | validation loss {scores['validation_loss']:.3f} | "
              f"weighted F1 {scores['weighted_f1']:.4f} | macro F1 {scores['macro_f1']:.4f}", flush=True)

    # Step 5: save the results of the last epoch
    save_outputs(model, tokenizer, validation, classes, probabilities)
    print(f"Saved validation probabilities and model. Total time: {(time.time() - start) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
