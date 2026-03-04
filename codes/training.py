from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from .config import ClassifierConfig, GeneratorConfig
from .metrics import compute_token_f1
from .moe import load_balancing_loss


@dataclass(slots=True)
class EpochMetrics:
    train_loss: float
    validation_loss: float
    train_macro_f1: float
    validation_macro_f1: float


def evaluate_classifier(model, loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(logits, batch_labels)
            losses.append(loss.item())
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())

    return sum(losses) / len(losses), accuracy_score(labels, predictions), f1_score(labels, predictions, average="macro")


def train_classifier(
    model,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: ClassifierConfig,
    device: torch.device,
) -> list[EpochMetrics]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[EpochMetrics] = []
    model.to(device)

    for _ in range(config.epochs):
        model.train()
        batch_losses: list[float] = []
        batch_predictions: list[int] = []
        batch_labels: list[int] = []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, gate_weights = model(input_ids=input_ids, attention_mask=attention_mask)
            classification_loss = F.cross_entropy(logits, labels)
            auxiliary_loss = load_balancing_loss(gate_weights)
            loss = classification_loss + config.load_balancing_weight * auxiliary_loss
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            batch_predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
            batch_labels.extend(labels.detach().cpu().tolist())

        validation_loss, _, validation_macro_f1 = evaluate_classifier(model, validation_loader, device)
        history.append(
            EpochMetrics(
                train_loss=sum(batch_losses) / len(batch_losses),
                validation_loss=validation_loss,
                train_macro_f1=f1_score(batch_labels, batch_predictions, average="macro"),
                validation_macro_f1=validation_macro_f1,
            )
        )

    return history


def build_generator_trainer(
    model,
    tokenizer,
    train_dataset,
    validation_dataset,
    config: GeneratorConfig,
    output_dir: str,
) -> Trainer:
    def compute_metrics(eval_prediction):
        predictions = eval_prediction.predictions
        labels = eval_prediction.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if predictions.ndim == 3:
            predictions = predictions.argmax(axis=-1)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {"token_f1": compute_token_f1(decoded_predictions, decoded_labels, tokenizer)}

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        logging_steps=50,
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
        save_safetensors=False,
        fp16=torch.cuda.is_available(),
    )
    return Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
