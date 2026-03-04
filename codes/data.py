from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .config import DataConfig
from .prompts import build_prompt

_TEXT_COLUMNS_TO_CLEAN = ("disease_A", "disease_B", "pathway_A", "pathway_B", "se_A", "se_B")


def load_dataframe(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".pkl":
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported data format: {path.suffix}")


def prepare_dataframe(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    frame = df.copy()

    if config.rename_drug_columns:
        frame = frame.rename(columns={"DrugName": "Drug_A", "DrugName_2": "Drug_B"})

    if "type" in frame.columns:
        min_type = frame["type"].min()
        if pd.notna(min_type) and min_type == 1:
            frame["type"] = frame["type"] - 1

    for column in _TEXT_COLUMNS_TO_CLEAN:
        if column in frame.columns:
            frame[column] = (
                frame[column]
                .astype(str)
                .str.replace(r"[{}']", "", regex=True)
                .str.strip()
            )

    frame[config.input_column] = frame.apply(
        lambda row: build_prompt(row, config.prompt_style),
        axis=1,
    )

    if config.target_text_column and config.target_text_column in frame.columns:
        frame[config.target_text_column] = frame[config.target_text_column].fillna("").astype(str)

    return frame


def split_dataframe(
    df: pd.DataFrame,
    label_column: Optional[str],
    config: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = config.split
    stratify = df[label_column] if split.stratify and label_column and label_column in df.columns else None

    train_df, temp_df = train_test_split(
        df,
        train_size=split.train_size,
        random_state=split.random_state,
        stratify=stratify,
    )

    relative_validation_size = split.validation_size / (split.validation_size + split.test_size)
    temp_stratify = temp_df[label_column] if stratify is not None else None
    validation_df, test_df = train_test_split(
        temp_df,
        train_size=relative_validation_size,
        random_state=split.random_state,
        stratify=temp_stratify,
    )
    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True), test_df.reset_index(drop=True)


class ClassificationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        text_column: str,
        label_column: str,
        max_length: int,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        encoding = self.tokenizer(
            row[self.text_column],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row[self.label_column]), dtype=torch.long),
        }


def build_seq2seq_datasets(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    data_config: DataConfig,
) -> tuple[HFDataset, HFDataset, HFDataset]:
    target_column = data_config.target_text_column
    if not target_column:
        raise ValueError("target_text_column must be set for generation tasks.")

    def preprocess(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        model_inputs = tokenizer(
            batch[data_config.input_column],
            max_length=data_config.max_input_length,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            batch[target_column],
            max_length=data_config.max_target_length,
            padding="max_length",
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def convert(frame: pd.DataFrame) -> HFDataset:
        dataset = HFDataset.from_pandas(frame, preserve_index=False)
        dataset = dataset.map(preprocess, batched=True)
        removable = [column for column in frame.columns if column in dataset.column_names]
        return dataset.remove_columns(removable)

    return convert(train_df), convert(validation_df), convert(test_df)
