from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    MarianTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from model import MarianConfig, MarianMTModel

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Fine-tune English→Greek MarianMT")
    p.add_argument("--train_file", required=True, help="CSV with columns en,el for training")
    p.add_argument("--val_file", help="Optional CSV with columns en,el for dev/eval")
    p.add_argument("--output_dir", default="finetuned_en_el", help="Where to save checkpoints")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=128, help="Max tokens per sequence")
    return p.parse_args()

def main() -> None:
    args = parse_args()

    model_name = "Helsinki-NLP/opus-mt-en-el"

    # 1) Tokeniser & Config come from HF hub (keeps vocab identical)
    tokenizer = MarianTokenizer.from_pretrained(model_name, use_fast=True)
    config    = MarianConfig.from_pretrained(model_name)

    # 2) Our *custom* MarianMT implementation loads the same weights
    #    thanks to the common state-dict structure provided by PreTrainedModel
    model = MarianMTModel.from_pretrained(model_name, config=config)

    # 3) Dataset ──────────────────────────────────────────────────────
    data_files = {"train": args.train_file}
    if args.val_file:
        data_files["validation"] = args.val_file

    ds = load_dataset("csv", data_files=data_files)

    def tok(batch):
        model_inputs = tokenizer(
            batch["en"], max_length=args.max_length, truncation=True, padding="max_length"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch["el"], max_length=args.max_length, truncation=True, padding="max_length"
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenised = ds.map(tok, batched=True, remove_columns=["en", "el"])

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 4) Training setup ───────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps" if "validation" in tokenised else "no",
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised.get("validation"),
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # 5) Train! ───────────────────────────────────────────────────────
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Training complete. Fine-tuned model saved to {Path(args.output_dir).resolve()}")

if __name__ == "__main__":
    main()
