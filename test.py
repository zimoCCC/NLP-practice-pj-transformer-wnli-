import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main(args):
    dataset = load_dataset("SetFit/wnli")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    def preprocess_function(examples):
        return {
            **tokenizer(
                examples["text1"],
                examples["text2"],
                padding="max_length",
                truncation=True,
            ),
            "label": examples["label"],
        }

    dataset = dataset.map(preprocess_function, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        trust_remote_code=True,
    )

    for data in dataset["test"]:
        res = model(torch.tensor([data["input_ids"]]))
        print(np.argmax(res.logits.detach().numpy(), axis=-1).item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="prajjwal1/bert-tiny")
    args = parser.parse_args()
    main(args)
