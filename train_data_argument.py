import argparse
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
import matplotlib.pyplot as plt
import random
from nltk.corpus import wordnet

def reverse_sentence_pair(example):
    """反转句子对的顺序"""
    new_example = example.copy()
    new_example["text1"], new_example["text2"] = example["text2"], example["text1"]
    return new_example

def synonym_replacement(text):
    """同义词替换"""
    words = text.split()
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words.append(synonym)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def add_distractor(example):
    """生成干扰项"""
    if "he" in example["text2"]:
        example["text2"] = example["text2"].replace("he", "she")
    elif "she" in example["text2"]:
        example["text2"] = example["text2"].replace("she", "he")
    return example

def add_negation(example):
    """增加否定词"""
    if "is" in example["text2"]:
        example["text2"] = example["text2"].replace("is", "is not")
    elif "was" in example["text2"]:
        example["text2"] = example["text2"].replace("was", "was not")
    return example

def replace_pronouns_with_names(example):
    """指代消解：将代词替换为名字"""
    name = "John"  # 假设名字为John，您可以使用NLP工具进行更灵活的替换
    if "he" in example["text2"]:
        example["text2"] = example["text2"].replace("he", name)
    elif "she" in example["text2"]:
        example["text2"] = example["text2"].replace("she", name)
    return example

def preprocess_function(examples, tokenizer):
    """数据增强和预处理"""
    # 随机应用增强方法
    if random.random() < 0.5:
        examples = reverse_sentence_pair(examples)
    if random.random() < 0.3:
        examples["text1"] = synonym_replacement(examples["text1"])
        examples["text2"] = synonym_replacement(examples["text2"])
    if random.random() < 0.2:
        examples = add_distractor(examples)
    if random.random() < 0.2:
        examples = add_negation(examples)
    if random.random() < 0.2:
        examples = replace_pronouns_with_names(examples)

    return tokenizer(
        examples["text1"] + " [SEP] " + examples["text2"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

def main(args):
    dataset = load_dataset("SetFit/wnli")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 应用数据增强的预处理函数，将 tokenizer 作为参数传递
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        trust_remote_code=True,
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=-1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        return result

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        args=TrainingArguments(
            max_grad_norm=1,
            output_dir=args.output,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=1,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            gradient_accumulation_steps=args.accum,
            num_train_epochs=args.epoch,
        ),
    )
    trainer.train()

    # 提取损失值和准确率
    losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
    accuracies = [log["eval_accuracy"] for log in trainer.state.log_history if "eval_accuracy" in log]
    epochs = [log["epoch"] for log in trainer.state.log_history if "eval_accuracy" in log]

    # 绘制损失曲线并保存
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, label="Training Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss over Time")
    plt.legend()
    plt.grid()
    plt.savefig("training_loss_curve.png")

    # 绘制准确率曲线并保存
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, label="Validation Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("validation_accuracy_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="albert-base-v2")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--output", type=str, default="output")
    args = parser.parse_args()
    main(args)
