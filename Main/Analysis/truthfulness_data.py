#!/usr/bin/env python3

import random
from datasets import load_dataset, Dataset

SEED = 42

OUTPUT_JSONL = ""
OUTPUT_CSV = ""
best_answer_num = # set how many best answers you want
correct_answer_num = # set how many correct answers you want
incorrect_answer_num = # # set how many incorrect answers you want

rng = random.Random(SEED)

dataset = load_dataset(
    "domenicrosati/TruthfulQA",
    split="train",
)

dataset = dataset.shuffle(seed=SEED)

answer_types = (
    ["best_answer"] * best_answer_num
    + ["correct_answer"] * correct_answer_num
    + ["incorrect_answer"] * incorrect_answer_num
)

rng.shuffle(answer_types)

results = []

for sample_id, answer_type in enumerate(answer_types):
    example = dataset[sample_id]

    question = example["Question"]

    if answer_type == "best_answer":
        answer = example["Best Answer"]

    elif answer_type == "correct_answer":
        correct_answers = [
            answer.strip()
            for answer in example["Correct Answers"].split(";")
            if answer.strip()
        ]
        answer = rng.choice(correct_answers)

    else:
        incorrect_answers = [
            answer.strip()
            for answer in example["Incorrect Answers"].split(";")
            if answer.strip()
        ]
        answer = rng.choice(incorrect_answers)

    results.append(
        {
            "sample_id": sample_id,
            "question": question,
            "answer": answer,
            "answer_type": answer_type,
        }
    )

sample = Dataset.from_list(results)

sample.to_json(
    OUTPUT_JSONL,
    orient="records",
    lines=True,
    force_ascii=False,
)

sample.to_csv(OUTPUT_CSV)

print(f"Saved {len(sample)} samples.")
print(f"JSONL: {OUTPUT_JSONL}")
print(f"CSV:   {OUTPUT_CSV}")

print("\nAnswer distribution:")
for answer_type in [
    "best_answer",
    "correct_answer",
    "incorrect_answer",
]:
    count = sum(
        value == answer_type
        for value in sample["answer_type"]
    )
    print(f"{answer_type}: {count}")