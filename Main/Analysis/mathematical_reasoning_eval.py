#!/usr/bin/env python3

import re

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Change name and output file
# ============================================================

MODEL_NAME = ""
OUTPUT_FILE = ""

# set number of samples
N_SAMPLES = 500
SAMPLE_SEED = 42
MAX_NEW_TOKENS = 10


# ============================================================
# LOAD MODEL
# ============================================================

print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()

print("Model loaded.")


# ============================================================
# LOAD AND SAMPLE GSM8K
# ============================================================

dataset = load_dataset(
    "openai/gsm8k",
    "main",
    split="test",
)

if N_SAMPLES > len(dataset):
    raise ValueError(
        f"N_SAMPLES={N_SAMPLES} exceeds the test split size ({len(dataset)})."
    )

dataset = dataset.shuffle(seed=SAMPLE_SEED).select(range(N_SAMPLES))

print(
    f"Sampled {len(dataset)} GSM8K test questions "
    f"with seed {SAMPLE_SEED}."
)


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

LITERAL_QUESTIONS = [
    "How good is the following solution?",
    "How strong is the following solution?",
    "How accurate is the following solution?",
    "How convincing is the following solution?",
    "How would you rate the overall quality of the following solution?",
]

FIGURATIVE_QUESTIONS = [
    "Where does the following solution land on the quality scale?",
    "How does the following solution stack up?",
    "Where does the following solution stand?",
    "Where does the following solution fall on the quality spectrum?",
    "How closely does the following solution track the correct path?",
]


# ============================================================
# PROMPTS
# ============================================================

def build_system_prompt(evaluation_question):
    return f"""You are evaluating a proposed solution to a multi-step science problem.

{evaluation_question}

Rate the solution on a scale from 1 to 10, where 1 is the lowest rating and 10 is the highest rating.

Evaluate whether the reasoning is mathematically valid, whether it addresses the question, and whether the final answer is correct.

Rules:

- Base your evaluation only on the problem and proposed solution provided.
- Output only one integer from 1 to 10.
- Do not provide an explanation or any additional text."""


def build_user_prompt(question, solution):
    return f"""Math problem:
{question}

Proposed solution:
{solution}"""


def generate_rating(question, solution, evaluation_question):
    messages = [
        {
            "role": "system",
            "content": build_system_prompt(evaluation_question),
        },
        {
            "role": "user",
            "content": build_user_prompt(question, solution),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    match = re.search(r"\b(10|[1-9])\b", response)

    if match:
        return int(match.group(1)), response

    return None, response


def extract_final_answer(solution):
    """Extract the GSM8K answer following the final #### marker."""
    match = re.search(r"####\s*(.+?)\s*$", solution)
    return match.group(1).strip() if match else ""


# ============================================================
# RUN EXPERIMENT
# ============================================================

results = []

for sample_id, example in enumerate(
    tqdm(dataset, total=len(dataset), desc="Rating GSM8K solutions")
):
    question = example["question"].strip()
    solution = example["answer"].strip()

    result = {
        "sample_id": sample_id,
        "sample_seed": SAMPLE_SEED,
        "question": question,
        "solution": solution,
        "final_answer": extract_final_answer(solution),
    }

    literal_scores = []

    for i, evaluation_question in enumerate(LITERAL_QUESTIONS, start=1):
        score, raw = generate_rating(
            question,
            solution,
            evaluation_question,
        )

        result[f"literal_score_{i}"] = score
        result[f"literal_raw_{i}"] = raw

        if score is not None:
            literal_scores.append(score)

    figurative_scores = []

    for i, evaluation_question in enumerate(FIGURATIVE_QUESTIONS, start=1):
        score, raw = generate_rating(
            question,
            solution,
            evaluation_question,
        )

        result[f"figurative_score_{i}"] = score
        result[f"figurative_raw_{i}"] = raw

        if score is not None:
            figurative_scores.append(score)

    result["literal_average"] = (
        sum(literal_scores) / len(literal_scores)
        if literal_scores
        else None
    )

    result["figurative_average"] = (
        sum(figurative_scores) / len(figurative_scores)
        if figurative_scores
        else None
    )

    if literal_scores and figurative_scores:
        result["difference"] = (
            result["figurative_average"]
            - result["literal_average"]
        )
    else:
        result["difference"] = None

    results.append(result)

    # Checkpoint after each problem so a long run can be recovered.
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)


# ============================================================
# Finaly analysis
# ============================================================

results_df = pd.DataFrame(results)

valid = results_df.dropna(
    subset=["literal_average", "figurative_average"]
).copy()

print(f"\nSaved {len(results_df)} samples to {OUTPUT_FILE}")

if valid.empty:
    print("No samples had valid scores in both conditions.")
    raise SystemExit(0)

literal_mean = valid["literal_average"].mean()
figurative_mean = valid["figurative_average"].mean()
difference = figurative_mean - literal_mean

harsh = valid["figurative_average"] < valid["literal_average"]
lenient = valid["figurative_average"] > valid["literal_average"]
equal = valid["figurative_average"] == valid["literal_average"]

n = len(valid)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Valid samples:              {n}")
print(f"Average literal score:      {literal_mean:.3f}")
print(f"Average figurative score:   {figurative_mean:.3f}")
print(f"Difference (F - L):         {difference:+.3f}")
print()
print(
    f"Harsh (F < L):              {harsh.sum()}/{n} "
    f"({harsh.mean() * 100:.2f}%)"
)
print(
    f"Lenient (F > L):            {lenient.sum()}/{n} "
    f"({lenient.mean() * 100:.2f}%)"
)
print(
    f"No change:                  {equal.sum()}/{n} "
    f"({equal.mean() * 100:.2f}%)"
)
print("=" * 60)

if difference < 0:
    print("Figurative framing produced HARSHER evaluations.")
elif difference > 0:
    print("Figurative framing produced MORE LENIENT evaluations.")
else:
    print("No average difference.")
