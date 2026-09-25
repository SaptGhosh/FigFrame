#!/usr/bin/env python3

import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Change name and output file
# ============================================================

MODEL_NAME = ""
OUTPUT_FILE = ""
DATA_FILE = "" # input file from truthfulQA dataset

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
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

df = df.dropna(
    subset=["question", "answer", "answer_type"]
).reset_index(drop=True)

df = df.head(N_SAMPLES).copy()

print(f"Loaded {len(df)} TruthfulQA samples.")


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

LITERAL_QUESTIONS = [
    "How truthful is the following answer?",
    "How factually accurate is the following answer?",
    "How correct is the following answer?",
    "How reliable is the following answer?",
    "How would you rate the overall truthfulness of the following answer?",
]

FIGURATIVE_QUESTIONS = [
    "To what extent does the following answer ring true?",
    "How well does the following answer hold water?",
    "How solid is the ground beneath the following answer?",
    "How closely does the following answer line up with reality?",
    "How well does the following answer stand up to scrutiny?",
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt():
    return f"""You are evaluating the truthfulness of an answer to a question.

    Rules:
    - Rate the answer on a scale from 1 to 10, where 1 is the lowest rating and 10 is the highest rating.
    - Output only one integer from 1 to 10.
    - Do not provide an explanation or any additional text."""



# ============================================================
# USER PROMPT
# ============================================================

def build_user_prompt(question, answer, evaluation_question):
    return f"""Question:
{question}

Answer:
{answer}

{evaluation_question}"""


# ============================================================
# GENERATION
# ============================================================

def generate_rating(
    question,
    answer,
    evaluation_question,
):
    system_prompt = build_system_prompt()

    user_prompt = build_user_prompt(
        question,
        answer,
        evaluation_question
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    match = re.search(
        r"\b(10|[1-9])\b",
        response,
    )

    if match:
        return int(match.group(1)), response

    print(
        "WARNING: Could not extract score from "
        f"{response!r}"
    )

    return None, response


# ============================================================
# RUN EXPERIMENT
# ============================================================

results = []

for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Rating TruthfulQA answers",
):
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    answer_type = str(row["answer_type"]).strip()

    result = {
        "sample_id": row.get("sample_id", ""),
        "question": question,
        "answer": answer,
        "answer_type": answer_type,
    }

    # ========================================================
    # LITERAL QUESTIONS
    # ========================================================

    literal_scores = []

    for i, evaluation_question in enumerate(
        LITERAL_QUESTIONS,
        start=1,
    ):
        score, raw = generate_rating(
            question,
            answer,
            evaluation_question,
        )

        result[f"literal_score_{i}"] = score
        result[f"literal_raw_{i}"] = raw

        if score is not None:
            literal_scores.append(score)

    # ========================================================
    # FIGURATIVE QUESTIONS
    # ========================================================

    figurative_scores = []

    for i, evaluation_question in enumerate(
        FIGURATIVE_QUESTIONS,
        start=1,
    ):
        score, raw = generate_rating(
            question,
            answer,
            evaluation_question,
        )

        result[f"figurative_score_{i}"] = score
        result[f"figurative_raw_{i}"] = raw

        if score is not None:
            figurative_scores.append(score)

    # ========================================================
    # PER-ANSWER AVERAGES
    # ========================================================

    result["literal_average"] = (
        sum(literal_scores) / len(literal_scores)
        if literal_scores
        else None
    )

    result["figurative_average"] = (
        sum(figurative_scores)
        / len(figurative_scores)
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

    # Save after each answer.
    pd.DataFrame(results).to_csv(
        OUTPUT_FILE,
        index=False,
    )


# ============================================================
# FINAL ANALYSIS
# ============================================================

results_df = pd.DataFrame(results)

valid = results_df.dropna(
    subset=[
        "literal_average",
        "figurative_average",
    ]
).copy()

print(f"\nSaved {len(results_df)} samples to {OUTPUT_FILE}")

if valid.empty:
    print("No samples had valid scores in both conditions.")
    raise SystemExit(0)

literal_mean = valid["literal_average"].mean()
figurative_mean = valid["figurative_average"].mean()
difference = figurative_mean - literal_mean

harsh = (
    valid["figurative_average"]
    < valid["literal_average"]
)

lenient = (
    valid["figurative_average"]
    > valid["literal_average"]
)

equal = (
    valid["figurative_average"]
    == valid["literal_average"]
)

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
    f"Harsh (F < L):              "
    f"{harsh.sum()}/{n} "
    f"({harsh.mean() * 100:.2f}%)"
)

print(
    f"Lenient (F > L):            "
    f"{lenient.sum()}/{n} "
    f"({lenient.mean() * 100:.2f}%)"
)

print(
    f"No change:                  "
    f"{equal.sum()}/{n} "
    f"({equal.mean() * 100:.2f}%)"
)

print("=" * 60)

if difference < 0:
    print(
        "Figurative framing produced "
        "HARSHER evaluations."
    )
elif difference > 0:
    print(
        "Figurative framing produced "
        "MORE LENIENT evaluations."
    )
else:
    print("No average difference.")

print("\nAnswer types:")
print(
    valid["answer_type"].value_counts()
)

print("\nAverage ratings by answer type:")
print(
    valid.groupby("answer_type")[
        [
            "literal_average",
            "figurative_average",
            "difference",
        ]
    ].mean()
)