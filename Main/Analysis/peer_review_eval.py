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
DATA_FILE = "" # input file from review_arena dataset

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
    subset=["title", "abstract"]
).reset_index(drop=True)

df = df.head(N_SAMPLES).copy()

print(f"Loaded {len(df)} papers.")

# ============================================================
# EVALUATION QUESTIONS
# ============================================================


LITERAL_QUESTIONS = [
    "How good is the following research?",
    "How strong is the following research?",
    "How promising is the following research?",
    "How compelling is the following research?",
    "How would you rate the overall quality of the following research?"
]

FIGURATIVE_QUESTIONS = [
    "Where does the following research land on the quality scale?",
    "How does the following research stack up?",
    "Where does the following research stand?",
    "Where does the following research fall on the quality spectrum?",
    "How closely does the following research track the correct path?",
]

# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt():

    return f"""You are evaluating the abstract of a scientific research paper.

Rules:

- Rate on a scale from 1 to 10. 1 being lowest and 10 being the highest.
- Output only one integer from 1 to 10.
- Do not provide any explanation or additional text."""



# ============================================================
# USER PROMPT
# ============================================================

def build_user_prompt(title, abstract, question):

    return f"""Title:
{title}

Abstract:
{abstract}

{question}"""

def generate_rating(title, abstract, question):

    system_prompt = build_system_prompt()

    user_prompt = build_user_prompt(
        title,
        abstract,
        question
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
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
        skip_special_tokens=True
    ).strip()

    match = re.search(
        r"\b(10|[1-9])\b",
        response
    )

    if match:
        return int(match.group(1)), response

    return None, response

# ============================================================
# RUN EXPERIMENT
# ============================================================

results = []

for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Reviewing full papers"
):

    title = str(row["title"]).strip()
    abstract = str(row["abstract"]).strip()

    result = {
        "title": title,
        "year": row.get("year", ""),
        "conference": row.get("conference", ""),
        "decision": row.get("decision", ""),
    }

    # ========================================================
    # 5 LITERAL QUESTIONS
    # ========================================================

    literal_scores = []

    for i, question in enumerate(
        LITERAL_QUESTIONS,
        start=1
    ):

        score, raw = generate_rating(
            title,
            abstract,
            question
        )

        result[f"literal_score_{i}"] = score
        result[f"literal_raw_{i}"] = raw

        if score is not None:
            literal_scores.append(score)

    # ========================================================
    # 5 FIGURATIVE QUESTIONS
    # ========================================================

    figurative_scores = []

    for i, question in enumerate(
        FIGURATIVE_QUESTIONS,
        start=1
    ):

        score, raw = generate_rating(
            title,
            abstract,
            question
        )

        result[f"figurative_score_{i}"] = score
        result[f"figurative_raw_{i}"] = raw

        if score is not None:
            figurative_scores.append(score)

    # ========================================================
    # PER-PAPER AVERAGES
    # ========================================================

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

# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved to {OUTPUT_FILE}")

# ============================================================
# FINAL ANALYSIS
# ============================================================

valid = results_df.dropna(
    subset=[
        "literal_average",
        "figurative_average"
    ]
).copy()

L = valid["literal_average"]
F = valid["figurative_average"]

literal_mean = L.mean()
figurative_mean = F.mean()

difference = figurative_mean - literal_mean

# ============================================================
# DIRECTION
# ============================================================

harsh = (F < L)
lenient = (F > L)
equal = (F == L)

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
        "Figurative framing produced HARSHER evaluations."
    )
elif difference > 0:
    print(
        "Figurative framing produced MORE LENIENT evaluations."
    )
else:
    print("No average difference.")

print("\nActual decisions:")
print(
    valid["decision"].value_counts()
)