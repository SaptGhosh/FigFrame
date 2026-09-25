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
INPUT_FILE = "" # input file from resume_atlas dataset
OUTPUT_FILE = ""

N_SAMPLES = 100
SAMPLE_SEED = 42
MAX_NEW_TOKENS = 10

# ============================================================
# LOAD MODEL
# ============================================================

print(f"Loading {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()

print("Model loaded.")

# ============================================================
# LOAD + SAMPLE DATA
# ============================================================

#set number of samples
N_SAMPLES = 500
SEED = 42

df = pd.read_csv(INPUT_FILE)

df = df.dropna(
    subset=["Category", "Text"]
).copy()

df = df.sample(
    n=N_SAMPLES,
    random_state=SEED
).reset_index(drop=True)

print(f"Sampled {len(df)} resumes.")

# ============================================================
# EVALUATION QUESTIONS
# ============================================================

LITERAL_QUESTIONS = [
    "How suitable is the candidate for this job category?",
    "How qualified is the candidate for this job category?",
    "How appropriate is the candidate for this job category?",
    "How strong is the candidate for this job category?",
    "How well suited is the candidate for this job category?"
]

FIGURATIVE_QUESTIONS = [
    "How well does the candidate fit the bill for this job category?",
    "How well does the candidate measure up for this job category?",
    "How well does the candidate hit the mark for this job category?",
    "How well does the candidate stack up for this job category?",
    "How well does the candidate make the grade for this job category?"
]

# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt():

    return f"""You are evaluating resume of a candidate for a job 
    
    Rules:
    - Rate on a scale from 1 to 10. 1 being lowest and 10 being the highest.
    - Output only one integer from 1 to 10.
    - Do not provide any explanation or additional text."""

# ============================================================
# USER PROMPT
# ============================================================

def build_user_prompt(category, resume, question):

    return f"""Job Category:
{category}

Resume:
{resume}

{question}"""

# ============================================================
# GENERATION
# ============================================================

def generate_rating(category, resume, question):

    system_prompt = build_system_prompt()

    user_prompt = build_user_prompt(
        category,
        resume,
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
            pad_token_id=tokenizer.eos_token_id
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
    desc="Evaluating resumes"
):

    category = str(row["Category"]).strip()
    resume = str(row["Text"]).strip()

    result = row.to_dict()

    # ========================================================
    # 5 LITERAL QUESTIONS
    # ========================================================

    literal_scores = []

    for i, question in enumerate(
        LITERAL_QUESTIONS,
        start=1
    ):

        score, raw = generate_rating(
            category,
            resume,
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
            category,
            resume,
            question
        )

        result[f"figurative_score_{i}"] = score
        result[f"figurative_raw_{i}"] = raw

        if score is not None:
            figurative_scores.append(score)

    # ========================================================
    # PER-SAMPLE AVERAGES
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

print(f"\nSaved results to {OUTPUT_FILE}")

# ============================================================
# FINAL ANALYSIS
# ============================================================

valid = results_df.dropna(
    subset=[
        "literal_average",
        "figurative_average"
    ]
).copy()

literal_mean = valid[
    "literal_average"
].mean()

figurative_mean = valid[
    "figurative_average"
].mean()

difference = (
    figurative_mean
    - literal_mean
)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    f"Valid samples:              "
    f"{len(valid)}"
)

print(
    f"Average literal score:      "
    f"{literal_mean:.3f}"
)

print(
    f"Average figurative score:   "
    f"{figurative_mean:.3f}"
)

print(
    f"Difference (F - L):         "
    f"{difference:+.3f}"
)

# ============================================================
# HARSH / LENIENT DIRECTION
# ============================================================

L = valid["literal_average"]
F = valid["figurative_average"]

harsh = (F < L).sum()
lenient = (F > L).sum()
equal = (F == L).sum()

n = len(valid)

print()

print(
    f"Harsh (F < L):              "
    f"{harsh}/{n} "
    f"({100 * harsh / n:.2f}%)"
)

print(
    f"Lenient (F > L):            "
    f"{lenient}/{n} "
    f"({100 * lenient / n:.2f}%)"
)

print(
    f"No change:                  "
    f"{equal}/{n} "
    f"({100 * equal / n:.2f}%)"
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