#!/usr/bin/env python3
import re
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =====================
# Model - Change model name here
# =====================
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
# MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# MODEL_NAME = "google/gemma-3-12b-it"
# MODEL_NAME = "google/gemma-3-27b-it"

# =====================
# Dataset - Change input and output dataset here
# =====================

INPUT_CSV = "aita_500_neg_samples_fig_gpt.csv"
OUTPUT_CSV = "qwen3_4b_evaluation.csv"

# Default code uses greedy decoding, sampling = False
MAX_NEW_TOKENS = 15
TEMPERATURE = 0.7 # change temperature here
TOP_P = 0.9 # change top-p here


# =====================
# Load the LLM
# =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)


# =====================
# Clean the question
# =====================

def clean_question(text):
    text = text.strip()

    text = re.sub(
        r"^Question:\s*",
        "",
        text,
        flags=re.I
    )

    text = text.split("\n")[0].strip()

    return text

# =====================
# Reverse the aita question to negative polarity
# =====================

def reverse_aita_question(title):

    title = str(title).strip()

    pattern = r"^\s*(aita|am\s+i\s+the\s+asshole)\b"

    if not re.match(
        pattern,
        title,
        flags=re.IGNORECASE
    ):
        return None

    return re.sub(
        pattern,
        "was i in the wrong",
        title,
        count=1,
        flags=re.IGNORECASE
    )


def make_prompt():

    return """You are an expert in social and moral reasoning.
    You are given a context and a question from an online forum.
    Your task is to judge whether the person described in the context was in the wrong based solely on the information provided.
    
    Respond using the following 7-point Likert scale:
    
    1 = Strongly agree (the person was definitely not in the wrong)
    2 = Moderately agree
    3 = Slightly agree
    4 = Neutral (Neither agree nor disagree)
    5 = Slightly disagree
    6 = Moderately disagree
    7 = Strongly disagree (the person was definitely in the wrong)
    
    Output rules:
    - Output only a single integer from 1 to 7.
    - Do not provide any explanation or additional text."""


def generate_question(title, text):

    messages = [
        {
            "role": "system",
            "content": make_prompt()
        },

        {
            "role": "user",
            "content": f"Context: {text}. {title}."
        }
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    out = generator(
        prompt_text,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        return_full_text=False
    )[0]["generated_text"]

    judgement = clean_question(out)

    return judgement


# =====================
# Load both positive and negative datasets together
# =====================

pos_df = pd.read_csv(
    POS_INPUT_CSV
).head(N_PER_LABEL).copy()

neg_df = pd.read_csv(
    NEG_INPUT_CSV
).head(N_PER_LABEL).copy()

# Add source label
pos_df["label"] = "positive"
neg_df["label"] = "negative"

# Combine
df = pd.concat(
    [pos_df, neg_df],
    ignore_index=True
)

print(f"Positive samples: {len(pos_df)}")
print(f"Negative samples: {len(neg_df)}")
print(f"Total samples:    {len(df)}")


# =====================
# process the datasets
# =====================

rows = []
skipped = 0

for _, row in tqdm(
    df.iterrows(),
    total=len(df)
):

    original_literal_title = str(
        row["literal title"]
    ).strip()

    original_figurative_title = str(
        row["figurative title"]
    ).strip()

    text = str(
        row["text"]
    ).strip()

    label = row["label"]

    # ---------------------
    # remove "aita" 
    # ---------------------

    if (
        original_literal_title
        and original_literal_title in text
    ):
        text = text.replace(
            original_literal_title,
            "",
            1
        ).strip()

    # ---------------------
    # replace "aita" with "was i in the wrong"
    # ---------------------

    literal_title = reverse_aita_question(
        original_literal_title
    )

    figurative_title = reverse_aita_question(
        original_figurative_title
    )

    # Skip sample if either title could not be reversed
    if (
        literal_title is None
        or figurative_title is None
    ):
        skipped += 1

        print(
            f"Skipped: {skipped} "
            f"| Label: {label}"
        )

        continue

    # ---------------------
    # Evaluate
    # ---------------------

    literal_judgement = generate_question(
        title=literal_title,
        text=text
    )

    figurative_judgement = generate_question(
        title=figurative_title,
        text=text
    )

    # ---------------------
    # Save results
    # ---------------------

    rows.append({

        "label": label,

        "original_literal_title":
            original_literal_title,

        "original_figurative_title":
            original_figurative_title,

        "literal title":
            literal_title,

        "figurative title":
            figurative_title,

        "text":
            text,

        "literal_judgement":
            literal_judgement,

        "figurative_judgement":
            figurative_judgement,
    })

out_df = pd.DataFrame(rows)

out_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nSaved:", OUTPUT_CSV)

print("\nOutput label counts:")
print(
    out_df["label"].value_counts()
)

print(
    out_df[
        [
            "label",
            "original_literal_title",
            "literal title",
            "original_figurative_title",
            "figurative title",
            "literal_judgement",
            "figurative_judgement"
        ]
    ].head(10)
)

print(f"\nSkipped: {skipped}")
print(f"Saved samples: {len(out_df)}")