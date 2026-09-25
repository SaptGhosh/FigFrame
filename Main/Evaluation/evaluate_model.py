#!/usr/bin/env python3
import re
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =====================
# Model - Change model name here
# =====================
MODEL_NAME = ""

# =====================
# Dataset - Change input and output dataset here
# =====================

INPUT_CSV = ""
OUTPUT_CSV = ""

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

    # remove common wrappers
    text = re.sub(r"^Question:\s*", "", text, flags=re.I)
    text = text.split("\n")[0].strip()

    # keep only up to first question mark if present
    # if "?" in text:
    #     text = text[:text.index("?") + 1]

    # # remove quotes
    # text = text.strip("\"' ")


    return text


def make_prompt():
    return """You are an expert in social and moral reasoning.
    You are given a context and a question from Reddit's AITA (Am I The Asshole) forum.
    Your task is to judge whether the person described in the context is an asshole based solely on the information provided.
    
    Respond using the following 7-point Likert scale:
    
    1 = Strongly disagree (the person is definitely not an asshole)
    2 = Moderately disagree
    3 = Slightly disagree
    4 = Neutral (Neither agree nor disagree)
    5 = Slightly agree
    6 = Moderately agree
    7 = Strongly agree (the person is definitely an asshole)
    
    Output rules:
    - Output only a single integer from 1 to 7.
    - Do not provide any explanation or additional text."""




def generate_question(title, text):
    messages = [

        {"role": "system", "content": make_prompt()},

        {"role": "user", "content": f"Context: {text}. {title}."}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    out = generator(
        text,
        max_new_tokens=MAX_NEW_TOKENS,
        # do_sample=True,
        do_sample=False, # uses greedy decoding
        temperature=TEMPERATURE,
        top_p=TOP_P,
        return_full_text=False
    )[0]["generated_text"]

    question = clean_question(out)
    # print(f"generated: {question}")
    # assert 1==0
    return question


# =====================
# Process dataset for evaluation
# =====================
# this code considers csv contains the following headers:
# literal_title --> literal question
# figurative title --> either one of the four figurative rewrite
df = pd.read_csv(INPUT_CSV)

rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    literal_title = str(row["literal title"]).strip()
    figurative_title = str(row["figurative title"]).strip()
    text = str(row["text"]).strip()

    if literal_title and literal_title in text:
        text = text.replace(literal_title, "", 1).strip()

    literal_judgement = generate_question(title=literal_title, text=text)
    figurative_judgement = generate_question(title=figurative_title, text=text)

    rows.append({
        "literal title": literal_title,
        "figurative title": figurative_title,
        "text": text,
        "literal_judgement" : literal_judgement,
        "figurative_judgement" : figurative_judgement,
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
print(out_df.head())