#!/usr/bin/env python3
import re
import pandas as pd
from tqdm import tqdm
import torch
from openai import OpenAI

# INPUT YOUR API KEY HERE
client = OpenAI(api_key="")


# Reads the input file
INPUT_CSV = ""
# change name of the output file
OUTPUT_CSV = ""

# =====================
# Clean question
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

    # print(text)
    # assert 1==0

    return text


# =====================
# Creat the dataset
# =====================
df = pd.read_csv(INPUT_CSV)

rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    title = str(row["title"]).strip()
    text = str(row["text"]).strip()
    verdict = str(row["verdict"]).strip()

    response = client.responses.create(
        model="gpt-5-mini",
        reasoning={"effort": "medium"},
        input=[
            {
                "role": "system",
                "content": '''You are an expert in sarcasm. 
                You are given a question title from Reddit’s AITA (Am I The Asshole) dataset. 
                Your task is to rewrite the title using **strong, explicit sarcasm** while preserving its original meaning. 

                Rules:
                * Preserve the original events, intent, and moral framing.
                * Do NOT change who did what or introduce any new facts.
                * Begin the rewritten title with “aita”.
                * Make the sarcasm obvious and emotionally expressive.
                * Do NOT explain the sarcasm.
                * Output ONLY the rewritten title.'''
            },
            {"role": "user", "content": "aita for ignoring my roommate after she insulted me?"},
            {"role": "assistant", "content": "aita for apparently committing the unforgivable act of ignoring my roommate after she insulted me?"},

            {"role": "user", "content": f"{title}"},
            
        ]
    )

    res= (response.output_text)
    # print(res)
    # assert 1==0

    rows.append({
        "literal title": title,
        "figurative title": res,
        "text": text,
        "verdict": verdict,
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
print(out_df.head())