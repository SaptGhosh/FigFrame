#!/usr/bin/env python3

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================
# Model - Change model name here
# =====================
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
# MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# MODEL_NAME = "google/gemma-3-12b-it"
# MODEL_NAME = "google/gemma-3-27b-it"

POS_FILE = "likert_pos_gemma_12b.csv"
NEG_FILE = "likert_neg_gemma_12b.csv"

MAX_INPUT_TOKENS = 2048

# Choose layer manually
ATTENTION_LAYER = 21

# Set None for all rows
N_ROWS = None


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)

model.eval()

# ============================================================
# Prompt
# ============================================================

def make_prompt_text():
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


def make_messages(context, question):

    return [
        {
            "role": "system",
            "content": make_prompt_text()
        },
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}"
        },
    ]


def make_prompt(context, question):

    return tokenizer.apply_chat_template(
        make_messages(context, question),
        tokenize=False,
        add_generation_prompt=True,
    )

# ============================================================
# CHARACTER -> TOKEN SPAN
# ============================================================

def span_from_chars(prompt, char_start, char_end):

    enc = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    offsets = enc["offset_mapping"]

    tok_start = None
    tok_end = None

    for i, (s, e) in enumerate(offsets):

        if tok_start is None and e > char_start:
            tok_start = i

        if s < char_end:
            tok_end = i + 1

    return tok_start, tok_end


def find_context_span(prompt, context):

    marker = f"Context: {context}"

    char_start = prompt.find(marker)

    if char_start == -1:
        raise ValueError("Could not find context in prompt.")

    char_start += len("Context: ")
    char_end = char_start + len(context)

    return span_from_chars(
        prompt,
        char_start,
        char_end
    )


def find_question_span(prompt, question):

    marker = f"Question: {question}"

    char_start = prompt.find(marker)

    if char_start == -1:
        raise ValueError(
            f"Could not find question in prompt:\n{question}"
        )

    char_start += len("Question: ")
    char_end = char_start + len(question)

    return span_from_chars(
        prompt,
        char_start,
        char_end
    )

# ============================================================
# ATTENTION
# ============================================================

@torch.no_grad()
def compute_question_attention(
    prompt,
    context,
    question
):

    # --------------------------------------------------------
    # Find spans
    # --------------------------------------------------------

    ctx_start, ctx_end = find_context_span(
        prompt,
        context
    )

    q_start, q_end = find_question_span(
        prompt,
        question
    )

    # --------------------------------------------------------
    # Tokenize actual model input
    # --------------------------------------------------------

    inputs = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    ).to(model.device)

    seq_len = inputs["input_ids"].shape[1]

    # Clamp spans after truncation
    ctx_start = min(ctx_start, seq_len)
    ctx_end = min(ctx_end, seq_len)

    q_start = min(q_start, seq_len)
    q_end = min(q_end, seq_len)

    if ctx_start >= ctx_end:
        return None

    if q_start >= q_end:
        return None

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    outputs = model(
        **inputs,
        output_attentions=True,
        use_cache=False,
    )

    # [heads, query_tokens, key_tokens]
    attn = outputs.attentions[ATTENTION_LAYER][0].float()

    # ========================================================
    # 1. QUESTION -> CONTEXT
    # ========================================================

    # Query = question tokens
    # Key   = context tokens

    context_slice = attn[
        :,
        q_start:q_end,
        ctx_start:ctx_end
    ]

    # For each head and question token:
    # sum attention over all context tokens.
    #
    # Then average across heads + question tokens.
    context_mass = (
        context_slice
        .sum(dim=-1)
        .mean()
        .item()
    )

    # ========================================================
    # 2. QUESTION -> PREVIOUS QUESTION TOKENS
    # ========================================================

    question_prev_masses = []

    for q in range(q_start, q_end):

        if q > q_start:

            # Current question token attending to earlier
            # question tokens only
            q_prev_slice = attn[
                :,
                q,
                q_start:q
            ]

            q_prev_mass = (
                q_prev_slice
                .sum(dim=-1)
                .mean()
                .item()
            )

        else:
            q_prev_mass = 0.0

        question_prev_masses.append(
            q_prev_mass
        )

    question_mass = (
        sum(question_prev_masses)
        / len(question_prev_masses)
    )

    # ========================================================
    # 3. QUESTION ATTENTION SHARE
    # ========================================================

    denominator = (
        question_mass +
        context_mass
    )

    question_share = (
        question_mass / denominator
        if denominator > 0
        else 0.0
    )

    context_share = (
        context_mass / denominator
        if denominator > 0
        else 0.0
    )

    # ========================================================
    # OPTIONAL: ATTENTION DENSITY
    # ========================================================
    #
    # Context is much longer than the question.
    # Therefore total mass alone has a span-length confound.
    #
    # These values measure attention per available token.

    context_length = ctx_end - ctx_start
    question_length = q_end - q_start

    context_density = (
        context_mass / context_length
        if context_length > 0
        else 0.0
    )

    # Number of previous question tokens available grows
    # across the question.
    avg_available_question_tokens = (
        (question_length - 1) / 2
        if question_length > 1
        else 1
    )

    question_density = (
        question_mass / avg_available_question_tokens
        if avg_available_question_tokens > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del outputs
    del attn

    torch.cuda.empty_cache()

    return {
        "seq_len": seq_len,

        "context_mass": context_mass,
        "question_mass": question_mass,

        "question_share": question_share,
        "context_share": context_share,

        "context_density": context_density,
        "question_density": question_density,

        "context_tokens": context_length,
        "question_tokens": question_length,
    }


# ============================================================
# LOAD SAVED RESPONSES
# ============================================================

# ============================================================
# LOAD + COMBINE POSITIVE AND NEGATIVE DATA
# ============================================================

df_pos = pd.read_csv(POS_FILE)
df_neg = pd.read_csv(NEG_FILE)

df = pd.concat(
    [df_pos, df_neg],
    ignore_index=True
)

required_cols = [
    "literal title",
    "figurative title",
    "text",
    "literal_judgement",
    "figurative_judgement",
]

df = df.dropna(
    subset=required_cols
).reset_index(drop=True)

if N_ROWS is not None:
    df = df.head(N_ROWS).copy()

# ============================================================
# RUN
# ============================================================

rows = []

for row_id, row in tqdm(
    df.iterrows(),
    total=len(df)
):

    literal_question = str(
        row["literal title"]
    ).strip()

    figurative_question = str(
        row["figurative title"]
    ).strip()

    context = str(
        row["text"]
    ).strip()

    # --------------------------------------------------------
    # Reconstruct prompts
    # --------------------------------------------------------

    literal_prompt = make_prompt(
        context,
        literal_question
    )

    figurative_prompt = make_prompt(
        context,
        figurative_question
    )

    # --------------------------------------------------------
    # Compute attention
    # --------------------------------------------------------

    literal_attn = compute_question_attention(
        literal_prompt,
        context,
        literal_question
    )

    figurative_attn = compute_question_attention(
        figurative_prompt,
        context,
        figurative_question
    )

    if (
        literal_attn is None
        or figurative_attn is None
    ):
        continue

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    rows.append({

        "row_id": row_id,

        "literal_title":
            literal_question,

        "figurative_title":
            figurative_question,

        "text":
            context,

        "literal_judgement":
            row["literal_judgement"],

        "figurative_judgement":
            row["figurative_judgement"],

        # -----------------------------------------
        # Context attention
        # -----------------------------------------

        "literal_context_mass":
            literal_attn["context_mass"],

        "figurative_context_mass":
            figurative_attn["context_mass"],

        "delta_context_mass":
            figurative_attn["context_mass"]
            - literal_attn["context_mass"],

        # -----------------------------------------
        # Question attention
        # -----------------------------------------

        "literal_question_mass":
            literal_attn["question_mass"],

        "figurative_question_mass":
            figurative_attn["question_mass"],

        "delta_question_mass":
            figurative_attn["question_mass"]
            - literal_attn["question_mass"],

        # -----------------------------------------
        # Relative question share
        # -----------------------------------------

        "literal_question_share":
            literal_attn["question_share"],

        "figurative_question_share":
            figurative_attn["question_share"],

        "delta_question_share":
            figurative_attn["question_share"]
            - literal_attn["question_share"],

        # -----------------------------------------
        # Attention density
        # -----------------------------------------

        "literal_context_density":
            literal_attn["context_density"],

        "figurative_context_density":
            figurative_attn["context_density"],

        "literal_question_density":
            literal_attn["question_density"],

        "figurative_question_density":
            figurative_attn["question_density"],

        "delta_question_density":
            figurative_attn["question_density"]
            - literal_attn["question_density"],

        # -----------------------------------------
        # Lengths
        # -----------------------------------------

        "literal_question_tokens":
            literal_attn["question_tokens"],

        "figurative_question_tokens":
            figurative_attn["question_tokens"],

        "literal_context_tokens":
            literal_attn["context_tokens"],

        "figurative_context_tokens":
            figurative_attn["context_tokens"],
    })

# ============================================================
# SAVE
# ============================================================

out_df = pd.DataFrame(rows)

# out_df.to_csv(
#     OUTPUT_CSV,
#     index=False
# )

# print("\nSaved:", OUTPUT_CSV)
# print("N:", len(out_df))

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("QUESTION -> CONTEXT ATTENTION MASS")
print("=" * 80)

print(
    "Literal:",
    f"{100 * out_df['literal_context_mass'].mean():.2f}%"
)

print(
    "Figurative:",
    f"{100 * out_df['figurative_context_mass'].mean():.2f}%"
)

print(
    "Delta:",
    f"{100 * out_df['delta_context_mass'].mean():+.2f}%"
)

print("\n" + "=" * 80)
print("QUESTION -> PREVIOUS QUESTION TOKENS")
print("=" * 80)

print(
    "Literal:",
    f"{100 * out_df['literal_question_mass'].mean():.2f}%"
)

print(
    "Figurative:",
    f"{100 * out_df['figurative_question_mass'].mean():.2f}%"
)

print(
    "Delta:",
    f"{100 * out_df['delta_question_mass'].mean():+.2f}%"
)

print("\n" + "=" * 80)
print("QUESTION ATTENTION SHARE")
print("=" * 80)

print(
    "Literal:",
    f"{100 * out_df['literal_question_share'].mean():.2f}%"
)

print(
    "Figurative:",
    f"{100 * out_df['figurative_question_share'].mean():.2f}%"
)

print(
    "Delta:",
    f"{100 * out_df['delta_question_share'].mean():+.2f}%"
)

print("\n" + "=" * 80)
print("ATTENTION DENSITY")
print("=" * 80)

print(
    "Literal question:",
    out_df["literal_question_density"].mean()
)

print(
    "Figurative question:",
    out_df["figurative_question_density"].mean()
)

print(
    "Literal context:",
    out_df["literal_context_density"].mean()
)

print(
    "Figurative context:",
    out_df["figurative_context_density"].mean()
)