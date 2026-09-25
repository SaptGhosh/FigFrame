# Code

This directory contains the code used for data preparation, evaluation, and analysis.

## I. Data Extraction

Scripts for extracting and preparing samples from the datasets used in our experiments:

- `truthfulness_data.py` — Extracts samples from TruthfulQA for the **truthfulness** domain.
- `resume_atlas_data.py` — Extracts samples from ResumeAtlas for the **resume review** domain.
- `review_arena_data.py` — Extracts samples from ReviewArena for the **scientific peer review** domain.

## II. Domain Evaluation

Scripts for running the evaluation across the five additional domains considered in our experiments:

- `mathematical_reasoning_eval.py` — Mathematical reasoning
- `scientific_reasoning_eval.py` — Scientific reasoning
- `truthfulness_eval.py` — Truthfulness
- `peer_review_eval.py` — Scientific peer review
- `resume_review_eval.py` — Resume review

## III. Attention Analysis

- `test_attention.py` — Runs the attention-based analysis used to examine differences in model attention patterns between literal and figurative prompts.
