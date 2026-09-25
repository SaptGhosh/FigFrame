# Data and Figures

This directory contains the data and figures used in our experiments and paper.

## I. Data

Each instance contains a fixed context paired with the original question and its figurative rewrites. The context remains unchanged across all versions, allowing us to isolate the effect of figurative framing.

The dataset contains the following fields:

- `context` — The user's story describing the situation in which the question is asked. This context remains unchanged across all rewrites.
- `reddit_verdict` — The verdict from the Reddit community. `1` indicates that the person is judged to be the asshole, while `0` indicates that the person is judged not to be the asshole.
- `literal_question` — The original, literal question asked by the user.
- `hyperbole_rewrite` — A rewrite of the original question using hyperbole.
- `metaphor_rewrite` — A rewrite of the original question using metaphor.
- `sarcasm_rewrite` — A rewrite of the original question using sarcasm.
- `simile_rewrite` — A rewrite of the original question using simile.

## II. Figures

This directory also contains the graphs and pipeline figure used in the paper.
