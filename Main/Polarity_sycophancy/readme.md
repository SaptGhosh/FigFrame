# Polarity and Sycophancy Evaluation

This directory contains the code used to test whether model judgments are affected by the polarity of the question.

The original AITA question is reframed using two different polarities:

1. **Negative polarity** — Replaces "AITA" with "Was I in the wrong?"
2. **Positive polarity** — Replaces "AITA" with "Was I in the right?"

This experiment examines how model judgments change when the same underlying action is framed from either a positive or negative perspective, allowing us to test whether the observed effects can be attributed to sycophantic agreement with the user's framing.

The scripts then run the selected model on the polarity-controlled questions and save its judgments to a CSV file.
