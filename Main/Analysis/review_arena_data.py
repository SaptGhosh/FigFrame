#!/usr/bin/env python3

import pandas as pd
from datasets import load_dataset

# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "Samarth0710/reviewarena"

CONFERENCES = [
    "iclr",
    "neurips",
    "icml",
    "emnlp",
    "colm",
]

N_PER_CONFERENCE = 100
SEED = 42

OUTPUT_FILE = "reviewarena_dataset_fullpaper.csv"


# ============================================================
# SAMPLE EACH CONFERENCE
# ============================================================

all_rows = []

for conference in CONFERENCES:

    print(f"\nLoading {conference.upper()}...")

    # Streaming avoids downloading the whole ReviewArena dataset
    ds = load_dataset(
        DATASET_NAME,
        split=conference,
        streaming=True
    )

    ds = ds.shuffle(
        seed=SEED,
        buffer_size=5000
    )

    collected = 0

    for row in ds:

        title = row.get("title")
        abstract = row.get("abstract")
        full_paper = row.get("markdown")
        year = row.get("year")
        decision = row.get("decision")

        # --------------------------------------------
        # REQUIRE FIELDS WE NEED
        # --------------------------------------------

        if (
            title is None
            or abstract is None
            or full_paper is None
        ):
            continue

        title = str(title).strip()
        abstract = str(abstract).strip()
        full_paper = str(full_paper).strip()

        if (
            not title
            or not abstract
            or not full_paper
        ):
            continue

        # Skip weird missing-string values
        if title.lower() == "nan":
            continue

        if abstract.lower() == "nan":
            continue

        if full_paper.lower() == "nan":
            continue

        # --------------------------------------------
        # SAVE SAMPLE
        # --------------------------------------------

        all_rows.append({
            "title": title,
            "abstract": abstract,
            "full_paper": full_paper,
            "year": year,
            "conference": conference.upper(),
            "decision": decision,
        })

        collected += 1

        if collected >= N_PER_CONFERENCE:
            break

    print(
        f"Collected {collected} samples "
        f"from {conference.upper()}"
    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(all_rows)

# Shuffle final dataset so conferences aren't grouped together
df = df.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("DATASET CREATED")
print("=" * 60)

print(f"Total samples: {len(df)}")
print(f"Saved to: {OUTPUT_FILE}")


# ============================================================
# SHOW DISTRIBUTION
# ============================================================

print("\nConference distribution:")
print(df["conference"].value_counts())

print("\nYear distribution:")
print(
    df.groupby(
        ["conference", "year"]
    ).size()
)

print("\nDecision distribution:")
print(
    df.groupby(
        ["conference", "decision"]
    ).size()
)


# ============================================================
# FULL PAPER LENGTHS
# ============================================================

df["paper_chars"] = df["full_paper"].str.len()

print("\nFull paper character lengths:")
print(
    df["paper_chars"].describe()
)


# ============================================================
# PRINT ONE SAMPLE
# ============================================================

sample = df.iloc[0]

print("\n" + "=" * 80)
print("SAMPLE")
print("=" * 80)

print(f"\nTITLE:\n{sample['title']}")

print(f"\nABSTRACT:\n{sample['abstract']}")

print(f"\nYEAR:\n{sample['year']}")

print(f"\nCONFERENCE:\n{sample['conference']}")

print(f"\nDECISION:\n{sample['decision']}")

print(
    f"\nFULL PAPER LENGTH:\n"
    f"{len(sample['full_paper'])} characters"
)

print("\nFULL PAPER (first 3000 characters):")
print(sample["full_paper"][:3000])