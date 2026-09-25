from datasets import load_dataset

# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "ahmedheakl/resume-atlas"
N_SAMPLES = 500
SEED = 42
OUTPUT_FILE = "resume_atlas_sample_200.csv"

# ============================================================
# LOAD
# ============================================================

dataset = load_dataset(DATASET_NAME)

df = dataset["train"].to_pandas()

print("Total samples:", len(df))
print("Columns:", df.columns.tolist())

# Remove empty resumes
df = df.dropna(subset=["Text"]).copy()
df = df[df["Text"].str.strip() != ""]

# ============================================================
# RANDOM SAMPLE
# ============================================================

sample_df = df.sample(
    n=N_SAMPLES,
    random_state=SEED
).reset_index(drop=True)

# ============================================================
# SAVE
# ============================================================

sample_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Saved {len(sample_df)} resumes to {OUTPUT_FILE}")

print("\nCategory distribution:")
print(sample_df["Category"].value_counts())

print("\nFirst resume:")
print(sample_df.iloc[0]["Text"][:2000])