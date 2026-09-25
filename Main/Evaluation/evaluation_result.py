import pandas as pd

# ============================================================
# Read data - change file name here
# ============================================================

df = pd.read_csv("")


cols = ["literal_judgement", "figurative_judgement"]

for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=cols)

df = df[
    df["literal_judgement"].between(1, 7) &
    df["figurative_judgement"].between(1, 7)
].copy()

total = len(df)

L = df["literal_judgement"]
F = df["figurative_judgement"]



# 1 = definitely NOT asshole
# 2 = moderately disagree
# 3 = slightly disagree
# 4 = neutral
# 5 = slightly agree
# 6 = moderately agree
# 7 = definitely asshole
#
# Therefore:
# F > L = harsher
# F < L = more lenient


# ============================================================
# 1. Perturbation Rate
# ============================================================

p_total_count = (F != L).sum()

p_harsh_count = (F > L).sum()

p_lenient_count = (F < L).sum()


P_total = p_total_count / total
P_harsh = p_harsh_count / total
P_lenient = p_lenient_count / total


# ============================================================
# 2. Flip Rate
# ============================================================

def category(x):
    if x <= 3:
        return "not_asshole"
    elif x == 4:
        return "neutral"
    else:
        return "asshole"


lit_cat = L.apply(category)
fig_cat = F.apply(category)

# Any category change
flip_mask = lit_cat != fig_cat

F_total_count = flip_mask.sum()


# ============================================================
# Flip Direction
# ============================================================

# Harsh flip = category changed AND score increased
harsh_flip_mask = flip_mask & (F > L)

# Lenient flip = category changed AND score decreased
lenient_flip_mask = flip_mask & (F < L)


F_harsh_count = harsh_flip_mask.sum()
F_lenient_count = lenient_flip_mask.sum()


F_total = F_total_count / total
F_harsh = F_harsh_count / total
F_lenient = F_lenient_count / total


# ============================================================
# Print results
# ============================================================

print(f"Total examples: {total}")

print("\n==============================")
print("PERTURBATION")
print("==============================")

print(
    f"P_total   : {p_total_count}/{total} "
    f"= {P_total:.2%}"
)

print(
    f"P_harsh   : {p_harsh_count}/{total} "
    f"= {P_harsh:.2%}"
)

print(
    f"P_lenient : {p_lenient_count}/{total} "
    f"= {P_lenient:.2%}"
)


print("\n==============================")
print("FLIPS")
print("==============================")

print(
    f"F_total   : {F_total_count}/{total} "
    f"= {F_total:.2%}"
)

print(
    f"F_harsh   : {F_harsh_count}/{total} "
    f"= {F_harsh:.2%}"
)

print(
    f"F_lenient : {F_lenient_count}/{total} "
    f"= {F_lenient:.2%}"
)


# ============================================================
# sanity checks
# ============================================================

assert p_total_count == p_harsh_count + p_lenient_count
assert F_total_count == F_harsh_count + F_lenient_count