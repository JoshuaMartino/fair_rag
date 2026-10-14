"""Corpus (Humans of Wikipedia subset) and Bias-in-Bios evaluation / CGLC-training set construction (App. A.2)."""
import re
import numpy as np
import pandas as pd


def norm_occ(s):
    return re.sub(r"[\s\-]+", "_", str(s).strip().lower())


def norm_gender(x):
    if isinstance(x, str):
        return 1 if x.strip().lower() in {"female", "f", "1"} else 0
    return int(x)


def load_corpus(csv_path, max_words=150, n_pages=126_609, seed=42):
    """CSV columns: text, gender, title, occupation_level_3. Pages are split into <=150-word passages.
    Returns passage dataframe: pid, text, gender(0=male,1=female), title, occupation."""
    df = pd.read_csv(csv_path)
    df = df.assign(occupation=df["occupation_level_3"].map(norm_occ), gender=df["gender"].map(norm_gender))
    if len(df) > n_pages:
        df = df.sample(n_pages, random_state=seed)
    rows = []
    for t, text, gen, occ in zip(df["title"], df["text"], df["gender"], df["occupation"]):
        words = str(text).split()
        for i in range(0, max(len(words), 1), max_words):
            rows.append((" ".join(words[i:i + max_words]), gen, t, occ))
    out = pd.DataFrame(rows, columns=["text", "gender", "title", "occupation"])
    out.insert(0, "pid", np.arange(len(out)))
    return out


def load_bias_in_bios(split):
    """LabHC/bias_in_bios: hard_text, profession (ClassLabel), gender (0=male, 1=female)."""
    from datasets import load_dataset
    ds = load_dataset("LabHC/bias_in_bios", split=split)
    names = ds.features["profession"].names
    df = ds.to_pandas()
    return pd.DataFrame({"bio": df["hard_text"],
                         "profession": df["profession"].map(lambda i: norm_occ(names[i])),
                         "gender": df["gender"].astype(int)})


def eligible_occupations(bios, corpus_occs, min_per_gender):
    c = bios.groupby(["profession", "gender"]).size().unstack(fill_value=0)
    ok = c[(c[0] >= min_per_gender) & (c[1] >= min_per_gender)].index
    return sorted(o for o in ok if o in set(corpus_occs))


def build_eval_set(bios, occupations, n_per_cell, seed=42):
    """Per occupation: n_per_cell positives and negatives for each gender (4*n rows), balanced.
    Negatives are drawn from other eligible occupations. Two distractor occupations are sampled per row."""
    rng = np.random.default_rng(seed)
    b = bios[bios.profession.isin(occupations)].reset_index().rename(columns={"index": "bio_id"})
    rows = []
    for oi, o in enumerate(occupations):
        for gen in (0, 1):
            pos = b[(b.profession == o) & (b.gender == gen)].sample(n_per_cell, random_state=int(rng.integers(1e9)))
            neg = b[(b.profession != o) & (b.gender == gen)].sample(n_per_cell, random_state=int(rng.integers(1e9)))
            for y, part in ((1, pos), (0, neg)):
                for r in part.itertuples():
                    others = [x for x in occupations if x != o]
                    d1, d2 = rng.choice(others, 2, replace=False)
                    rows.append((r.bio_id, r.bio, r.profession, r.gender, o, oi, y, d1, d2))
    return pd.DataFrame(rows, columns=["bio_id", "bio", "profession", "gender", "target", "occ", "y", "d1", "d2"])


def build_inference_set(bios, corpus_occs, seed=42):
    occs = eligible_occupations(bios, corpus_occs, 100)
    return build_eval_set(bios, occs, 100, seed), occs        # 22 x 400 rows in the paper


def build_cglc_train_set(bios_test, occs, seed=42):
    keep = [o for o in occs if eligible_occupations(bios_test, [o], 25)]
    df = build_eval_set(bios_test, keep, 25, seed)             # 100 rows / occupation
    df["occ"] = df["target"].map({o: i for i, o in enumerate(occs)})   # keep ids aligned with the inference set
    return df, keep
