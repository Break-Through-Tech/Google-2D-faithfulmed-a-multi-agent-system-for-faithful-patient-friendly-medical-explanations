"""
eda_medquad.py
Exploratory data analysis + predictive-feature analysis for the parsed MedQuAD dataset.

Usage:
    python eda_medquad.py medquad_full.csv output_dir/
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_bar(series, title, xlabel, ylabel, fname, outdir, top_n=None, horizontal=True):
    data = series if top_n is None else series.head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(data))))
    if horizontal:
        ax.barh(data.index.astype(str)[::-1], data.values[::-1], color="#3b6ea5")
        ax.set_xlabel(ylabel)
    else:
        ax.bar(data.index.astype(str), data.values, color="#3b6ea5")
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=140)
    plt.close(fig)


def save_hist(series, title, xlabel, fname, outdir, bins=50, clip_upper=None):
    data = series.dropna()
    if clip_upper:
        data = data.clip(upper=clip_upper)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins=bins, color="#3b6ea5", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=140)
    plt.close(fig)


def main(csv_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    report_lines = []

    def log(x=""):
        print(x)
        report_lines.append(str(x))

    df = pd.read_csv(csv_path, low_memory=False)
    df["document_id"] = df["document_id"].astype(str)

    # ---------------------------------------------------------------
    log("MEDQUAD — SUMMARY STATISTICS")
    log(f"Total QA pairs: {len(df):,}")
    log(f"Unique source documents: {df['document_id'].nunique():,}")
    log(f"Unique focus topics (disease/drug/entity): {df['focus'].nunique():,}")
    log(f"Unique question types (qtype): {df['qtype'].nunique():,}")
    log(f"Source folders/sites: {df['source_site'].nunique():,}")

    # ---------------------------------------------------------------
    log("\n--- QA pairs per source site ---")
    by_source = df["source_site"].value_counts()
    log(by_source.to_string())
    save_bar(by_source, "QA pairs per NIH source", "Source", "# QA pairs",
              "01_pairs_per_source.png", outdir)

    # ---------------------------------------------------------------
    log("\n--- Missing/empty answers by source (copyright redaction) ---")
    empty_by_source = df.groupby("source_site")["answer_is_empty"].mean().sort_values(ascending=False) * 100
    log(empty_by_source.round(1).to_string())
    log(f"\nOverall empty-answer rate: {df['answer_is_empty'].mean()*100:.1f}%")
    save_bar(empty_by_source, "Share of answers redacted (empty), by source (%)", "Source",
              "% empty", "02_empty_answer_rate.png", outdir)

    # ---------------------------------------------------------------
    log("\n--- Top 20 question types (qtype) ---")
    qtype_counts = df["qtype"].value_counts()
    log(qtype_counts.head(20).to_string())
    save_bar(qtype_counts, "Top 20 question types", "qtype", "# QA pairs",
              "03_top_qtypes.png", outdir, top_n=20)

    # ---------------------------------------------------------------
    log("\n--- Focus category (Disease/Drug/Other), where annotated ---")
    cat_counts = df["focus_category"].fillna("(not annotated)").replace("", "(not annotated)").value_counts()
    log(cat_counts.to_string())

    # ---------------------------------------------------------------
    log("\n--- Question / answer length stats (in words) ---")
    log(df[["question_len_words", "answer_len_words"]].describe().round(1).to_string())
    save_hist(df["question_len_words"], "Question length distribution (words)",
              "Words per question", "04_question_length_hist.png", outdir, clip_upper=40)
    save_hist(df[df["answer_len_words"] > 0]["answer_len_words"],
              "Answer length distribution (words, non-empty only)",
              "Words per answer", "05_answer_length_hist.png", outdir, clip_upper=1500)

    # ---------------------------------------------------------------
    log("\n--- Top 15 UMLS semantic types ---")
    sem_types = df["umls_semantic_types"].dropna().str.split("|").explode()
    sem_types = sem_types[sem_types != ""]
    top_sem = sem_types.value_counts().head(15)
    log(top_sem.to_string())
    save_bar(top_sem, "Top 15 UMLS semantic types (focus concepts)", "Semantic type code",
              "# occurrences", "06_top_semantic_types.png", outdir, top_n=15)

    # ---------------------------------------------------------------
    log("\n--- Duplicate questions ---")
    dup_q = df["question"].str.strip().str.lower()
    n_dupes = dup_q.duplicated().sum()
    log(f"Exact duplicate question strings: {n_dupes:,} ({n_dupes/len(df)*100:.1f}% of all rows)")

    # ---------------------------------------------------------------
    log("\n--- QA pairs per document (how many Q&As per disease/drug page) ---")
    per_doc = df.groupby(["source_site", "document_id"]).size()
    log(per_doc.describe().round(1).to_string())

    # =================================================================
    section_title = "PREDICTIVE FEATURE ANALYSIS"
    log("\n" + "=" * 70)
    log(section_title)
    log("=" * 70)

    # Task A: which words in the QUESTION are most predictive of qtype?
    # Restrict to qtypes with enough support so the model is meaningful.
    vc = df["qtype"].value_counts()
    keep_qtypes = vc[vc >= 150].index
    sub = df[df["qtype"].isin(keep_qtypes) & df["question"].str.len().gt(0)].copy()

    log(f"\nTask A: Predict qtype from question text")
    log(f"Using {len(sub):,} rows across {len(keep_qtypes)} qtypes with >=150 examples each.")

    X_train, X_test, y_train, y_test = train_test_split(
        sub["question"], sub["qtype"], test_size=0.2, random_state=42, stratify=sub["qtype"]
    )
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)
    acc = accuracy_score(y_test, preds)
    log(f"Test accuracy: {acc:.3f}  (baseline if always predicting the most common class: "
        f"{vc.iloc[0]/vc[keep_qtypes].sum():.3f})")

    log("\nTop 8 most predictive words/bigrams per qtype (logistic regression coefficients):")
    feat_names = np.array(tfidf.get_feature_names_out())
    for i, cls in enumerate(clf.classes_):
        top_idx = np.argsort(clf.coef_[i])[-8:][::-1]
        top_terms = feat_names[top_idx]
        log(f"  {cls:20s} -> {', '.join(top_terms)}")

    # Task B: which structured/engineered features predict whether an answer is redacted (empty)?
    log(f"\nTask B: Predict whether an answer is empty (copyright-redacted) from structured metadata")
    feat_df = df.copy()
    le_source = LabelEncoder()
    le_qtype = LabelEncoder()
    feat_df["source_enc"] = le_source.fit_transform(feat_df["source_site"].astype(str))
    feat_df["qtype_enc"] = le_qtype.fit_transform(feat_df["qtype"].astype(str))

    feature_cols = ["source_enc", "qtype_enc", "question_len_words", "num_cuis",
                     "num_semantic_types", "num_synonyms"]
    Xb = feat_df[feature_cols]
    yb = feat_df["answer_is_empty"]

    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        Xb, yb, test_size=0.2, random_state=42, stratify=yb
    )
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(Xb_train, yb_train)
    yb_pred = rf.predict(Xb_test)
    log(f"Test accuracy: {accuracy_score(yb_test, yb_pred):.3f}")
    log("\nFeature importances (Random Forest):")
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    log(importances.round(3).to_string())
    log("\n(Interpretation: this basically re-discovers that 'source' almost fully determines "
        "redaction, since redaction was applied at the source-collection level, not randomly. "
        "qtype and text-length features add only marginal signal on top of that.)")

    save_bar(importances, "Feature importance for predicting redacted answers", "Feature",
              "Importance", "07_feature_importance_redaction.png", outdir, horizontal=True)

    # ---------------------------------------------------------------
    with open(os.path.join(outdir, "eda_report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nAll charts + eda_report.txt written to: {outdir}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "medquad_full.csv"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "eda_output"
    main(csv_path, outdir)
