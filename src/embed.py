'''
FaithfulMed Embeddings 
Task Owner: Renita 

Chunks and embeds the csv file produced from the medline plus glossary EDA

Setup:
    pip install google-adk
    export GEMINI_API_KEY=...      # free key from https://aistudio.google.com
'''

import pandas as pd 
from google import genai
import os
import time 

MAX_WORDS = 1500
BATCH_SIZE = 10 

client = genai.Client()

MODEL = "gemini-embedding-001"

# Creating chunks by grouping rows until they reach max words 
def create_chunks(df):
    chunked_rows = []

    current_entries = []
    current_row_ids = []
    current_length = 0
    chunk_id = 0

    for row_id, row in df.iterrows():
        text = (
            f"Title: {row['title']}\n\n"
            f"Summary:\n{row['summary']}"
        ).strip()

        separator_length = 2 if current_entries else 0
        added_length = len(text) + separator_length

        if current_entries and current_length + added_length > MAX_WORDS:
            chunked_rows.append({
                "row_id": current_row_ids,
                "chunk_id": chunk_id,
                "text": "\n\n".join(current_entries),
            })

            chunk_id += 1
            current_entries = []
            current_row_ids = []
            current_length = 0
            separator_length = 0
            added_length = len(text)

        current_entries.append(text)
        current_row_ids.append(row_id)
        current_length += added_length

    if current_entries:
        chunked_rows.append({
            "row_id": current_row_ids,
            "chunk_id": chunk_id,
            "text": "\n\n".join(current_entries),
        })

    return pd.DataFrame(chunked_rows)

# Embeds the text, if the we reach api limit, waits 60 seconds and retries 
# Retries the api call a maximum of 8 times 
def embed_text(text, max_retries=8):
    for attempt in range(max_retries + 1):
        try:
            results = client.models.embed_content(
                model=MODEL,
                contents=text
            )

            return [result.values for result in results.embeddings]

        except Exception as e:
            error_message = str(e)

            is_rate_limit = (
                "429" in error_message
                or "RESOURCE_EXHAUSTED" in error_message
            )

            if not is_rate_limit:
                raise

            if attempt == max_retries:
                raise

            wait_seconds = min(60 * (2 ** attempt), 300)

            print(
                f"Rate limit reached. Waiting {wait_seconds} seconds "
                f"before retrying ({attempt + 1}/{max_retries})..."
            )

            time.sleep(wait_seconds)

def main():
    output_file = "data/embeddings/medlineplus_embeddings.parquet"
    df = pd.read_csv("data/cleaned-medlineplus/medlineplus_topics.csv")

    chunks_df = create_chunks(df) 

    # If output file exists load finished embeddings otherwise create file 
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file):
        saved_df = pd.read_parquet(output_file)
        print(f"Loaded {len(saved_df)} saved embeddings")
    else:
        saved_df = pd.DataFrame(
            columns=["row_id", "chunk_id", "text", "embedding"]
        )

    completed_ids = set(saved_df["chunk_id"].tolist())

    pending_df = chunks_df[
        ~chunks_df["chunk_id"].isin(completed_ids)
    ]

    print(f"Already completed: {len(completed_ids)}")
    print(f"Remaining: {len(pending_df)}")

    # Sends chunks to be embedded in batches to not overwhelm the api
    for start in range(0, len(pending_df), BATCH_SIZE):
        batch = pending_df.iloc[start:start + BATCH_SIZE]
        texts = batch["text"].tolist()

        print(
            f"Embedding {start} to "
            f"{start + len(batch)} of {len(pending_df)} remaining"
        )

        try:
            embeddings = embed_text(texts)

            if len(embeddings) != len(batch):
                raise ValueError(
                    "Embedding count doesn't match batch size"
                )

            batch_results = batch.copy()
            batch_results["embedding"] = embeddings

            saved_df = pd.concat(
                [saved_df, batch_results],
                ignore_index=True
            )

            temp_file = output_file + ".tmp"
            saved_df.to_parquet(temp_file, index=False)
            os.replace(temp_file, output_file)

            print(
                f"Saved {len(saved_df)} / "
                f"{len(chunks_df)} embeddings"
            )

        except Exception as e:
            print(f"Batch failed: {e}")
            print("Previously saved embeddings are preserved.")
            print("Fix the issue and rerun main() to resume.")
            return

    print("Done! All embeddings are saved.")


if __name__ == "__main__": 
    main()