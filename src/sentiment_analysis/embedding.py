import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity

# 1. PROJECT ROOT
PROJECT_ROOT = os.getcwd()

# 2. MODEL INITIALIZATION
MODEL_NAME = "yiyanghkust/finbert-tone"
device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Model for embeddings
model_embed = AutoModel.from_pretrained(MODEL_NAME)
model_embed.to(device)
model_embed.eval()

# Model for hawkishness (classification head)
model_cls = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model_cls.to(device)
model_cls.eval()

# 3. UTILITY FUNCTIONS
def resolve_path(relative_path, project_root):
    return os.path.join(project_root, relative_path.replace("\\", os.sep).replace("/", os.sep))

# ---------- EMBEDDING (Hierarchical Mean Pooling) ----------

def embed_paragraph(text, tokenizer, model, max_length=512):

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True
    )["input_ids"][0]

    chunk_embeddings = []

    with torch.no_grad():
        # Reverse windowing (start from end)
        for i in range(len(tokens), 0, -max_length):
            start = max(i - max_length, 0)
            chunk = tokens[start:i]

            inputs = {
                "input_ids": chunk.unsqueeze(0).to(device),
                "attention_mask": torch.ones_like(chunk).unsqueeze(0).to(device)
            }

            outputs = model(**inputs)
            token_embeddings = outputs.last_hidden_state.squeeze(0)

            # Mean pooling token-level
            v_chunk = token_embeddings.mean(dim=0)
            chunk_embeddings.append(v_chunk)

    # Mean pooling chunk-level
    return torch.stack(chunk_embeddings).mean(dim=0)


def embed_document(text, tokenizer, model):

    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 0]

    paragraph_embeddings = [
        embed_paragraph(p, tokenizer, model)
        for p in paragraphs
    ]

    paragraph_embeddings = torch.stack(paragraph_embeddings)

    v_doc = paragraph_embeddings.mean(dim=0)
    intra_doc_variance = paragraph_embeddings.var(dim=0).mean()

    return v_doc.cpu().numpy(), intra_doc_variance.item()

# ---------- HAWKISHNESS (Chunk-based aggregation) ----------

def hawkishness_score(text, tokenizer, model, max_length=512):

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True
    )["input_ids"][0]

    chunks = tokens.split(max_length)
    chunk_scores = []

    with torch.no_grad():
        for chunk in chunks:
            inputs = {
                "input_ids": chunk.unsqueeze(0).to(device),
                "attention_mask": torch.ones_like(chunk).unsqueeze(0).to(device)
            }

            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze()

            # FinBERT order: [positive, neutral, negative]
            score_chunk = probs[2] - probs[0]
            chunk_scores.append(score_chunk)

    return torch.stack(chunk_scores).mean().item()

# 4. MAIN PROCESSING
def process_fomc_documents(doc_type, prefix):

    metadata_path = os.path.join(
        PROJECT_ROOT,
        "Deep_learning_gold_stock",
        "data",
        "raw",
        "text",
        doc_type,
        "metadata.csv"
    )

    meta = pd.read_csv(metadata_path, parse_dates=["date"])

    doc_embeddings, variances, dates, hawk_scores = [], [], [], []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc=doc_type):

        try:
            full_path = resolve_path(row["filepath"], PROJECT_ROOT)

            if not os.path.exists(full_path):
                raise FileNotFoundError(full_path)

            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            emb, var = embed_document(text, tokenizer, model_embed)
            score = hawkishness_score(text, tokenizer, model_cls)

            doc_embeddings.append(emb)
            variances.append(var)
            hawk_scores.append(score)
            dates.append(row["date"])

        except Exception as e:
            print(f"Erreur {doc_type}: {row['filepath']} -> {e}")
            doc_embeddings.append(np.zeros(768))
            variances.append(float('nan'))
            hawk_scores.append(float('nan'))
            dates.append(row["date"])

    emb_matrix = np.vstack(doc_embeddings)

    df = pd.DataFrame(
        emb_matrix,
        index=pd.to_datetime(dates),
        columns=[f"{prefix}_Emb_{i}" for i in range(emb_matrix.shape[1])]
    )

    df[f"{prefix}_IntraDocVar"] = variances

    # DocShift (cosine distance)
    shifts = [np.nan]
    for i in range(1, len(df)):
        sim = cosine_similarity(
            df.iloc[i-1:i, :768],
            df.iloc[i:i+1, :768]
        )[0][0]
        shifts.append(1 - sim)

    df[f"{prefix}_DocShift"] = shifts
    df[f"{prefix}_Hawkishness"] = hawk_scores

    return df

statements_df = process_fomc_documents("statements", "FOMC_Statement")
projections_df = process_fomc_documents("projections", "FOMC_Projection")

statements_df.to_pickle('data/processed/statements_embedding.pkl')
projections_df.to_pickle('data/processed/projections_embedding.pkl')
