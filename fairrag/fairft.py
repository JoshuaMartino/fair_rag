"""FairFT baseline (Zhang et al., 2025b), adapted to this task.

FairFT fine-tunes the dense retriever so that its ranking of the retrieved
passages agrees with how much each passage helps a frozen LLM give the
expected output:

    P_R(d | q)     = softmax_d( s(d, q) / gamma )             retriever
    Q_LM(d | q, a) = softmax_d( P_LM(a | d, q) / beta )       frozen LLM
    loss           = mean_q KL( P_R || Q_LM )

with s the cosine similarity, gamma = beta = 0.1, the top 20 passages per
query, Adam and a small learning rate (values from Zhang et al.).

What the Fair RAG paper does not say, and this file therefore decides:

* The expected output a. Here it is the correct occupation of a positive
  training biography, and P_LM(a | d, q) is the softmax probability of that
  occupation among the three candidates when d is the only context passage.
* The training queries. Here: the positive rows of the CGLC training set.
* The candidate passages are retrieved once with the original encoder and
  kept fixed (Zhang et al. refresh the index during training).

So treat results from this file as an approximation of the FairFT rows, not
as a replication of them. FairFT cannot be applied to BM25.

Workflow
--------
1. ``python fairft.py``                     trains and saves the encoder
2. rebuild the dense index with it          index.create_dense_index(
                                                wiki_df, "wiki_people_fairft",
                                                model_name="models/bge-fairft")
3. rerun run_retrieval_evaluation_main.py with dense_index_path,
   dense_model_name and output_root ("results/retrieval_fairft") changed
4. run_generation_main.py then picks the slates up as the "fairft" system
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from prompt import build_example_prompt


def retrieval_log_distribution(similarities: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """log P_R(d | q) over the candidate passages (last dimension)."""
    return torch.log_softmax(similarities / gamma, dim=-1)


def llm_log_distribution(answer_probabilities: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """log Q_LM(d | q, a) over the candidate passages (last dimension)."""
    return torch.log_softmax(answer_probabilities / beta, dim=-1)


def fairft_loss(
    similarities: torch.Tensor,
    answer_probabilities: torch.Tensor,
    gamma: float = 0.1,
    beta: float = 0.1,
) -> torch.Tensor:
    """KL(P_R || Q_LM), averaged over queries.

    similarities, answer_probabilities : (queries, passages) or (passages,)
    """
    log_p = retrieval_log_distribution(similarities, gamma)
    log_q = llm_log_distribution(answer_probabilities.to(log_p.dtype), beta)
    kl = torch.sum(log_p.exp() * (log_p - log_q), dim=-1)
    return kl.mean()


def answer_probability(scores: dict, answer: str) -> float:
    """Softmax probability of ``answer`` among the scored candidates."""
    values = torch.tensor(list(scores.values()), dtype=torch.float64)
    probabilities = torch.softmax(values, dim=0)
    return float(probabilities[list(scores).index(answer)])


def collect_answer_probabilities(
    queries,
    pool,
    occupations,
    scorer,
    cache_path=None,
    seed=42,
    max_passage_words=None,
):
    """P_LM(a | d, q) for every (positive query, candidate passage) pair.

    scorer(prompt, candidates) -> {candidate: score}

    Returns {bio_id: [probability per passage, in pool order]}. With
    ``cache_path`` the result is written as JSON lines and reused on rerun,
    since this is the expensive, LLM-bound part.
    """
    done = {}
    if cache_path is not None and Path(cache_path).exists():
        with open(cache_path, encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    done[record["bio_id"]] = record["probabilities"]

    handle = None
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        handle = open(cache_path, "a", encoding="utf-8")

    try:
        for row in queries.to_dict("records"):
            bio_id = row["bio_id"]
            if int(row["y"]) != 1 or bio_id in done or not pool.get(bio_id):
                continue

            probabilities = []
            for doc in pool[bio_id]:
                prompt, candidates = build_example_prompt(
                    row,
                    [doc],
                    occupations,
                    seed=seed,
                    max_passage_words=max_passage_words,
                )
                scores = scorer(prompt, candidates)
                probabilities.append(answer_probability(scores, row["target"]))

            done[bio_id] = probabilities
            if handle is not None:
                handle.write(
                    json.dumps({"bio_id": bio_id, "probabilities": probabilities}) + "\n"
                )
                handle.flush()
    finally:
        if handle is not None:
            handle.close()

    return done


def encode_with_gradient(encoder, texts, device):
    """Normalised sentence embeddings that keep the computation graph.

    ``encoder`` is a SentenceTransformer (or any module with ``tokenize`` and
    a forward pass returning ``{"sentence_embedding": tensor}``).
    """
    features = encoder.tokenize(list(texts))
    features = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in features.items()
    }
    embeddings = encoder(features)["sentence_embedding"]
    return torch.nn.functional.normalize(embeddings, dim=-1)


def train_fairft(
    encoder,
    queries,
    pool,
    answer_probabilities,
    epochs=1,
    batch_size=20,
    learning_rate=1e-5,
    gamma=0.1,
    beta=0.1,
    device=None,
    seed=42,
    log_every=10,
):
    """Fine-tune ``encoder`` with the FairFT objective. Returns the loss history.

    queries              DataFrame with ``bio_id`` and ``bio``
    pool                 {bio_id: [documents]}, the fixed candidate passages
    answer_probabilities {bio_id: [P_LM(a | d, q) per passage]}
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    encoder.to(device)
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    rows = [
        row
        for row in queries.to_dict("records")
        if row["bio_id"] in answer_probabilities and pool.get(row["bio_id"])
    ]
    if not rows:
        raise ValueError("No training queries have answer probabilities.")

    generator = torch.Generator().manual_seed(seed)
    history = []

    for epoch in range(epochs):
        order = torch.randperm(len(rows), generator=generator).tolist()

        for step, start in enumerate(range(0, len(order), batch_size), start=1):
            batch = [rows[i] for i in order[start:start + batch_size]]
            optimizer.zero_grad()
            batch_loss = 0.0

            # One query at a time keeps memory bounded: 1 + 20 texts carry
            # gradients per backward pass, and gradients add up over the batch.
            for row in batch:
                docs = pool[row["bio_id"]]
                target = torch.tensor(
                    answer_probabilities[row["bio_id"]], device=device
                )
                query_embedding = encode_with_gradient(encoder, [row["bio"]], device)
                doc_embeddings = encode_with_gradient(
                    encoder, [doc.content for doc in docs], device
                )
                similarities = (doc_embeddings @ query_embedding.T).squeeze(-1)

                loss = fairft_loss(similarities, target, gamma, beta) / len(batch)
                loss.backward()
                batch_loss += float(loss.item())

            optimizer.step()
            history.append(batch_loss)

            if log_every and step % log_every == 0:
                print(f"epoch {epoch + 1} step {step}: loss {batch_loss:.4f}")

    encoder.eval()
    return history


def main():
    # ------------------------------------------------------------------
    # Configuration: edit these values before running.
    # ------------------------------------------------------------------
    encoder_name = "BAAI/bge-large-en"
    llm_name = "google/gemma-2b"
    llm_dtype = "float16"

    train_queries_path = Path("datasets/train/train.csv")
    dense_index_path = "wiki_people"
    top_n = 20                      # passages per query (Zhang et al.)
    max_passage_words = 64

    epochs = 1
    batch_size = 20
    learning_rate = 1e-5
    gamma = beta = 0.1

    output_dir = Path("models/bge-fairft")
    cache_path = Path("results/fairft/answer_probabilities.jsonl")

    # ------------------------------------------------------------------
    from sentence_transformers import SentenceTransformer

    from generator import load_model, score_candidates
    from index import load_dense_index
    from retrieval import dense_retrieve
    from run_generation_main import load_queries

    queries = load_queries(train_queries_path)
    queries = queries[queries["y"] == 1].reset_index(drop=True)
    occupations = sorted(load_queries(train_queries_path)["target"].astype(str).unique())

    print("Retrieving candidate passages...")
    store = load_dense_index(dense_index_path)
    pool = dense_retrieve(
        store, queries[["bio_id", "bio"]], top_k=top_n, model_name=encoder_name
    )

    print(f"Scoring passages with {llm_name}...")
    model, tokenizer, device = load_model(llm_name, torch_dtype=llm_dtype)

    def scorer(prompt, candidates):
        return score_candidates(prompt, candidates, model, tokenizer, device)

    probabilities = collect_answer_probabilities(
        queries, pool, occupations, scorer,
        cache_path=cache_path, max_passage_words=max_passage_words,
    )
    # Free the GPU for the encoder.
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Fine-tuning the encoder...")
    encoder = SentenceTransformer(encoder_name)
    train_fairft(
        encoder, queries, pool, probabilities,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        gamma=gamma, beta=beta,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder.save(str(output_dir))
    print(f"Saved the tuned encoder to {output_dir}")


if __name__ == "__main__":
    main()
