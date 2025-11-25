"""
PyKEEN Link Prediction from a CSV of Triples
-------------------------------------------

Teaching script for students:

1. Load triples from a CSV (subject, predicate, object)
2. Build a TriplesFactory and split into train/val/test
3. Train a TransE model for link prediction
4. Evaluate with MRR and Hits@k
5. Ask questions like: (head, relation, ?tail) and (?head, relation, tail)

Run:
    pip install pykeen pandas
    python link_prediction_pykeen.py
"""
# pip install pykeen pandas
import torch
import pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline


def main():
    # --------------------------------------------------------------
    # 1. Load triples from CSV
    # --------------------------------------------------------------
    # Your CSV should have columns: subject,predicate,object
    csv_path = "my_kg.csv"     # <-- change to your file path

    print(f"Loading triples from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Quick sanity check
    print("\nFirst few rows of the CSV:")
    print(df.head())

    # Extract just the three columns and convert to numpy array
    labeled_triples = df[["subject", "predicate", "object"]].values

    # --------------------------------------------------------------
    # 2. Build TriplesFactory and create train/val/test splits
    # --------------------------------------------------------------
    print("\nBuilding TriplesFactory...")
    tf_all = TriplesFactory.from_labeled_triples(labeled_triples)

    print(f"#entities:  {tf_all.num_entities}")
    print(f"#relations: {tf_all.num_relations}")
    print(f"#triples total: {tf_all.num_triples}")

    # Split: 80% train, 10% val, 10% test (you can change proportions)
    training, testing, validation = tf_all.split(
        ratios=(0.8, 0.1, 0.1),
        random_state=42,
    )

    print("\nSplit sizes:")
    print(f"  train: {training.num_triples}")
    print(f"  valid: {validation.num_triples}")
    print(f"  test : {testing.num_triples}")

    # --------------------------------------------------------------
    # 3. Train a link prediction model
    # --------------------------------------------------------------
    print("\nTraining TransE model...\n")

    result = pipeline(
        training=training,
        validation=validation,
        testing=testing,
        model="TransE",
        model_kwargs=dict(
            embedding_dim=100,
            margin=1.0,
        ),
        negative_sampler="basic",
        optimizer="Adam",
        optimizer_kwargs=dict(lr=1e-3),
        training_kwargs=dict(
            num_epochs=100,   # for a demo; increase for real use
            batch_size=1024,
        ),
        stopper="early",
        stopper_kwargs=dict(
            frequency=5,
            patience=3,
            relative_delta=0.002,
        ),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    model = result.model
    print("\nTraining finished.")

    # --------------------------------------------------------------
    # 4. Evaluate on the test split
    # --------------------------------------------------------------
    print("\nEvaluating on the test split...")
    metrics = result.metric_results.to_dict()

    print("\nFull metric dict:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")

    print("\nKey metrics:")
    mrr = metrics["both.realistic.inverse_harmonic_mean_rank"]
    hits1 = metrics["both.realistic.hits_at_1"]
    hits3 = metrics["both.realistic.hits_at_3"]
    hits10 = metrics["both.realistic.hits_at_10"]
    print(f"  MRR   : {mrr:.4f}")
    print(f"  Hits@1: {hits1:.4f}")
    print(f"  Hits@3: {hits3:.4f}")
    print(f"  Hits@10: {hits10:.4f}")

    # --------------------------------------------------------------
    # 5. Make interactive link prediction queries
    # --------------------------------------------------------------
    print("\n=== Link Prediction Examples ===")

    # Invert dictionaries so we can move between IDs and labels
    ent2id = training.entity_to_id
    rel2id = training.relation_to_id
    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}

    # ---- Example A: (head, relation, ?tail) ----
    # Pick any labels that exist in your CSV
    # e.g., "Alice", "has", "?" or "usa", "accuses", "?"
    example_head = list(ent2id.keys())[0]
    example_rel = list(rel2id.keys())[0]

    print(f"\nExample A: ({example_head}, {example_rel}, ?)")
    h_id = ent2id[example_head]
    r_id = rel2id[example_rel]

    with torch.no_grad():
        tail_df = model.get_tail_prediction_df(
            head_id=h_id,
            relation_id=r_id,
            triples_factory=training,
            k=10,  # top 10 predictions
        )

    print("\nTop 10 predicted tails:")
    print(tail_df)

    # ---- Example B: (?head, relation, tail) ----
    example_tail = list(ent2id.keys())[1]
    print(f"\nExample B: (?, {example_rel}, {example_tail})")
    t_id = ent2id[example_tail]

    with torch.no_grad():
        head_df = model.get_head_prediction_df(
            relation_id=r_id,
            tail_id=t_id,
            triples_factory=training,
            k=10,
        )

    print("\nTop 10 predicted heads:")
    print(head_df)

    # Optionally: show how to decode IDs manually
    print("\nDecoding the first predicted triple from Example A:")
    best_tail_id = int(tail_df["tail_id"].iloc[0])
    print(
        f"  ({id2ent[h_id]}, {id2rel[r_id]}, {id2ent[best_tail_id]}) "
        f"with score {tail_df['score'].iloc[0]:.4f}"
    )


if __name__ == "__main__":
    main()

