

from pathlib import Path

import chromadb
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

QNA_CSV = "medical_qa.csv"
DEVICE_CSV = "medical_device.csv"
QNA_COLLECTION = "medical_qna"
DEVICE_COLLECTION = "medical_device_manual"

SAMPLE_SIZE = 500
RANDOM_STATE = 0

_collections = None


def _read_sample(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}. Put the Kaggle CSV files in the data/ folder.")
    df = pd.read_csv(path)
    n = min(SAMPLE_SIZE, len(df))
    return df.sample(n, random_state=RANDOM_STATE).reset_index(drop=True)


def load_qna_dataframe(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = _read_sample(data_dir / QNA_CSV)
    df["combined_text"] = (
        "Question: " + df["Question"].astype(str) + ". "
        + "Answer: " + df["Answer"].astype(str) + ". "
        + "Type: " + df["qtype"].astype(str) + ". "
    )
    return df


def load_device_dataframe(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = _read_sample(data_dir / DEVICE_CSV)
    df["combined_text"] = (
        "Device Name: " + df["Device_Name"].astype(str) + ". "
        + "Model: " + df["Model_Number"].astype(str) + ". "
        + "Manufacturer: " + df["Manufacturer"].astype(str) + ". "
        + "Indications: " + df["Indications_for_Use"].astype(str) + ". "
        + "Contraindications: " + df["Contraindications"].fillna("None").astype(str)
    )
    return df


def _to_metadata(df: pd.DataFrame) -> list[dict]:
    # Chroma metadata values must be simple scalars, so stringify and remove NaNs.
    return (
        df.drop(columns=["combined_text"])
        .fillna("")
        .astype(str)
        .to_dict(orient="records")
    )


def _build_collection(client, name: str, df: pd.DataFrame, rebuild: bool):
    if rebuild:
        try:
            client.delete_collection(name)
        except Exception:
            pass  # collection did not exist yet

    collection = client.get_or_create_collection(name=name)
    if collection.count() == 0:
        print(f"Indexing {len(df)} documents into '{name}'...")
        collection.add(
            documents=df["combined_text"].tolist(),
            metadatas=_to_metadata(df),
            ids=[f"{name}_{i}" for i in range(len(df))],
        )
    else:
        print(f"Using existing collection '{name}' ({collection.count()} documents)")
    return collection


def init_collections(rebuild: bool = False) -> dict:
    """Create (or reuse) both ChromaDB collections and cache them.

    Set rebuild=True to delete and re-index the collections.
    """
    global _collections
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collections = {
        "qna": _build_collection(client, QNA_COLLECTION, load_qna_dataframe(), rebuild),
        "device": _build_collection(client, DEVICE_COLLECTION, load_device_dataframe(), rebuild),
    }
    return _collections


def get_collections() -> dict:
    """Return the cached collections, initializing them on first use."""
    if _collections is None:
        return init_collections()
    return _collections