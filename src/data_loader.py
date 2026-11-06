"""Read the medical CSV files and build the ChromaDB collections."""

import logging
from pathlib import Path

import chromadb
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

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
    """Read a CSV, take a fixed random sample, and fail clearly if the file is missing."""
    if not path.exists():
        logger.error(f"Missing dataset: {path}. Put the CSV files in the data/ folder.")
        raise FileNotFoundError(f"Missing dataset: {path}")

    logger.info(f"Loading data from {path.name}...")
    df = pd.read_csv(path)
    n = min(SAMPLE_SIZE, len(df))
    logger.info(f"Sampling {n} records from {path.name}.")
    return df.sample(n, random_state=RANDOM_STATE).reset_index(drop=True)


def _clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Replace missing values with empty strings, then make every column text."""
    return df.fillna("").astype(str)


def load_qna_dataframe(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the Q&A dataset and combine its columns into one searchable text field."""
    df = _clean_text(_read_sample(data_dir / QNA_CSV))

    df["combined_text"] = (
        "Question: " + df["Question"] + ". "
        + "Answer: " + df["Answer"] + ". "
        + "Type: " + df["qtype"] + ". "
    )
    logger.info("Medical Q&A dataset successfully processed.")
    return df


def load_device_dataframe(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the medical device dataset and combine its columns into one searchable text field."""
    df = _clean_text(_read_sample(data_dir / DEVICE_CSV))

    df["combined_text"] = (
        "Device Name: " + df["Device_Name"] + ". "
        + "Model: " + df["Model_Number"] + ". "
        + "Manufacturer: " + df["Manufacturer"] + ". "
        + "Indications: " + df["Indications_for_Use"] + ". "
        + "Contraindications: " + df["Contraindications"]
    )
    logger.info("Medical device dataset successfully processed.")
    return df


def _to_metadata(df: pd.DataFrame) -> list[dict]:
    """Convert each row to a metadata dict, leaving out the combined text column."""
    return df.drop(columns=["combined_text"]).to_dict(orient="records")


def _build_collection(client, name: str, df: pd.DataFrame, rebuild: bool):
    """Create the collection and index its documents, or reuse it if it already has data."""
    if rebuild:
        try:
            logger.info(f"Rebuild requested: deleting collection '{name}' if it exists...")
            client.delete_collection(name)
        except Exception:
            pass  # the collection did not exist yet

    collection = client.get_or_create_collection(name=name)
    if collection.count() == 0:
        logger.info(f"Indexing {len(df)} documents into '{name}'...")
        collection.add(
            documents=df["combined_text"].tolist(),
            metadatas=_to_metadata(df),
            ids=[f"{name}_{i}" for i in range(len(df))],
        )
        logger.info(f"Finished indexing '{name}'.")
    else:
        logger.info(f"Using existing collection '{name}' ({collection.count()} documents).")
    return collection


def init_collections(rebuild: bool = False) -> dict:
    """Open the local ChromaDB store and create both collections. Pass rebuild=True to re-index."""
    global _collections
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collections = {
        "qna": _build_collection(client, QNA_COLLECTION, load_qna_dataframe(), rebuild),
        "device": _build_collection(client, DEVICE_COLLECTION, load_device_dataframe(), rebuild),
    }
    logger.info("All collections are ready for retrieval.")
    return _collections


def get_collections() -> dict:
    """Return the cached collections, loading them on first use."""
    if _collections is None:
        return init_collections()
    return _collections


if __name__ == "__main__":
    init_collections()