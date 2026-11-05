
#Read the medical CSV files and build the ChromaDB collections 

from pathlib import Path
import chromadb
import pandas as pd
import logging 


# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) 

# 1. DYNAMIC PATHING: Automatically resolve file paths relative to this script's location 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

QNA_CSV = "medical_qa.csv"
DEVICE_CSV = "medical_device.csv"
QNA_COLLECTION = "medical_qna"
DEVICE_COLLECTION = "medical_device_manual"

SAMPLE_SIZE = 500
RANDOM_STATE = 0

# Global cache to hold the database connections in memory
_collections = None


def _read_sample(path: Path) -> pd.DataFrame:
    """Reads a CSV file into a Pandas DataFrame, samples it, and handles missing files."""
    if not path.exists():
        logger.error(f"Missing dataset: {path}. Put the Kaggle CSV files in the data/ folder.")
        raise FileNotFoundError(f"Missing dataset: {path}. Put the Kaggle CSV files in the data/ folder.")
    logger.info(f"Loading data from {path.name}...")
    df = pd.read_csv(path)
    n = min(SAMPLE_SIZE, len(df))
    # Randomly sample the data so the database builds quickly for testing
    logger.info(f"Sampling {n} records from {path.name}.")
    return df.sample(n, random_state=RANDOM_STATE).reset_index(drop=True)


def load_qna_dataframe(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Loads the Q&A dataset and combines columns into a single searchable text block."""
    df = _read_sample(data_dir / QNA_CSV)
    df["combined_text"] = (
        "Question: " + df["Question"].astype(str) + ". "
        + "Answer: " + df["Answer"].astype(str) + ". "
        + "Type: " + df["qtype"].astype(str) + ". "
    )
    logger.info("Medical Q&A dataset successfully processed.")
    return df


def load_device_dataframe(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Loads the Medical Devices dataset and combines hardware details into a single text block."""
    df = _read_sample(data_dir / DEVICE_CSV)
    df["combined_text"] = (
        "Device Name: " + df["Device_Name"].astype(str) + ". "
        + "Model: " + df["Model_Number"].astype(str) + ". "
        + "Manufacturer: " + df["Manufacturer"].astype(str) + ". "
        + "Indications: " + df["Indications_for_Use"].astype(str) + ". "
        + "Contraindications: " + df["Contraindications"].fillna("None").astype(str)
    )
    logger.info("Medical Devices dataset successfully processed.")
    return df


def _to_metadata(df: pd.DataFrame) -> list[dict]:
    """
    Cleans DataFrame for ChromaDB metadata compatibility.
    ChromaDB metadata values must be simple strings, integers, or floats.
    """
    return (
        df.drop(columns=["combined_text"])
        .fillna("")
        .astype(str)
        .to_dict(orient="records")
    )


def _build_collection(client, name: str, df: pd.DataFrame, rebuild: bool):
    """Creates a new ChromaDB collection or returns an existing one."""
    if rebuild:
        try:
            logger.info(f"Rebuild flagged: Deleting existing collection '{name}' if it exists...")
            client.delete_collection(name)
        except Exception:
            pass  # collection did not exist yet

    collection = client.get_or_create_collection(name=name)
    if collection.count() == 0:
        logger.info(f"Indexing {len(df)} documents into '{name}'...")
        collection.add(
            documents=df["combined_text"].tolist(),
            metadatas=_to_metadata(df),
            ids=[f"{name}_{i}" for i in range(len(df))],
        )
        logger.info(f"Successfully finished indexing '{name}'.")
    else:
        logger.info(f"Using existing collection '{name}' ({collection.count()} documents)")
    return collection


def init_collections(rebuild: bool = False) -> dict:
    """
    Initializes the local ChromaDB client and creates the two required collections.
    Set rebuild=True to delete and re-index the collections from scratch.
    """
    global _collections
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collections = {
        "qna": _build_collection(client, QNA_COLLECTION, load_qna_dataframe(), rebuild),
        "device": _build_collection(client, DEVICE_COLLECTION, load_device_dataframe(), rebuild),
    }
    logger.info("All collections are ready for retrieval.")
    return _collections


def get_collections() -> dict:
    """Returns the cached collections so the agent doesn't reload the database on every query.""" 
    if _collections is None:
        return init_collections()
    return _collections

if __name__ == "__main__":
    init_collections()