import pandas as pd
import chromadb
import os

def load_medical_data():
    print("LOG: Initializing ChromaDB Client...")
    # Store the database locally in the chroma_db folder
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # ---------------------------------------------------------
    # 1. Process Medical Q&A Dataset
    # ---------------------------------------------------------
    print("LOG: Loading Medical Q&A Dataset from CSV...")
    try:
        df_qa = pd.read_csv("data/medical_qa.csv").sample(500).reset_index()
        
        # Strip hidden spaces from the column headers just in case
        df_qa.columns = df_qa.columns.str.strip()
        
        print(f"LOG: Successfully loaded {len(df_qa)} rows from the Q&A dataset.")
    except Exception as e:
        print(f"ERROR: Could not read data/medical_qa.csv. Check if the file exists. Details: {e}")
        return
        
    # Using the exact Capitalized headers based on your terminal output
    combined_qa_text = "Question: " + df_qa['Question'].astype(str) + " \nAnswer: " + df_qa['Answer'].astype(str)
    
    print("LOG: Creating Medical Q&A Collection in ChromaDB...")
    qa_collection = client.get_or_create_collection(name="medical_Q_and_A")
    qa_collection.add(
        documents=combined_qa_text.tolist(),
        metadatas=df_qa.to_dict(orient="records"),
        ids=df_qa.index.astype(str).tolist()
    )
    print("LOG: Q&A Collection successfully populated.")
    
    # ---------------------------------------------------------
    # 2. Process Medical Devices Dataset
    # ---------------------------------------------------------
    print("LOG: Loading Medical Devices Dataset from CSV...")
    try:
        df_md = pd.read_csv("data/medical_devices.csv").sample(500).reset_index()
        df_md.columns = df_md.columns.str.strip()
        print(f"LOG: Successfully loaded {len(df_md)} rows from the Devices dataset.")
    except Exception as e:
         print(f"ERROR: Could not read data/medical_devices.csv. Check if the file exists. Details: {e}")
         return
         
    # Assuming standard headers from the Kaggle dataset
    combined_md_text = (
        "Device: " + df_md['Device_Name'].astype(str) +
        " \nManufacturer: " + df_md['Manufacturer'].astype(str) + 
        " \nIndications: " + df_md['Indications'].astype(str)
    )
    
    print("LOG: Creating Medical Devices Collection in ChromaDB...")
    md_collection = client.get_or_create_collection(name="medical_devices")
    md_collection.add(
        documents=combined_md_text.tolist(),
        metadatas=df_md.to_dict(orient="records"),
        ids=df_md.index.astype(str).tolist()
    )
    print("LOG: Devices Collection successfully populated.")
    
    print("LOG: SUCCESS! All datasets are loaded into the local ChromaDB.")

if __name__ == "__main__":
    load_medical_data()