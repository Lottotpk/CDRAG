# requires transformers>=4.51.0

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from vectordb import VectorDB
import logging
import PyPDF2
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

model_embed = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto", "torch_dtype": torch.float16},
    tokenizer_kwargs={"padding_side": "left"},
)


def query_retrieval(query: str, 
                    vectordb: VectorDB, 
                    model: SentenceTransformer = model_embed,
                    top_k: int = 5) -> list:
    """Retrieve the top k list of similar vector to the query.

    Parameters
    ----------
    query : str
        The input query from the user
    vectordb : VectorDB
        The vector database to search
    model : SentenceTransformer
        The transformer model to embed query into vector
    top_k : int
        The k number of returned list, default set to 5

    Returns
    -------
    list
        The top k-th most similar contexts to the query 
    """
    query_vector = model.encode(query, convert_to_tensor=True)
    return vectordb.get_topk_similar(query_vector, top_k)


def main():
    vectordb = VectorDB()

    srcdir = "./data/raw"
    for subdir in os.listdir(srcdir):
        if subdir in ["fin_docs", "tat_docs"]:
            continue
        subdir_name = os.path.join(srcdir, subdir)
        logging.info(f"processing {subdir_name} directory...")
        for path in os.listdir(subdir_name):
            logging.info(f"\tprocessing {path}")
            # Convert PDF to raw text
            pdf_path = os.path.join(subdir_name, path)
            raw_text = ""
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file, strict=False)
                for pg in reader.pages:
                    raw_text += pg.extract_text()
            
            # Text Chunking and Chunk embedding
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000,
                chunk_overlap=300,
            )
            chunk = text_splitter.split_text(raw_text)
            embedded = model_embed.encode(chunk, convert_to_tensor=True)
            logging.info(f"\tEmbedded shape = {embedded.shape}")

            # Add to DB
            for i in range(len(chunk)):
                # logging.info(f"{chunk[i][:10]}, {embedded[i]}, {path.strip(".pdf")}")
                vectordb.add_vector(chunk[i], embedded[i], [subdir.strip("_docs").strip("wiki_"), path.strip(".pdf")])
            
        logging.info(f"Saving to json...")
        vectordb.save_to_json()
        logging.info(f"Saving successful")


if __name__ == "__main__":
    main()