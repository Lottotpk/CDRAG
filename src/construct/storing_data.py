# requires transformers>=4.51.0

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from vectordb import VectorDB
import logging
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

model_embed = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto", "torch_dtype": torch.float16},
    tokenizer_kwargs={"padding_side": "left"},
)


def main():
    vectordb = VectorDB()

    # Get json
    with open("./data/MINE.json", "r") as f:
        data = json.load(f)
    
    for i in range(len(data)):
        raw_text = data[i]['content']
    
        # Text Chunking and Chunk embedding
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=250,
        #     chunk_overlap=25,
        # )
        # chunk = text_splitter.split_text(raw_text)
        chunk = raw_text.split(".")
        embedded = model_embed.encode(chunk, convert_to_tensor=True)
        logging.info(f"\tEmbedded shape = {embedded.shape}")

        for j in range(len(chunk)):
            # Add to DB
            vectordb.add_vector(chunk[j], embedded[j], (data[i]['topic']))
        
    logging.info(f"Saving to json...")
    vectordb.save_to_pt()
    logging.info(f"Saving successful")


if __name__ == "__main__":
    main()