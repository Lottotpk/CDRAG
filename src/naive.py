from transformers import AutoTokenizer, AutoModelForCausalLM
from construct.vectordb import VectorDB
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import logging
import json
from tqdm.auto import tqdm
from datasets import load_dataset
from config import PROMPT_DICT

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Running on {device}")
# embedding model
logging.info("Initializing embedding transformer model...")
model_embed = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto", "torch_dtype": torch.float16},
    tokenizer_kwargs={"padding_side": "left"},
)
logging.info("Embedding transformer model initialized")

# base LLM model
# Note: run `huggingface-cli login` first in the terminal before using
logging.info("Initializing LLM model...")

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
logging.info("LLM model initialized.")

# Our vector database data structure
logging.info("Loading vector database...")
vectordb = VectorDB()
logging.info("Vector database successfully retrieved.")


def prompt_format(query: str, retrieved_context: list):
    docs = set()
    prompt_template = PROMPT_DICT[vectordb.get_metadata(retrieved_context[0][0])[0]]
    user_content = f" ### Context:\n"
    for msg, _ in retrieved_context:
        docs.add(vectordb.get_metadata(msg)[1])
        user_content += f"- {msg}\n"
    user_content +="### Question: {query}\n ### Response:"
    messages = [
            {
                "role": "user",
                "content": prompt_template["system"] + "\n" + prompt_template["user_1"],
            },
            {"role": "assistant", "content": prompt_template["assistant_1"]},
            {"role": "user", "content": user_content},
        ]
    return messages, docs


def ask(prompt: str):
    # Retrieve relevant query
    embedded_text = model_embed.encode(prompt, convert_to_tensor=True)
    retrieved = vectordb.get_topk_similar(embedded_text)
    
    # prepare the model input
    messages, docs = prompt_format(prompt, retrieved)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    return content, list(docs)


def eval():
    # topics = ["fin", "paper_text", "paper_tab", "tat", "feta", "nq"]
    topics = ["paper_text", "paper_tab", "feta", "nq"]
    for topic in topics:
        logging.info(f"Start evaluating {topic} topic")
        ans_file = open(f"./data/results/naive_{topic}.jsonl", "a")
        ds = load_dataset("qinchuanhui/UDA-QA", topic, split="test")
        count = 0
        for item in ds:
            response, docs = ask(item['question'])
            item['response'] = response
            item['docs'] = docs
            ans_file.write(json.dumps(item) + "\n")
            count += 1
            if count % 100 == 0:
                logging.info(f"Done {count} QA.")
            ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    # print(ask("What is the interest expense in 2009?"))
    eval()