from transformers import AutoTokenizer, AutoModelForCausalLM
from construct.vectordb import VectorDB
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import logging
import json
from tqdm.auto import tqdm
from datasets import load_dataset
from config import ALL_QUESTIONS_ANSWERS

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

device = "cuda" if torch.cuda.is_available() else "cpu"
model_embed = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto", "torch_dtype": torch.float16},
    tokenizer_kwargs={"padding_side": "left"},
)

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

vectordb = VectorDB()


def gpt_evaluate_response(correct_answer, context):
    prompt = f"""
    Context:
    {context}

    Correct Answer:
    {correct_answer}

    Task:
    Determine whether the context contains the information stated in the correct answer. \\
    Respond with "1" if yes, and "0" if no. Do not provide any explanation, just the number.
    """
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

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

    return int(content)


def eval():
    for num, qa in enumerate(ALL_QUESTIONS_ANSWERS):
        results = []
        correct = 0
        for q in qa:
            embedded = model_embed.encode(q['answer'], convert_to_tensor=True)
            retrieved = vectordb.get_topk_similar(embedded, k=1)
            context = ""
            for i in range(len(retrieved)):
                context += f"{retrieved[i][0]} "
            score = gpt_evaluate_response(q['answer'], context)
            results.append({
                "correct_answer": q['answer'],
                "retrieved_context": context,
                "evaluation": score
            })
            correct += score
        
        accuracy = correct / len(qa)
        results.append({"accuracy": f"{accuracy * 100:.2f}%"})
        
        output_file = f"./data/naive_rag/{num+1}_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
            

if __name__ == "__main__":
    eval()