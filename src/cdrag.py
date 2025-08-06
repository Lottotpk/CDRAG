import json
import networkx as nx
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import os
import traceback
from config import ALL_QUESTIONS_ANSWERS

model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

def load_graph_from_json(file_path):
    with open(file_path, "r",encoding='utf-8') as f:
        data = f.read()
    data = json.loads(data)
    G = nx.DiGraph()

    for entity in data["entities"]:
        node_attrs = {
            "type": entity.get("type", ""),
            "description": entity.get("description", ""),
            "attributes": entity.get("attributes", {})
        }
        G.add_node(entity["name"], **node_attrs)

    # 添加关系
    for relation in data["relations"]:
        source, rel, target,rel_description = relation
        # G.add_edge(source, target, relation=rel)
        G.add_edge(source, target, relation=rel, rel_description=rel_description)

    return G


def generate_embeddings(graph, model):
    node_embeddings = {}
    for node in graph.nodes:
        node_data = graph.nodes[node]
        text_parts = [
            f"{node}",
            f"{node_data['type']}",
        ]
        full_text = " ".join([part for part in text_parts if part])
        node_embeddings[node] = model.encode(full_text).tolist()

    relation_embeddings = {
        rel: model.encode(rel).tolist()
        for rel in set(edge[2]["relation"] for edge in graph.edges(data=True))
    }
    
    return node_embeddings, relation_embeddings


def retrieve_relevant_nodes(query, node_embeddings, model, communities, k=8):
    # Global search: look like vector database (similarity search)
    query_embedding = model.encode(query).reshape(1, -1)
    similarities = [(node, cosine_similarity(query_embedding, np.array(embed).reshape(1, -1))[0][0])
                    for node, embed in node_embeddings.items()]
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    similarities = similarities[:k]

    # Local search: retrieve relevant node in the community for more fine-grained information for the most similar
    for group in communities:
        if similarities[0][0] in group:
            for item in group:
                similarities.append((item, None))
    return similarities


def retrieve_context(node, graph, depth=2):
    context = set()
    
    def get_node_info(node):
        data = graph.nodes[node]
        info = [
            f"name: {node}",
            f"type: {data.get('type', '')}",
            f"description: {data.get('description', 'u')}"
        ]
        return "\n".join(info)
    
    def explore_neighbors(current_node, current_depth):
        if current_depth > depth:
            return
        context.add(get_node_info(current_node))
        
        for neighbor in graph.neighbors(current_node):
            # rel = graph[current_node][neighbor]["relation"]
            # rel_descripntion = graph[current_node][neighbor]["rel_description"]
            # context.add(f"relation: {current_node} --[{rel}]-> {neighbor}--description:{rel_descripntion}--")
            # context.add(f"{rel_descripntion}")
            explore_neighbors(neighbor, current_depth + 1)
    explore_neighbors(node, 1)
    return list(context)


# Use LLM to evaluate if the correct answer is in the context
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


# Evaluate accuracy
def evaluate_accuracy(questions_answers, node_embeddings, model, graph, output_file, communities):
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    correct = 0
    results = []

    for qa in questions_answers:
        correct_answer = qa["answer"]
        print(f"\nEvaluating answer: {correct_answer}")
        top_nodes = retrieve_relevant_nodes(correct_answer, node_embeddings, model, communities)
        print(f"Top nodes: {top_nodes}")
        context = set()
        for node, _ in top_nodes:
            node_context = retrieve_context(node, graph)
            print(f"Context for node {node}: {node_context}")
            context.update(node_context)
        context_text = " ".join(context)
        print(f"Combined context: '{context_text}'\n---")

        evaluation = gpt_evaluate_response(correct_answer, context_text)
        results.append({
            "correct_answer": correct_answer,
            "retrieved_context": context_text,
            "evaluation": evaluation
        })
        correct += evaluation

    accuracy = correct / len(questions_answers)
    results.append({"accuracy": f"{accuracy * 100:.2f}%"})

    # Save results to file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


# Main function to process multiple files
def main():

    # Initialize embedding model
    embedding_model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto", "torch_dtype": torch.float16},
        tokenizer_kwargs={"padding_side": "left"},
    )
    
    for i in range(1, 106):
        json_file = f"data/CDRAG/{i}.json"
        questions_answers = ALL_QUESTIONS_ANSWERS[i-1]
        output_file = json_file.replace(".json", "_results.json")
        G = load_graph_from_json(json_file)
        node_embeddings, _ = generate_embeddings(G, embedding_model)
        communities = list(nx.community.louvain_communities(G))
        print(f"Processing file: {json_file}")
        evaluate_accuracy(questions_answers, node_embeddings, embedding_model, G, output_file, communities)

if __name__ == "__main__":
    main()