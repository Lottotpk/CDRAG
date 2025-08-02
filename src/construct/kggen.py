import json
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import torch
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from config import ALL_QUESTIONS_ANSWERS

from kg_gen import KGGen

# Load JSON data
def load_graph_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    # Initialize a directed graph
    G = nx.DiGraph()

    # Add entities (nodes)
    for entity in data["entities"]:
        G.add_node(entity)

    # Add relationships (edges)
    for relation in data["relations"]:
        source, rel, target = relation
        G.add_edge(source, target, relation=rel)

    return G

# Generate embeddings
def generate_embeddings(graph, model):
    node_embeddings = {node: model.encode(node).tolist() for node in graph.nodes}
    relation_embeddings = {rel: model.encode(rel).tolist()
                           for rel in set(edge[2]["relation"] for edge in graph.edges(data=True))}
    return node_embeddings, relation_embeddings

# Retrieve top-k relevant nodes
def retrieve_relevant_nodes(query, node_embeddings, model, k=8):
    query_embedding = model.encode(query).reshape(1, -1)
    similarities = [(node, cosine_similarity(query_embedding, np.array(embed).reshape(1, -1))[0][0])
                    for node, embed in node_embeddings.items()]
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:k]

# Retrieve context from relationships
def retrieve_context(node, graph, depth=2):
    context = set()
    def explore_neighbors(current_node, current_depth):
        if current_depth > depth:
            return
        # Outgoing edges
        for neighbor in graph.neighbors(current_node):
            rel = graph[current_node][neighbor]["relation"]
            context.add(f"{current_node} {rel} {neighbor}.")
            explore_neighbors(neighbor, current_depth + 1)
        # Incoming edges
        for neighbor in graph.predecessors(current_node):
            rel = graph[neighbor][current_node]["relation"]
            context.add(f"{neighbor} {rel} {current_node}.")
            explore_neighbors(neighbor, current_depth + 1)
    explore_neighbors(node, 1)
    return list(context)

# Use GPT to evaluate if the correct answer is in the context
def gpt_evaluate_response(correct_answer, context):
    pass
    # prompt = f"""
    # Context:
    # {context}

    # Correct Answer:
    # {correct_answer}

    # Task:
    # Determine whether the context contains the information stated in the correct answer. \\
    # Respond with "1" if yes, and "0" if no. Do not provide any explanation, just the number.
    # """
    # response = client.chat.completions.create(
    #     model="gpt-4",
    #     messages=[{"role": "system", "content": "You are an evaluator that checks if the correct answer can be deduced from the information in the context."},
    #               {"role": "user", "content": prompt}],
    #     max_tokens=1,
    #     temperature=0.0
    # )
    # return int(response.choices[0].message.content.strip())

# Evaluate accuracy
def evaluate_accuracy(questions_answers, node_embeddings, model, graph, output_file):
    print(f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    correct = 0
    results = []

    for qa in questions_answers:
        correct_answer = qa["answer"]
        print(f"\nEvaluating answer: {correct_answer}")
        top_nodes = retrieve_relevant_nodes(correct_answer, node_embeddings, model)
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
    json_files = [f"./data/KGs/{i}.json" for i in range(1, 107)] 

    # Initialize embedding model
    embedding_model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto", "torch_dtype": torch.float16},
        tokenizer_kwargs={"padding_side": "left"},
    )

    valid_pairs = [
        (json_file, qa) 
        for json_file, qa in zip(json_files, ALL_QUESTIONS_ANSWERS) 
        if os.path.exists(json_file)
    ]

    for json_file, questions_answers in valid_pairs:
        output_file = json_file.replace(".json", "_results.json")
        print(f"Processing file: {json_file}")
        try:
            G = load_graph_from_json(json_file)
            node_embeddings, _ = generate_embeddings(G, embedding_model)
            evaluate_accuracy(questions_answers, node_embeddings, embedding_model, G, output_file)
        except Exception as e:
            print(f"Error processing file {json_file}: {str(e)}, skipping...")
            continue


def construct_graph():
    kg = KGGen(
        model="huggingface/sambanova/Qwen/Qwen2.5-72B-Instruct",
        temperature=0.0,
    )
    with open("./data/MINE.json", "r") as f:
        data = json.load(f)
    
    for i in range(len(data)):
        kg.generate(
            input_data=data[i]['content'],
            context=data[i]['topic'],
            chunk_size=3000,
            output_folder=f"./data/KGs/{i+1}.json"
        )


if __name__ == "__main__":
    construct_graph()
    # main()