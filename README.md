# CDRAG
Cross-Document Retrieval-Augmented Generation (CDRAG): An RAG framework for enhancing cross-document reasoning of Large Language Models (LLMs). The poster of this study can be found [here](https://github.com/Lottotpk/CDRAG/blob/main/fig/Poster.pdf).

## CDRAG Pipelines
### Graph Construction
The knowledge graph architecture is very similar to the one in [this](https://github.com/LMMApplication/RAKG) repository, where the LLM will process the text in chunk and construct knowledge graph with appropriate entities-relationships. The text will be separated into multiple chunks and passed into LLM to construct the graph, along with Named Entities Recoginition (NER) mechanism to enrich the representation of the graph. Then, the new created graph will merge with the existing graph. The similar entities may be merged into a single entity, which will be determined the cosine similarity threshold (like the similarity in vector database). Below is the architecture of the graph construction pipeline.
![Graph construct](https://github.com/Lottotpk/CDRAG/blob/main/fig/Construct_white.png)

### Information Retrieval
The information retrieval pipeline is derived from the [GraphRAG](https://github.com/microsoft/graphrag) idea, which combines the global and local search. The gloobal search will search through all of the relevant nodes and find the top-k most relevant nodes. After that, the local search will continue to search the nodes belong to the communities of the top-k retrieval. This procedure will give make the retrieval steps have more fine-grained and detailed information of LLMs to process and generate the answer. Below is the architecture of the information retrieval pipeline.
![Information retrieval](https://github.com/Lottotpk/CDRAG/blob/main/fig/Retreive_white.png)

## Results
- The Measure of Information in Nodes and Edges (MINE) dataset contains 105 articles, ~1000 words each, which covers multiple fields: art, science, history, ethics, and psychology.
- The relevancy of the retrieval in each RAG method will be measured through this dataset. Three RAG methods are tested.

We use the three RAG frameworks for comparison: the traditional RAG, Knowledge Graph Generation ([KGGen](https://github.com/stair-lab/kg-gen)), and our framework CDRAG. The results are shown below.
![Results](https://github.com/Lottotpk/CDRAG/blob/main/fig/Results.png)
- The tradition RAG struggles to retrieve the relevant context from multiple documents with only 83.05%, almost 10% lower than KGGen.
- KGGen and CDRAG are both graph‐based method, which also have similar accuracy, with CDRAG slightly ahead by 2.60% due to richer details construction.
- CDRAG also perform more consistent than KGGen, which no domains have under 60% while KGGen has some topics around 50% accuracy.

## Files in this repository
There are may files contained in this repository. The most important files are all the `src` directory where all the source codes stay there.
```
.
├── data
│   ├── CDRAG
│   ├── KGs
│   ├── MINE.json
│   └── naive_rag
├── LICENSE
├── requirements.txt
├── src
│   ├── cdrag.py
│   ├── config.py
│   ├── construct
│   │   ├── rakg_processing.py
│   │   ├── RAKG.py
│   │   ├── storing_data.py
│   │   └── vectordb.py
│   ├── eval
│   │   ├── eval.py
│   ├── graphrag_result_store.py
│   ├── kgAgent.py
│   ├── kggen.py
│   ├── llm_provider.py
│   ├── naive.py
│   ├── pdfProcess.py
│   ├── prompt.py
│   └── textPrcess.py
```
- `data`: Store all raw and processed text chunk. The files usually stored in `.json` file.
- `requirements.txt`: The library dependencies for executing the provided code.
- `src`: The source code of all tested RAG methods.
- `src/construct`: The python scripts to process the raw text into vector database (traditional RAG) or knowledge graph (KGGen and CDRAG).
- `src/eval`: The evalutation script to process the retrieval accuracy of each method.
- `src/naive.py`: The retrieval python script for traditional RAG method.
- `src/construct/storing_data.py` and `src/construct/vectordb.py`: The vector database initialization and text processing for traditional RAG and The vector database (`vectordb` class) respectively.
- `src/kggen.py`: The graph construction and retrieval python script for KGGen method.
- `src/rakg_processing`: The data cleaner for JSON format from [RAKG](https://github.com/LMMApplication/RAKG) method. RAKG is our graph construction baseline. Moreover, this script also converts the JSON constructed graph in parquet file, ready to construct the graph into [GraphRAG](https://github.com/microsoft/graphrag) architecture (not used in this project).
- The rest of `src` files: The python scripts to construct the graph, enhancing the content with NER, which is drawn from [RAKG](https://github.com/LMMApplication/RAKG).

## Run this repository
1. Download python version 3.12, or create conda environment.
```
conda create -n <your environment name> python=3.12
```

2. Install the dependencies.
```
pip install -r requirements.txt
```
Note that there are some dependency conflicts that use different version of openai. You can ignore this if the openai library is not used.

3. Set up all of the configs in `src/config.py`. Please see [RAKG repository](https://github.com/LMMApplication/RAKG) for more info.

4. To reproduce the traditional RAG results, execute this command 
```
python src/construct/storing_data.py
```
to create a vector database. Then, run 
```
python src/naive.py
```
to perform a test on the dataset.

5. To reproduce the KGGen results, execute this command 
```
python src/kggen.py
```
to create a knowledge graph and test on the dataset simultaneously. Note that you may want to change the base LLM model and embedding model if the code does not work.

6. To reproduce CDRAG results, execute this command 
```
python src/construct/RAKG.py
```
to construct a graph, or directly use the constructed graph in `data/CDRAG`. Then, run 
```
python src/cdrag.py
```
to perform information retrieval and perform benchmarking on the dataset.
