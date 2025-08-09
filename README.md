# CDRAG
Cross-Document Retrieval-Augmented Generation (CDRAG): An RAG framework for enhancing cross-document reasoning of Large Language Models (LLMs). 

## CDRAG Pipelines
### Graph Construction
The knowledge graph architecture is very similar to the one in [this](https://github.com/LMMApplication/RAKG) repository, where the LLM will process the text in chunk and construct knowledge graph with appropriate entities-relationships. The text will be separated into multiple chunks and passed into LLM to construct the graph, along with Named Entities Recoginition (NER) mechanism to enrich the representation of the graph. Then, the new created graph will merge with the existing graph. The similar entities may be merged into a single entity, which will be determined the cosine similarity threshold (like the similarity in vector database). Below is the architecture of the graph construction pipeline.
![image](https://github.com/Lottotpk/CDRAG/blob/main/fig/Construct.png)

### Information Retrieval
