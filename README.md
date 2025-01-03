# Retrieval-Augmented Generation (RAG) for PodCast Q&A
## 1. Background

Andrew Huberman, Ph.D., is a neuroscientist and tenured professor at Stanford.
In 2021, He launched the [Huberman Lab podcast](https://www.hubermanlab.com/about). The podcast is frequently ranked in the top 10 of all podcasts globally and is often ranked #1 in the categories of Science, Education, and Health & Fitness.

**The goal of the project** was to retrieve relevant information across different  episodes and generate response based on user's query. The response will be  largely based on the context provided by Andrew Huberman in his conversations. The project is mostly written with [**LangChain**](https://www.langchain.com/).

Current experiment shows prompt engineering, choice of chunk size/overlap, reranker are more effective compared with model size.
## 2. Model Details
| Model             | Param Size    | Function |
| ----------------- | ------------  | ------- |
|  [Llama 3.2](https://ollama.com/library/llama3.2)       | 3B             |Generator      
|  [Llama 3-Instruct](https://ollama.com/library/llama3)   | 8B           |Generator    
|  [Bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)  | 560M    |Embedding  
|  [Bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) |335M     |Rerank  
|  [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)       | 33.4M   |Embedding 
| [ms-marco-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2)  | 33.4M   |Rerank    


## 3. Work flow
1. Process input documents and split into desired size of chunks.
2. Create vector embeddings from the chunks using a sentense transformer and store them in a [Chroma database](https://github.com/chroma-core/chroma). I used a [BGE model](https://github.com/FlagOpen/FlagEmbedding/tree/master) and a smaller [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2). The BGE and L12 model convert text into  1024 and 384 dimension embeddings, respectively. 
3. Perform similarity search and retrieve relevant chunks from the database. I used a hybrid search approch, which is an ensemble retriever combining [BM25](https://python.langchain.com/docs/integrations/retrievers/bm25/) and semantic search.
4. Rerank the retrieved chunks with cross encoder ([BGE reranker](https://huggingface.co/BAAI/bge-reranker-large) and [ms-marco-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2)).
5. Using the reranked chunks as context, [Llama 3.2](https://ollama.com/library/llama3.2) (deployed with [Ollama](https://github.com/ollama/ollama)) will generate the response.
6. Call `gpt-4o` to compare AI answer with reference using [OpenAI API](https://openai.com/index/openai-api/), and evalute the AI response on 'Relevance', 'Accuracy', 'Completeness', 'Conciseness', in a scale 1-5. recall@K is also measured.
- **On going experimentation: query augmentation, finetune Llama on domain specific QA datasets, DPO...**


## 4. Advance RAG features
### Query Augmentation

**Rewrite**
```
Original Query: what if I'm anxious?
Expanded Query: Here are 4 related ideas or questions:
                1. Physiological Response to Anxiety: How does anxiety affect the body's physiological responses, such as heart rate, blood pressure, and respiration? Are there any specific biomarkers or physiological changes that can be measured to assess the level of anxiety?
                2. Neurotransmitters and Anxiety: What role do neurotransmitters like serotonin, dopamine, and GABA play in regulating mood and emotions, particularly during anxious states? Can certain dietary supplements, medications, or therapies targeting these neurotransmitters help alleviate anxiety symptoms?
                3. Social and Environmental Triggers for Anxiety: What are the common social and environmental triggers that can exacerbate or induce anxiety? How do factors like social media, news, or personal expectations influence an individual's anxiety levels, and what strategies can be employed to mitigate these triggers?
                4. Coping Mechanisms for Managing Anxiety: What evidence-based coping mechanisms and stress-reduction techniques are effective in managing anxiety? Can mindfulness practices
Rewritten Query: What are the physiological, neurological, social, and behavioral factors that contribute to anxiety, and what effective coping mechanisms and interventions can be employed to manage anxiety symptoms? Physiological responses to anxiety Neurotransmitters and anxiety management. Social and environmental triggers for anxiety. Coping mechanisms and stress-reduction techniques
```
**Hyde (Hypothetical Document Embedding)**
```
Original Query: what if I'm anxious?
Hallucination: If you are anxious, recognize that anxiety is a normal and adaptive response to perceived threats. It serves as an alert system to help you respond to danger. However, excessive or persistent anxiety can impair daily functioning. Key strategies for managing anxiety include:Mindfulness practices to regulate emotions. Cognitive-behavioral therapy (CBT) techniques to reframe negative thoughts. Physical activity to reduce stress and improve mood. Grounding techniques to focus on the present moment. It is also essential to identify personal triggers and develop coping mechanisms to manage overwhelming feelings.
```
### Retrieval Augmentation
* **Small to big indexing (Recursive retrieval)**: Starting with smaller chunks and recursively expanding to parent chunk.

* **Chunking by passage structure**:
Based on information structure, ensuring coherence and semantic integrety of chunks. 

### Multi-stage LLM Finetuing 
The PodCast covers a wide range of topics, including Psychology, Neuroscience, Biology, Behavioral Science, Health & Fitness... 
* General QA
* Domain Specific QA
* Direct Preference Optimization (DPO)
## 5. Usage (See demo notebook)

* Install dependencies
```
pip install -r requirements.txt
```
* Create the Chroma Database
```
python buildDB.py
```
* Query the Chroma Database
```
python query.py "[Insert your question here]"
```
* Evaluate the system
```
python evaluate.py
``` 

## 6. Instructions
#### Docs
* Add '.pdf','.doc', or '.docx' files to folder and set `document_path` in `config.yaml`. The demo has 27 transcripts for the PodCast.
* Set `save_directory` in `config.yaml` to specify the path to create FAISS and Chorma database. 
#### Evaluation dataset
* Set `dataset_path` and `results_file` in `config.yaml`. `dataset_path` is the path to a Q&A dataset used for evalutaion, and `results_file` is the path to store evaluation  resutls, including score and reasoning. Both are `json` files.
#### Other params
* `chunk_size`: The individual chunk size used to split docs.
* `chunk_overlap`: Overlapping size for splitting.
* `raw_rank_k`: Number of retrieved chunks.
* `rerank_k`: Top k chunks to return after reranking
* `retriever_weights`: The system uses a hybrid search approch, this param specifies weight of  BM25, FAISS, Chorma retrievers
* `embedding_model`: The model choice for sentence transformer, i.e. the embedding model. Specifying this param will determine  reranker model accordingly. The model used in this project are `all-MiniLM-L12-v2` with `ms-marco-MiniLM-L-12-v` and `BAAI/bge-large-en-v1.5` with `BAAI/bge-reranker-large`
* `augmentation`: Whether to use query augmentaion. 
#### Some notes
* An OpenAI key is needed to run `evaluate.py`, since the project uses `gpt-4o` as the final evalutaion model.
Set `openai_api_key` variable  in  `config.yaml`. 

* The QA dataset was generated with GPT4o. The prompt can be found in `evaluation_QAdataset/prompt_template_QA_dataset.txt`. It is designed to ensure diversity and varied difficulty. The demo has 17 questions from first 3 documents (the full dataset has more entries).
