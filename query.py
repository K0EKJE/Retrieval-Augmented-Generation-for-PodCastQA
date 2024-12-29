import argparse

from config import config
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

from buildDB import VectorStoreBuilder, HybridSearcher
import buildDB

import warnings
warnings.filterwarnings("ignore")

def answer_with_rag(faiss_store, chroma_store, query = ''):

    # Initialize searcher
    Searcher = HybridSearcher(faiss_store, chroma_store)

    results = Searcher.search(query)
    
    chat_model = ChatOllama(model="llama3.2")

    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""

    prompt_template ="""You are a leading expert in the field related to the following question. 
        Summarize the context to give a confident, 
        authoritative and concise answer. Avoid phrases like "according to" or "based on the document." 
        Instead, speak directly and assertively, as an expert would. 

        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Remember to keep your answer clear and concise.
        Context:
        {context}

        Question: {question}

        Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    context = "\n\n".join([doc.page_content for doc in results])

    answer = chat_model.predict(text=PROMPT.format_prompt(
        context=context,
        question=query
    ).text,
    max_tokens=1000,
    )

    return results, answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Question to ask the model")
    args = parser.parse_args()

    faiss_store, chroma_store = VectorStoreBuilder().load_stores(config['save_directory'])
    results, answer = answer_with_rag(faiss_store, chroma_store, args.query)
    print(f"Answer: {answer}")