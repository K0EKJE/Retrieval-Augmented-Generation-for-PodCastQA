import argparse

from config import config
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

from buildDB import VectorStoreBuilder, HybridSearcher
import buildDB

import warnings
warnings.filterwarnings("ignore")

def augment_query_hyde(original_query, chat_model):
    """Augment the original query by hucillation."""
    
    # Prompt for generating a hypothetical answer
    hyde_template = PromptTemplate(
        input_variables=["query"],
        template="""Given the question: "{query}"
        Please provide a hypothetical, brief, but plausible answer to this question. 
        The answer should be concise and focus on key points, as if it were a snippet from a relevant document.
        Do not preface the answer or use phrases like 'A hypothetical answer could be...'. 
        Simply provide the answer directly.

        Hypothetical answer:"""
    )
    print('hyde used')
    # Generate the hypothetical answer
    augmented_query = chat_model.predict(hyde_template.format(query=original_query), num_predict = 300,temperature = 1)
    
    return augmented_query

def augment_query_rewrite(original_query, chat_model):
    """Augment the original query by rewriting it using a language model."""
    
    # Create a prompt template for query expansion
    expansion_template = PromptTemplate(
        input_variables=["original_query"],
        template="""As an AI language model, I want you to expand on the following query 
        by imagining related technical concepts. 
        Don't answer the query, instead, generate a list of 3-5 related ideas or questions.

        Original query: {original_query}

        Related ideas and questions:
        1."""
    )

    # Generate expanded ideas
    expansion_result = chat_model.predict(expansion_template.format(original_query=original_query), num_predict = 200,temperature = 1)

    # Create a prompt template for refining the query
    refinement_template = PromptTemplate(
        input_variables=["original_query", "expansion_result"],
        template="""Create a single, concise, enhanced query based on this information:

        Original query: {original_query}

        Related ideas: {expansion_result}

        Remember to generate the main body query only without any further explaination. 

        Enhanced query:"""
    )

    # Generate the refined query
    refined_query = chat_model.predict(refinement_template.format(
        original_query=original_query,
        expansion_result=expansion_result,
        num_predict = 50
    ),temperature = 1)

    return refined_query

def answer_with_rag(faiss_store, chroma_store, query = '', augmentation = 'None'):
    """Generate answer based on query"""
    chat_model = ChatOllama(model="llama3.2")
    
    if augmentation ==  'hyde': query = augment_query_hyde(query, chat_model)
    elif augmentation == 'rewrite': query = augment_query_rewrite(query, chat_model)

    # Initialize searcher
    Searcher = HybridSearcher(faiss_store, chroma_store)

    results = Searcher.search(query)

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