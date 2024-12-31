
import yaml
import shutil
import os
import re
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from config import config

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder, SentenceTransformer
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from FlagEmbedding import FlagReranker

import torch

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

class PDFProcessor:
    """Handles PDF loading and text splitting operations"""
    
    def __init__(self, method: str = 'topic', chunk_size: int = 1000, chunk_overlap: int = 200):
        self.method = method
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_by_topics(self, text: str, metadata: dict) -> List[Document]:
        """Split text based on timestamps and topics, then further split by chunk size"""
        pattern = r'(\d{2}:\d{2})\s+(.+?)\n\n([\s\S]+?)(?=\n\n\d{2}:\d{2}|\Z)'
        chunks = []

        for match in re.finditer(pattern, text, re.MULTILINE):
            timestamp, topic, content = match.groups()
            if content.strip():
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'timestamp': timestamp,
                    'topic': topic
                })
                
                # Create a temporary Document for the entire topic content
                temp_doc = Document(page_content=content.strip(), metadata=chunk_metadata)
                
                # Split the topic content if it exceeds chunk_size
                if len(content) > self.chunk_size:
                    sub_chunks = self.text_splitter.split_documents([temp_doc])
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(temp_doc)

        return chunks

    def load_documents(self, document_directory: str) -> List[Document]:
        """
        Load multiple PDFs and Word documents from a directory
        """
        documents = []
        doc_files = [file for file in os.listdir(document_directory) if file.endswith(('.pdf', '.docx', '.doc'))]
        
        for file in tqdm(doc_files, desc="Loading Documents"):
            file_path = os.path.join(document_directory, file)
            try:
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                elif file.endswith('.docx'):
                    loader = Docx2txtLoader(file_path)
                elif file.endswith('.doc'):
                    loader = UnstructuredWordDocumentLoader(file_path)
                
                docs = loader.load()
                
                # Apply the selected chunking method
                for doc in docs:
                    doc.metadata['source_file'] = file
                    if self.method == 'topic':
                        split_docs = self.split_by_topics(doc.page_content, doc.metadata)
                    elif self.method == 'sliding_window':
                        split_docs = self.text_splitter.split_documents([doc])
                    else:
                        raise ValueError(f"Unknown chunking method: {self.method}")
                    documents.extend(split_docs)
                
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        
        print(f"Loaded {len(documents)} document chunks from {len(doc_files)} documents")
        return documents

class VectorStoreBuilder:
    """Handles creation and management of vector stores"""
    
    def __init__(self, model_name: str =  config['embedding_model']):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"device": self.device, "batch_size": 32}
        )
        # self.embedding_model = SentenceTransformer(model_name).cuda()
    
    def build_and_save_stores(self, documents: List[Document], save_dir: str) -> Tuple[FAISS, Chroma]:
        """
        Build FAISS and Chroma vector stores and save them to disk,
        removing existing content if it exists
        """
        os.makedirs(save_dir, exist_ok=True)
        
        faiss_path = os.path.join(save_dir, "faiss_index")
        chroma_path = os.path.join(save_dir, "chroma_store")

        # Remove existing FAISS index if it exists
        if os.path.exists(faiss_path):
            shutil.rmtree(faiss_path)
            print(f"Removed existing FAISS index at {faiss_path}")

        # Remove existing Chroma store if it exists
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            print(f"Removed existing Chroma store at {chroma_path}")

        # Create and save FAISS store
        faiss_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )
        faiss_store.save_local(faiss_path)
        
        # Create Chroma store (it will be automatically persisted)
        chroma_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=chroma_path
        )

        return faiss_store, chroma_store

    def load_stores(self, save_dir: str, allow_dangerous_deserialization: bool = True) -> Tuple[FAISS, Chroma]:
        """Load vector stores from disk"""
        # Load FAISS store
        faiss_store = FAISS.load_local(
            os.path.join(save_dir, "faiss_index"), 
            self.embedding_model, 
            allow_dangerous_deserialization=allow_dangerous_deserialization
        )
        
        # Load Chroma store
        chroma_store = Chroma(
            persist_directory=os.path.join(save_dir, "chroma_store"), 
            embedding_function=self.embedding_model
        )

        return faiss_store, chroma_store
        
class HybridSearcher:
    """Implements hybrid search functionality"""
    
    def __init__(self, faiss_store: FAISS, chroma_store: Chroma):

        # Initialize retrievers
        self.faiss_retriever = faiss_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config['raw_rank_k']}
        )
        
        self.chroma_retriever = chroma_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k":  config['raw_rank_k']}
        )
        self.chroma_docs = self._get_chroma_docs(chroma_store)

        self.bm25_retriever = BM25Retriever.from_documents(self.chroma_docs)
        self.bm25_retriever.k = config['raw_rank_k']
        
        # Create ensemble retriever
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[
                self.bm25_retriever,
                self.faiss_retriever,
                self.chroma_retriever
            ],
            weights=config['retriever_weights']
        )

    def _get_chroma_docs(self, chroma_store: Chroma) -> List[Document]:
            """Retrieve all documents from ChromaDB and convert to Document objects"""
            raw_docs = chroma_store.get()
            documents = []
            for i, doc in enumerate(raw_docs['documents']):
                metadata = {}
                if 'metadatas' in raw_docs and len(raw_docs['metadatas']) > 0:
                    metadata = raw_docs['metadatas'][i]
                documents.append(Document(page_content=doc, metadata=metadata))
            return documents
            

    def rerank_with_cross_encoder(self, query, initial_results):

        # Prepare input pairs
        pairs = [[query, doc.page_content] for doc in initial_results]

        # Load the cross-encoder model and get scores
        # model = CrossEncoder(config['reranking_model'], device='cuda')
        # scores = model.predict(pairs)
        if config['embedding_model'] == "all-MiniLM-L12-v2":
            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device='cuda')
            scores = model.predict(pairs)
        elif config['embedding_model'] == "BAAI/bge-large-en-v1.5":
            # Setting use_fp16 to True speeds up computation with a slight performance degradation
            model = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) # Automatically operates on GPU
            scores = model.compute_score(pairs)
            
        # Combine scores with documents and sort
        scored_results = list(zip(scores, initial_results))
        reranked_results = sorted(scored_results, key=lambda x: x[0], reverse=True)
        
        # Return top k results
        return [doc for _, doc in reranked_results[:config['rerank_k']]]

    def search(self, query: str) -> List[Document]:
        """
        Perform hybrid search
        """
        results = self.hybrid_retriever.get_relevant_documents(query)
        # Return only the top k results
        return self.rerank_with_cross_encoder(query, results)
        
    def analyze_results(self, query: str) -> Dict[str, Any]:
        """
        Analyze search results
        """
        results = self.search(query)
        analysis = {
            "query": query,
            "num_results": len(results),
            "sources": set(),
            "avg_chunk_length": 0,
        }
        
        total_length = 0
        for doc in results:
            analysis["sources"].add(doc.metadata.get("source", "Unknown"))
            total_length += len(doc.page_content)
        
        analysis["avg_chunk_length"] = total_length / len(results)
        analysis["unique_sources"] = len(analysis["sources"])

        print("\nAnalysis:")
        print(f"Query: {analysis['query']}")
        print(f"Number of results: {analysis['num_results']}")
        print(f"Unique sources: {analysis['unique_sources']}")
        print(f"Average chunk length: {analysis['avg_chunk_length']:.2f}")

        return analysis


if __name__ == "__main__":

    # Initialize PDF Processor
    pdf_processor = PDFProcessor(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap']
    )

    # Load documents
    docs = pdf_processor.load_documents(config['document_path'])

    # Initialize and use Vector Store Builder
    builder = VectorStoreBuilder()
    builder.build_and_save_stores(docs, config['save_directory'])

    print(f"Vector stores built and saved to {config['save_directory']}")