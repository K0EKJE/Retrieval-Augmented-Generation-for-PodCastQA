import os
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from config import config
from buildDB import VectorStoreBuilder, HybridSearcher

from query import answer_with_rag
import query
# Function to load your dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def evaluate_qa_pair(chat_model, qa_pair):
    
    question = qa_pair['Question']
    true_answer = qa_pair['Answer']
    results, ai_answer = answer_with_rag(faiss_store, chroma_store, question, augmentation = config['augmentation'])

    include_source = False
    for doc in results:
        # measure recall @ k
        if doc.metadata['source_file']  == qa_pair['SourceFile']:
            include_source = True
            break


    system_message = SystemMessage(content="""
    You are an impartial judge evaluating the quality of an AI assistant's answer to a user's question. 
    You will be given the question, the true answer, and the AI assistant's answer.
    Your task is to rate the AI assistant's answer on a scale from 1 to 5 (where 1 is the worst and 5 is the best) for each of the following criteria:
    - Relevance: How relevant is the answer to the question?
    - Accuracy: How factually correct is the answer compared to the true answer?
    - Completeness: How complete is the answer?
    - Conciseness: How concise and to-the-point is the answer?
    After rating, provide a brief explanation for your ratings.
    Always provide a numerical rating between 1 and 5 for each criterion. Do not use N/A or any non-numeric values.
    """)

    human_message = HumanMessage(content=f"""
    Question: {question}
    True Answer: {true_answer}
    AI Assistant's Answer: {ai_answer}

    Please provide your ratings and explanation in the following format:

    Ratings (1-5):
    Relevance: 
    Accuracy: 
    Completeness: 
    Conciseness: 

    Explanation:
    """)

    response = chat_model([system_message, human_message])
    
    # Parse the evaluation result
    lines = response.content.split('\n')
    ratings = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':')
            if key.strip() in ['Relevance', 'Accuracy', 'Completeness', 'Conciseness'] and len(line)<20:
                try:
                    ratings[key.strip()] = int(value.strip())
                except ValueError:
                    print(f"Warning: Invalid rating value for {key.strip()}: {value.strip()}. Using default value of 1.")
                    ratings[key.strip()] = 1

    explanation_start = next((i for i, line in enumerate(lines) if line.strip().startswith('Explanation:')), -1)
    explanation = '\n'.join(lines[explanation_start+1:]) if explanation_start != -1 else "No explanation provided."

    return include_source, {
        'question': question,
        'true_answer': true_answer,
        'ai_answer': ai_answer,
        'ratings': ratings,
        'explanation': explanation.strip(),
        'metadata': {
            'SourceFile': qa_pair.get('SourceFile', 'Unknown'),
            'Difficulty': qa_pair.get('Difficulty', 'Unknown'),
            'Category': qa_pair.get('Category', 'Unknown')
        }
    }

def evaluate_rag_system(dataset_path):
    dataset = load_dataset(dataset_path)
    chat_model = ChatOpenAI(model_name=config['evaluation_model'], temperature=0)
    # chat_model = ChatOllama(model="llama3.2")   
    results = []
    correct = 0
    for qa_pair in tqdm(dataset, desc="Evaluating QA pairs"):
        include_source, result = evaluate_qa_pair(chat_model, qa_pair)
        correct += include_source
        results.append(result)
    recall_at_k = correct/len(dataset)

    # Calculate overall scores
    overall_scores = {}
    for metric in ['Relevance', 'Accuracy', 'Completeness', 'Conciseness']:
        valid_scores = [r['ratings'][metric] for r in results if metric in r['ratings']]
        if valid_scores:
            overall_scores[metric] = sum(valid_scores) / len(valid_scores)
        else:
            overall_scores[metric] = 0
            print(f"Warning: No valid scores for {metric}")

    overall_score = sum(overall_scores.values()) if overall_scores else 0

    return  recall_at_k, results, overall_scores, overall_score


if __name__ == "__main__":
    # Load the stores
    faiss_store, chroma_store = VectorStoreBuilder().load_stores(config["save_directory"])
    print("DataDB loaded successfully")

    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
        
    # Specify Path
    dataset_path = config["dataset_path"]
    results_file = config["results_file"]

    recall_at_k, results, overall_scores, overall_score = evaluate_rag_system(dataset_path)
    # Print results
    print(f"Overall System Score: 【{overall_score:.2f}】")
    print(f"Recall@K: {recall_at_k:.2f}")
    print("\nScores by metric:")
    for metric, score in overall_scores.items():
        print(f"{metric}: {score:.2f}")

    # Check if the file exists
    if os.path.exists(results_file):
        # If it exists, clear its contents
        open(results_file, 'w').close()
        print(f"\nCleared existing {results_file}")
    else:
        print(f"\nCreating new file: {results_file}")

    # Write the new results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Detailed results saved to {results_file}")
    