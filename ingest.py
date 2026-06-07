import os
import re
import time
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Verify Google API Key is set
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is not set in the environment or .env file.")

def add_documents_with_retry(embeddings, documents, collection_name, persist_dir, batch_size=10, delay=3):
    """
    Initializes a Chroma vector store and adds documents in batches, with automatic
    retry and backing off if rate limits (429) or other errors occur.
    """
    print(f"Initializing collection '{collection_name}'...")
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    
    total = len(documents)
    print(f"Adding {total} documents to '{collection_name}' in batches of {batch_size}...")
    
    i = 0
    while i < total:
        batch = documents[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"  -> Uploading batch {batch_num}/{total_batches}...")
        
        try:
            db.add_documents(batch)
            i += batch_size
            # Small delay between successful batches
            if i < total:
                time.sleep(delay)
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                print("  [Warning] Rate limit (429 / Resource Exhausted) hit. Waiting 35 seconds for quota reset...")
                time.sleep(35)
            else:
                print(f"  [Warning] Transient error occurred: {err_msg}. Retrying in 10 seconds...")
                time.sleep(10)
                
    print(f"Collection '{collection_name}' populated successfully!\n")

def ingest_data():
    persist_dir = "./vector_store"
    
    # Clean up the previous vector store if it exists
    if os.path.exists(persist_dir):
        print(f"Clearing existing vector store directory at {persist_dir}...")
        shutil.rmtree(persist_dir)
        
    print("Initializing Google Generative AI Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # 1. Process university_info.txt
    print("Processing university_info.txt...")
    univ_path = "data/university_info.txt"
    if os.path.exists(univ_path):
        with open(univ_path, "r", encoding="utf-8") as f:
            univ_content = f.read()
        
        lines = [line.strip() for line in univ_content.split("\n") if line.strip()]
        
        # Group lines in chunks of 6
        univ_docs = []
        chunk_size_lines = 6
        for i in range(0, len(lines), chunk_size_lines):
            chunk_lines = lines[i:i+chunk_size_lines]
            text = "\n".join(chunk_lines)
            univ_docs.append(Document(
                page_content=text, 
                metadata={"source": "university_info", "chunk_id": i // chunk_size_lines}
            ))
        
        print(f"Loaded {len(univ_docs)} chunks from university_info.txt")
        add_documents_with_retry(embeddings, univ_docs, "university_info", persist_dir)
    else:
        print(f"Warning: {univ_path} not found.")

    # 2. Process B.Tech_CSE.txt
    print("Processing B.Tech_CSE.txt...")
    cse_path = "data/B.Tech_CSE.txt"
    if os.path.exists(cse_path):
        with open(cse_path, "r", encoding="utf-8") as f:
            cse_content = f.read()
        
        # Split by double newline to preserve section blocks
        sections = [sec.strip() for sec in cse_content.split("\n\n") if sec.strip()]
        cse_docs = []
        for i, sec in enumerate(sections):
            cse_docs.append(Document(
                page_content=sec,
                metadata={"source": "cse_info", "section_id": i}
            ))
            
        print(f"Loaded {len(cse_docs)} sections from B.Tech_CSE.txt")
        add_documents_with_retry(embeddings, cse_docs, "cse_info", persist_dir)
    else:
        print(f"Warning: {cse_path} not found.")

    # 3. Process student_feedback.txt
    print("Processing student_feedback.txt...")
    feedback_path = "data/student_feedback.txt"
    if os.path.exists(feedback_path):
        with open(feedback_path, "r", encoding="utf-8") as f:
            feedback_content = f.read()
        
        lines = [line.strip() for line in feedback_content.split("\n") if line.strip()]
        
        # Group feedback in chunks of 5 lines to preserve category-based feedback and reduce requests count
        feedback_docs = []
        chunk_size_feedback = 5
        for i in range(0, len(lines), chunk_size_feedback):
            chunk_lines = lines[i:i+chunk_size_feedback]
            text = "\n".join(chunk_lines)
            
            # Extract main category from the first line of the chunk
            category = "General"
            cat_match = re.search(r"Category:\s*([^|]+)", chunk_lines[0])
            if cat_match:
                category = cat_match.group(1).strip()
                
            feedback_docs.append(Document(
                page_content=text,
                metadata={"source": "student_feedback", "chunk_id": i // chunk_size_feedback, "primary_category": category}
            ))
            
        print(f"Loaded {len(feedback_docs)} grouped chunks from student_feedback.txt")
        add_documents_with_retry(embeddings, feedback_docs, "student_feedback", persist_dir)
    else:
        print(f"Warning: {feedback_path} not found.")

    print("Successfully ingested and indexed all data into Chroma DB collections!")

if __name__ == "__main__":
    ingest_data()
