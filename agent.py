import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is not set in the environment or .env file.")

# Initialize embeddings (reused for retrieval)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Initialize models

classifier_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.0,
    model_kwargs={"response_mime_type": "application/json"}
)

response_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)
def get_clean_text(content) -> str:
    """
    Safely extracts the clean text string from the model response content.
    Handles standard string outputs as well as list-of-dict structures.
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if "text" in part:
                    parts.append(part["text"])
                elif "type" in part and part["type"] == "text" and "text" in part:
                    parts.append(part["text"])
        return "\n".join(parts)
    return str(content)

def check_relevance(query: str) -> dict:
    prompt = f"""
    You are an advanced query relevance classifier for the University of Calcutta Student Support Chatbot.
    Your job is to determine whether the user's query is relevant to the chatbot's domains.
    The chatbot can help with:
    1. General university information (established date, locations, campus, administration, departments, faculties, general courses list, libraries, hostels, student support cells).
    2. Detailed B.Tech CSE (Computer Science & Engineering) program details (admission exams like WBJEE, eligibility, semesters, subjects, labs, exam structures, certifications, career paths, project types, etc.).
    3. Student feedback and surveys (student ratings, reviews, opinions, complaints on hostels, WiFi, cafeteria, placements, teachers, libraries, etc.).
    4. General greetings or chatbot-related meta-questions (e.g. "hi", "who are you", "what can you do").

    A query is IRRELEVANT if it is completely unrelated to the university, admissions, academic programs, B.Tech CSE, student life, feedback, or the chatbot itself (e.g. queries about cooking recipes, unrelated sports news, programming code help unrelated to the syllabus, general politics, etc.).

    Analyze the query: "{query}"

    Respond in the following JSON format:
    {{
      "relevant": true/false,
      "reason": "Brief reason if irrelevant, or empty if relevant"
    }}
    Ensure the output contains ONLY the valid JSON block.
    """
    
    try:
        response = classifier_llm.invoke(prompt)
        content_str = get_clean_text(response.content)
        return json.loads(content_str.strip())
    except Exception as e:
        print(f"Error checking relevance: {e}")
        # Fallback to true in case of parsing errors
        return {
            "relevant": True,
            "reason": ""
        }

def classify_intent(query: str) -> dict:
    """
    Determines which databases/collections need to be searched for the query.
    """
    prompt = f"""
    You are a database selector for the University of Calcutta Student Support Chatbot.
    Your job is to determine which of the three databases contain information relevant to answering the user's query.
    The available databases and their contents are:
    1. "university_info": Contains general university information, campuses, administrative structure, general courses (non-CSE, or high-level lists), hostel allocation rules, libraries, student support services (Anti-Ragging, Grievance), and general FAQs.
    2. "cse_info": Contains specific curriculum, semester-wise subjects, labs, admission exams (WBJEE), eligibility, skills, tools, career roles, internships, and certifications specifically for the B.Tech CSE (Computer Science and Engineering) program.
    3. "student_feedback": Contains student feedback, survey reviews, ratings, and complaints categorized by Faculty, WiFi, Hostel, Placement, Cafeteria, Library, etc.

    Analyze the query: "{query}"

    Select one or more databases that are likely to contain information to answer the query. 
    If the query is a simple greeting (like "hi" or "hello") or a meta-question about the chatbot (like "who are you?") and does not require search, return an empty list.

    Respond in the following JSON format:
    {{
      "databases": ["database_name1", "database_name2"]
    }}
    Only choose from ["university_info", "cse_info", "student_feedback"].

    Ensure the output contains ONLY the JSON block.
    """
    
    try:
        response = classifier_llm.invoke(prompt)
        content_str = get_clean_text(response.content)
        data = json.loads(content_str.strip())
        valid_dbs = ["university_info", "cse_info", "student_feedback"]
        data["databases"] = [db for db in data.get("databases", []) if db in valid_dbs]
        return data
    except Exception as e:
        print(f"Error classifying intent: {e}")
        # Fallback to querying all databases
        return {
            "databases": ["university_info", "cse_info", "student_feedback"]
        }

def retrieve_context(query: str, databases: list) -> list:
    """
    Searches the selected Chroma collection(s) and returns matched documents.
    """
    retrieved_docs = []
    for db_name in databases:
        try:
            db = Chroma(
                collection_name=db_name,
                embedding_function=embeddings,
                persist_directory="./vector_store"
            )
            # Retrieve top 4 relevant chunks
            docs = db.similarity_search(query, k=4)
            for doc in docs:
                content_with_source = f"[Source: {db_name}] {doc.page_content}"
                retrieved_docs.append({
                    "content": content_with_source,
                    "metadata": doc.metadata,
                    "source": db_name
                })
        except Exception as e:
            print(f"Error querying collection {db_name}: {e}")
    return retrieved_docs

def generate_answer(query: str, context_docs: list) -> str:
    """
    Generates the final response based on user query and retrieved context snippets.
    """
    if not context_docs:
        prompt = f"""
        You are the University of Calcutta Student Support Chatbot, a friendly, warm, and highly expressive assistant.
        
        Your goal is to welcome the student and make them feel supported. Respond to their greeting or meta-question in an enthusiastic, engaging, and polite manner. 
        
        Introduce yourself warmly, explain your role, and invite them to ask questions. Be expressive and mention in detail the various areas you can help them with, including:
        1. General University Information (such as established date, campuses, faculties, departments, libraries, hostel allocation rules, and student support cells).
        2. Detailed B.Tech Computer Science & Engineering (CSE) Program details (such as curriculum details, semester-by-semester subjects, labs, admission exams like WBJEE, eligibility criteria, internship preparation, and career paths).
        3. Student Feedback & Surveys (what actual students say about the campus facilities, WiFi quality, hostels, placements, faculty, cafeteria, and student welfare services).
        
        End with an encouraging invite to ask any query they have in mind.
        
        Query: {query}
        
        Answer:
        """
    else:
        context_str = "\n\n".join([doc["content"] for doc in context_docs])
        prompt = f"""
        You are the University of Calcutta Student Support Chatbot, an expressive, highly detailed, and helpful academic support assistant. Your task is to provide a comprehensive, detailed, and beautifully formatted response to the student's query based ONLY on the provided context.
        
        To serve the student best, follow these guidelines:
        - **Be Expressive and Detailed**: Do not give short, single-sentence or overly brief summaries. Elaborate on the information available in the context. Provide complete explanations, break down multi-part answers, and include relevant background context from the provided text.
        - **Use Rich Formatting**: Make your response highly readable and engaging. Use headers, bullet points, numbered lists, bold text for key terms, and line breaks to organize the information logically.
        - **Maintain a Friendly and Encouraging Tone**: Speak in a warm, welcoming, and encouraging voice, similar to a dedicated university academic advisor.
        - **Maintain Clean Text**: Do NOT mention, cite, or reference any source names or file names (such as "university_info", "cse_info", "student_feedback", or [university_info], etc.) in your response. Do NOT include any "[Source: ...]" markers or raw metadata tags. Just provide a clean, direct, and well-formatted plain text answer to the user.
        - **Handling Missing Information**: If the context does not contain enough information to fully answer the question, clearly and politely state what details you have, answer as much as possible with the available facts, and explain what is missing.
        
        Context:
        {context_str}
        
        User Query: {query}
        
        Answer:
        """
    
    response = response_llm.invoke(prompt)
    return get_clean_text(response.content)

def generate_answer_stream(query: str, context_docs: list):
    """
    Generates the final response stream based on user query and retrieved context snippets.
    """
    if not context_docs:
        prompt = f"""
        You are the University of Calcutta Student Support Chatbot, a friendly, warm, and highly expressive assistant.
        
        Your goal is to welcome the student and make them feel supported. Respond to their greeting or meta-question in an enthusiastic, engaging, and polite manner. 
        
        Introduce yourself warmly, explain your role, and invite them to ask questions. Be expressive and mention in detail the various areas you can help them with, including:
        1. General University Information (such as established date, campuses, faculties, departments, libraries, hostel allocation rules, and student support cells).
        2. Detailed B.Tech Computer Science & Engineering (CSE) Program details (such as curriculum details, semester-by-semester subjects, labs, admission exams like WBJEE, eligibility criteria, internship preparation, and career paths).
        3. Student Feedback & Surveys (what actual students say about the campus facilities, WiFi quality, hostels, placements, faculty, cafeteria, and student welfare services).
        
        End with an encouraging invite to ask any query they have in mind.
        
        Query: {query}
        
        Answer:
        """
    else:
        context_str = "\n\n".join([doc["content"] for doc in context_docs])
        prompt = f"""
        You are the University of Calcutta Student Support Chatbot, an expressive, highly detailed, and helpful academic support assistant. Your task is to provide a comprehensive, detailed, and beautifully formatted response to the student's query based ONLY on the provided context.
        
        To serve the student best, follow these guidelines:
        - **Be Expressive and Detailed**: Do not give short, single-sentence or overly brief summaries. Elaborate on the information available in the context. Provide complete explanations, break down multi-part answers, and include relevant background context from the provided text.
        - **Use Rich Formatting**: Make your response highly readable and engaging. Use headers, bullet points, numbered lists, bold text for key terms, and line breaks to organize the information logically.
        - **Maintain a Friendly and Encouraging Tone**: Speak in a warm, welcoming, and encouraging voice, similar to a dedicated university academic advisor.
        - **Maintain Clean Text**: Do NOT mention, cite, or reference any source names or file names (such as "university_info", "cse_info", "student_feedback", or [university_info], etc.) in your response. Do NOT include any "[Source: ...]" markers or raw metadata tags. Just provide a clean, direct, and well-formatted plain text answer to the user.
        - **Handling Missing Information**: If the context does not contain enough information to fully answer the question, clearly and politely state what details you have, answer as much as possible with the available facts, and explain what is missing.
        
        Context:
        {context_str}
        
        User Query: {query}
        
        Answer:
        """
    
    for chunk in response_llm.stream(prompt):
        yield get_clean_text(chunk.content)


def run_agent_pipeline(query: str):
    # Relevance Check
    yield {
        "step": "relevance_check",
        "status": "running",
        "message": "Checking query relevance..."
    }
    
    rel_data = check_relevance(query)
    is_relevant = rel_data.get("relevant", True)
    reason = rel_data.get("reason", "Query is not related to university support.")
    
    if not is_relevant:
        yield {
            "step": "relevance_check",
            "status": "completed",
            "is_relevant": False,
            "reason": reason,
            "message": "Query classified as IRRELEVANT."
        }
        # Short circuit
        ans = f"I'm sorry, but your query seems to be outside the scope of the University Student Support Chatbot.\n\n**Reason:** {reason}\n\nI can only assist you with questions regarding university regulations, course details (B.Tech CSE), campus amenities, hostel rules, or student feedback/surveys."
        yield {
            "step": "final_answer",
            "status": "completed",
            "answer": ans
        }
        return
        
    yield {
        "step": "relevance_check",
        "status": "completed",
        "is_relevant": True,
        "message": "Query classified as RELEVANT."
    }
    
    # Intent Classification & DB Selection
    yield {
        "step": "db_selection",
        "status": "running",
        "message": "Identifying relevant information sources..."
    }
    
    intent_data = classify_intent(query)
    selected_dbs = intent_data.get("databases", [])
    
    yield {
        "step": "db_selection",
        "status": "completed",
        "databases": selected_dbs,
        "message": f"Selected source databases: {', '.join(selected_dbs) if selected_dbs else 'None (Direct Chat)'}"
    }
    
    # Retrieval
    context_docs = []
    if selected_dbs:
        yield {
            "step": "retrieval",
            "status": "running",
            "message": f"Querying database(s): {', '.join(selected_dbs)}..."
        }
        
        context_docs = retrieve_context(query, selected_dbs)
        
        yield {
            "step": "retrieval",
            "status": "completed",
            "doc_count": len(context_docs),
            "message": f"Retrieved {len(context_docs)} relevant context snippet(s)."
        }
    else:
        yield {
            "step": "retrieval",
            "status": "completed",
            "doc_count": 0,
            "message": "No database search needed for this query."
        }
        
    # Answer Generation
    yield {
        "step": "synthesis",
        "status": "running",
        "message": "Synthesizing final answer..."
    }
    
    full_answer = ""
    for chunk in generate_answer_stream(query, context_docs):
        full_answer += chunk
        yield {
            "step": "final_answer",
            "status": "running",
            "chunk": chunk,
            "answer": full_answer
        }
    
    yield {
        "step": "synthesis",
        "status": "completed",
        "message": "Response successfully compiled!"
    }
    
    yield {
        "step": "final_answer",
        "status": "completed",
        "answer": full_answer
    }
