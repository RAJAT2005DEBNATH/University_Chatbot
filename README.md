# University Student Support Chatbot

An AI-driven assistant designed for the University of Calcutta, with a special focus on the B.Tech Computer Science and Engineering (CSE) program. The agent is built using the Gemini API, LangChain, and ChromaDB, presenting a detailed step-by-step reasoning flow in a Streamlit interface.

## Features
- Query Relevance Filter: Evaluates queries to filter out off-topic requests before processing.
- Intelligent Router: Classifies queries to target the relevant databases (general university regulations, B.Tech CSE details, or student feedback) or a combination of them.
- Multi-Source RAG: Conducts similarity searches across multiple vector stores and merges the context.
- Sanitized Output: Cleans response content to output plain text answers, removing database source tags, JSON wrappers, or signature properties.
- Reasoning Progress Flow: Displays step-by-step actions in the UI (relevance check status, chosen databases, and document retrieval count) in real time.

## Tech Stack
- Frontend: Streamlit (interactive chat interface with custom CSS styling)
- AI Orchestration: LangChain (agents, document loaders, and embedding integration)
- LLM: Google Gemini API (gemini-2.5-flash for intent routing and response synthesis)
- Embeddings: Google Generative AI Embeddings (gemini-embedding-001)
- Vector Database: ChromaDB (multi-collection vector storage)
- Configuration: python-dotenv (environment variables management)

## Prerequisites
- Python 3.10 or higher
- A Google Gemini API Key

## How to Download and Set Up
1. Clone the repository:
   ```bash
   git clone https://github.com/username/university-chatbot.git
   cd university-chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   Create a .env file in the root directory and add your Google Gemini API key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

## How to Launch
1. Populate the Vector Database:
   Ingest and index the source data from the data directory into ChromaDB:
   ```bash
   python ingest.py
   ```

2. Start the Streamlit Application:
   ```bash
   streamlit run app.py
   ```

3. Open the application:
   Open your browser and navigate to the address displayed in the terminal (by default, http://localhost:8501 or http://localhost:8502).
