import streamlit as st
import time
import os
from dotenv import load_dotenv
from agent import run_agent_pipeline

# Page configuration
st.set_page_config(
    page_title="Calcutta University Student Support Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load env
load_dotenv()

# Inject custom CSS for premium aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Apply modern font globally */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Solid dark background */
    .stApp {
        background-color: #121214 !important;
        color: #f4f4f5 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #18181b !important;
        border-right: 1px solid #27272a !important;
        padding-top: 2rem !important;
    }
    
    /* Titles and Header styles */
    h1, h2, h3 {
        font-weight: 600 !important;
        color: #ffffff !important;
    }
    
    .main-title {
        color: #ffffff !important;
        font-size: 2.2rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .subtitle {
        color: #a1a1aa !important;
        font-size: 1.05rem;
        margin-bottom: 2rem !important;
    }
    
    /* Solid Cards */
    .db-card {
        background-color: #1f1f23 !important;
        border: 1px solid #27272a !important;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 12px;
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }
    
    .db-card:hover {
        border-color: #3f3f46 !important;
        background-color: #27272a !important;
    }
    
    /* Dynamic database status lights */
    .status-light {
        height: 8px;
        width: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .status-active {
        background-color: #10b981;
    }
    
    .status-label {
        font-weight: 500;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
    }
    
    /* Chat message bubble customizations */
    div[data-testid="stChatMessage"] {
        background-color: #18181b !important;
        border: 1px solid #27272a !important;
        border-radius: 6px !important;
        margin-bottom: 1rem !important;
        padding: 1rem !important;
    }
    
    div[data-testid="stChatMessage"][data-user="true"] {
        background-color: #27272a !important;
        border-color: #3f3f46 !important;
    }
    
    /* Agent flow logger container */
    div[data-testid="stStatusWidget"] {
        background-color: #18181b !important;
        border: 1px solid #27272a !important;
        border-radius: 6px !important;
    }
    
    /* Solid line divider */
    .solid-line {
        height: 1px;
        background-color: #27272a;
        margin-bottom: 1.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #71717a;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #27272a;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your University of Calcutta Student Support Agent. I can assist you with details regarding course structure (B.Tech CSE), admission eligibility, campus facilities, general university rules, and student feedback. How can I help you today?"}
    ]

if "click_query" not in st.session_state:
    st.session_state.click_query = None

# Sidebar Design
st.sidebar.markdown("<div style='text-align: center;'><h2 style='color: #ffffff; margin-bottom: 0px;'>Support Agent</h2><p style='color: #a1a1aa; font-size: 0.85rem; margin-bottom: 1.5rem;'>University of Calcutta</p></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='solid-line'></div>", unsafe_allow_html=True)

# Database Status Section
st.sidebar.markdown("### Knowledge Sources")

# Helper for displaying database card
def db_status_card(name, label, desc):
    vector_store_exists = os.path.exists("./vector_store")
    status_class = "status-active" if vector_store_exists else ""
    status_text = "Online" if vector_store_exists else "Not Initialized"
    
    st.sidebar.markdown(f"""
    <div class="db-card">
        <div class="status-label">
            <span class="status-light {status_class}"></span>
            <span style="font-weight: 600; color: #ffffff;">{label}</span>
        </div>
        <div style="font-size: 0.8rem; color: #a1a1aa; margin-top: 4px;">{desc}</div>
        <div style="font-size: 0.75rem; color: #71717a; margin-top: 8px; font-weight: 500;">Status: {status_text}</div>
    </div>
    """, unsafe_allow_html=True)

db_status_card("university_info", "University Regulations", "Campuses, hostels, departments, libraries, general rules & FAQs.")
db_status_card("cse_info", "B.Tech CSE Database", "Detailed 8-semester curriculum, subjects, eligibility, WBJEE details, certifications.")
db_status_card("student_feedback", "Student Reviews & Ratings", "Aggregated feedback and ratings on faculty, placements, cafeteria, WiFi, hostel.")

# Suggested Queries Section
st.sidebar.markdown("### Suggested Queries")

suggestions = [
    ("CSE Eligibility", "What are the eligibility criteria and admission exams for B.Tech CSE?"),
    ("Semester 1 & 2 Subjects", "What subjects are taught in the first and second semesters of B.Tech CSE?"),
    ("Hostel Allocation & Rules", "What are the rules and basis for hostel allocation in the university?"),
    ("WiFi & Hostel Feedback", "What are the student ratings and feedback regarding hostel maintenance and WiFi?"),
    ("Placement Preparation & Roles", "What career roles and internship areas does the B.Tech CSE program prepare you for?")
]

for label, query_text in suggestions:
    if st.sidebar.button(label, use_container_width=True):
        st.session_state.click_query = query_text

# Main Page Layout
st.markdown("<h1 class='main-title'>University Student Support</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Instant intelligent answers about university rules, curriculum, and student feedback using Gemini API & LangChain RAG pipeline.</p>", unsafe_allow_html=True)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Check if there is a clicked query from the sidebar
query_input = st.chat_input("Ask a question about the university...")
if st.session_state.click_query:
    query_input = st.session_state.click_query
    st.session_state.click_query = None  # reset

# Handle User Input
if query_input:
    # Render user query immediately
    with st.chat_message("user"):
        st.write(query_input)
    st.session_state.messages.append({"role": "user", "content": query_input})
    
    # Process with the agent pipeline and show step-by-step reasoning flow
    with st.chat_message("assistant"):
        # We use st.status to display the step-by-step execution flow of the agent
        status_placeholder = st.status("Agent reasoning and retrieval...", expanded=True)
        with status_placeholder:
            final_answer = None
            
            for update in run_agent_pipeline(query_input):
                step = update["step"]
                status = update["status"]
                msg = update.get("message", "")
                
                if status == "running":
                    st.markdown(f"**{msg}**")
                elif status == "completed":
                    if step == "relevance_check":
                        is_relevant = update.get("is_relevant", True)
                        if is_relevant:
                            st.markdown(f"**{msg}**")
                        else:
                            st.markdown(f"**{msg}** (Reason: {update.get('reason')})")
                    elif step == "db_selection":
                        dbs = update.get("databases", [])
                        st.markdown(f"**{msg}**")
                    elif step == "retrieval":
                        st.markdown(f"**{msg}**")
                    elif step == "synthesis":
                        st.markdown(f"**{msg}**")
                    elif step == "final_answer":
                        final_answer = update.get("answer")
            
            status_placeholder.update(
                label="Reasoning & Retrieval complete" if final_answer else "Query blocked", 
                state="complete" if final_answer else "error", 
                expanded=False
            )
            
        # Write final answer outside status
        if final_answer:
            st.write(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
            # Simple rerun to update view smoothly
            st.rerun()

st.markdown("<div class='footer'> Calcutta University AI Support Agent</div>", unsafe_allow_html=True)
