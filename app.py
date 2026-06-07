import streamlit as st
import time
import os
from dotenv import load_dotenv
from agent import run_agent_pipeline

# Page configuration
st.set_page_config(
    page_title="Calcutta University Student Support Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load env
load_dotenv()

# Inject custom CSS for premium aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Apply modern font globally */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Sleek gradient background */
    .stApp {
        background: linear-gradient(135deg, #0e121a 0%, #171d2c 100%) !important;
        color: #f1f5f9 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0b0e14 !important;
        border-right: 1px solid #1e293b !important;
        padding-top: 2rem !important;
    }
    
    /* Titles and Header styles */
    h1, h2, h3 {
        font-weight: 700 !important;
    }
    
    .main-title {
        background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .subtitle {
        color: #94a3b8 !important;
        font-size: 1.1rem;
        margin-bottom: 2rem !important;
    }
    
    /* Glassmorphic Cards */
    .db-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .db-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3);
        box-shadow: 0 8px 30px rgba(56, 189, 248, 0.1);
        background: rgba(30, 41, 59, 0.6);
    }
    
    /* Dynamic database status lights */
    .status-light {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 8px currentColor;
    }
    
    .status-active {
        color: #10b981;
        background-color: #10b981;
    }
    
    .status-label {
        font-weight: 500;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
    }
    
    /* Chat message bubble customizations */
    div[data-testid="stChatMessage"] {
        background-color: rgba(30, 41, 59, 0.25) !important;
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        margin-bottom: 1rem !important;
        padding: 1rem !important;
    }
    
    div[data-testid="stChatMessage"][data-user="true"] {
        background-color: rgba(168, 85, 247, 0.08) !important;
        border-color: rgba(168, 85, 247, 0.15) !important;
    }
    
    /* Agent flow logger container */
    div[data-testid="stStatusWidget"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 10px !important;
    }
    
    /* Gradient line divider */
    .gradient-line {
        height: 2px;
        background: linear-gradient(90deg, #38bdf8, #a855f7, transparent);
        margin-bottom: 1.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
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
st.sidebar.markdown("<div style='text-align: center;'><h2 style='color: #f1f5f9; margin-bottom: 0px;'>🎓 Support Agent</h2><p style='color: #64748b; font-size: 0.85rem; margin-bottom: 1.5rem;'>University of Calcutta</p></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='gradient-line'></div>", unsafe_allow_html=True)

# Database Status Section
st.sidebar.markdown("### 🗄️ Knowledge Sources")

# Helper for displaying database card
def db_status_card(name, label, desc):
    vector_store_exists = os.path.exists("./vector_store")
    status_class = "status-active" if vector_store_exists else ""
    status_text = "Online" if vector_store_exists else "Not Initialized"
    
    st.sidebar.markdown(f"""
    <div class="db-card">
        <div class="status-label">
            <span class="status-light {status_class}"></span>
            <span style="font-weight: 600; color: #f1f5f9;">{label}</span>
        </div>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">{desc}</div>
        <div style="font-size: 0.75rem; color: #38bdf8; margin-top: 8px; font-weight: 500;">Status: {status_text}</div>
    </div>
    """, unsafe_allow_html=True)

db_status_card("university_info", "University Regulations", "Campuses, hostels, departments, libraries, general rules & FAQs.")
db_status_card("cse_info", "B.Tech CSE Database", "Detailed 8-semester curriculum, subjects, eligibility, WBJEE details, certifications.")
db_status_card("student_feedback", "Student Reviews & Ratings", "Aggregated feedback and ratings on faculty, placements, cafeteria, WiFi, hostel.")

# Suggested Queries Section
st.sidebar.markdown("### 💡 Try Asking")

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
st.markdown("<h1 class='main-title'>🎓 University AI Student Support</h1>", unsafe_allow_html=True)
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
        status_placeholder = st.status("🔮 Agent reasoning and retrieval...", expanded=True)
        with status_placeholder:
            final_answer = None
            
            for update in run_agent_pipeline(query_input):
                step = update["step"]
                status = update["status"]
                msg = update.get("message", "")
                
                if status == "running":
                    st.markdown(f"⏳ **{msg}**")
                elif status == "completed":
                    if step == "relevance_check":
                        is_relevant = update.get("is_relevant", True)
                        if is_relevant:
                            st.markdown(f"✅ **{msg}**")
                        else:
                            st.markdown(f"❌ **{msg}** (Reason: {update.get('reason')})")
                    elif step == "db_selection":
                        dbs = update.get("databases", [])
                        st.markdown(f"🗂️ **{msg}**")
                    elif step == "retrieval":
                        st.markdown(f"🔍 **{msg}**")
                    elif step == "synthesis":
                        st.markdown(f"✨ **{msg}**")
                    elif step == "final_answer":
                        final_answer = update.get("answer")
            
            status_placeholder.update(
                label="✅ Reasoning & Retrieval complete" if final_answer else "❌ Query blocked", 
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
