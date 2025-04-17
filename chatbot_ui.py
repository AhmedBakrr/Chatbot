
import streamlit as st
from ai_manager import AIManager

# Initialize RAG pipeline
ai = AIManager()

# Page config
st.set_page_config(page_title="ACS Assistant", page_icon="🩺", layout="centered")

# Inject custom CSS for styling and theming
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .chat-container {
        display: flex;
        flex-direction: column;
    }

    .user-bubble, .bot-bubble {
        padding: 0.75em 1em;
        border-radius: 1em;
        margin-bottom: 1em;
        max-width: 90%;
        font-size: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }

    .user-bubble:hover, .bot-bubble:hover {
        transform: scale(1.02);
    }

    .user-bubble {
        background: linear-gradient(135deg, #cce6ff, #99ccff);
        color: #003366;
        align-self: flex-end;
    }

    .bot-bubble {
        background: linear-gradient(135deg, #e6ffe6, #ccffcc);
        color: #004d00;
        align-self: flex-start;
    }

    .stTextInput > div > input {
        font-size: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🩺 ACS Medical chatbot")
st.markdown("Ask any question about **Acute Coronary Syndrome (ACS)** below:")

# Chat history setup
if "history" not in st.session_state:
    st.session_state.history = []

# Input form
with st.form("chat_form", clear_on_submit=True):
    user_query = st.text_input(
        "Your question:",
        placeholder="e.g., What is ACS?",
        label_visibility="collapsed"
    )
    submit = st.form_submit_button("Ask")

# Handle submission
if submit and user_query:
    with st.spinner("Thinking..."):
        response = ai.ask(user_query)
        st.session_state.history.append({"query": user_query, "response": response})

# Display conversation
if st.session_state.history:
    st.markdown("### 🧠 Conversation History")
    for turn in reversed(st.session_state.history):
        st.markdown(f"""
            <div class='chat-container'>
                <div class='user-bubble'><strong>You:</strong> {turn['query']}</div>
                <div class='bot-bubble'><strong>ACS Assistant:</strong> {turn['response']}</div>
            </div>
        """, unsafe_allow_html=True)
