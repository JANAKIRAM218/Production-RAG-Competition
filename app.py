from rag_backend import ask_rag
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Zyro HR Assistant",
    page_icon="🏢",
    layout="wide"
)

# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>
.stApp {
    background-color: #050816;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.title("🏢 Zyro HR Assistant")

st.caption(
    "Ask questions about leave, benefits, work-from-home, onboarding, travel, and company policies."
)

# ============================================================
# QUICK ACTIONS
# ============================================================

col1, col2, col3 = st.columns(3)

question = None

with col1:
    if st.button("🏖 Leave Policy", use_container_width=True):
        question = "How many earned leaves do employees get?"

with col2:
    if st.button("🏠 Work From Home", use_container_width=True):
        question = "Can I work from home?"

with col3:
    if st.button("👶 Maternity Leave", use_container_width=True):
        question = "What is the maternity leave policy?"

# ============================================================
# CLEAR CHAT
# ============================================================

col1, col2 = st.columns([10, 1])

with col2:
    if st.button("🗑"):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        
        # Backward compatibility for legacy string-based message history
        if message["role"] == "assistant" and isinstance(content, str) and "Sources:" in content:
            answer_text, source_text = content.split("Sources:", maxsplit=1)
            st.markdown(answer_text)
            with st.expander("📚 Sources Used", expanded=False):
                lines = []
                for line in source_text.strip().split("\n"):
                    line = line.strip().lstrip("-").strip()
                    if line:
                        if line.startswith("📄"):
                            line = line[1:].strip()
                        lines.append(line)
                st.markdown("\n".join(f"- 📄 {line}" for line in lines))
        else:
            st.markdown(content)
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 Sources Used", expanded=False):
                    st.markdown("\n".join(f"- 📄 {s}" for s in message["sources"]))

# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask a question about Zyro Dynamics HR policies..."
)

if user_input:
    question = user_input

# ============================================================
# PROCESS QUESTION
# ============================================================

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            result = ask_rag(question)
            answer = result["answer"]
            sources = result["sources"]

        st.markdown(answer)

        if sources:
            with st.expander("📚 Sources Used", expanded=False):
                st.markdown("\n".join(f"- 📄 {s}" for s in sources))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "FAISS • BGE Large Embeddings • Cross Encoder Reranking • Llama 3.3 70B"
)