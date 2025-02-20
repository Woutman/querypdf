import asyncio

import streamlit as st

from rag.ingestion import ingest_pdf_async
from rag.rag import generate_answer

st.title("QueryPDF")

# Upload PDF
pdf_file = st.file_uploader("Upload a PDF", type="pdf")
if pdf_file is not None and not st.session_state.get("pdf_uploaded", False):
    with st.spinner("Processing PDF..."):
        asyncio.run(ingest_pdf_async(pdf_file))
        st.session_state.pdf_uploaded = True
        st.success("Document indexed! You can now ask questions about the PDF.")


# Chat interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_input" not in st.session_state:
    st.session_state.user_input = ""


def _handle_query() -> None:
    """Callback function to handle sent queries on button press."""
    query = st.session_state.user_input.strip()
    if not query:
        return
    
    with st.spinner("Generating answer..."):
        st.session_state.chat_history.append({"role": "user", "content": query})
        answer = generate_answer(message_history=st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    st.session_state.user_input = ""


with st.container(border=True):
    for message in st.session_state.chat_history:
        if message['role'] == 'User':
            st.markdown(f"**{message['role'].upper()}:** _{message['content']}_")
        else:
            st.markdown(f"**{message['role'].upper()}:** {message['content']}")

    st.text_input("Ask a question about the PDF", key="user_input", on_change=_handle_query)    
    st.button("Send", on_click=_handle_query)
