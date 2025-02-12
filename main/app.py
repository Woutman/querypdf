import streamlit as st

from main.rag.ingestion import ingest_pdf
from main.rag.rag import generate_answer
from main.llm.instructions import INSTRUCTIONS_CHATBOT


st.title("QueryPDF")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")
if pdf_file is not None:
    ingest_pdf(pdf_file)

    st.success("Document indexed! You can now ask questions about the PDF.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "system", "content": INSTRUCTIONS_CHATBOT}]

    for message in st.session_state.chat_history:
        st.markdown(f"**{message['role']}:** {message['content']}")

    user_question = st.text_input("Ask a question about the PDF")
    if st.button("Send") and user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # TODO: RAG logic
        answer = generate_answer(message_history=st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
