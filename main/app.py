import streamlit as st

from ingestion import ingest_pdf


st.title("QueryPDF")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")
if pdf_file is not None:
    ingest_pdf(pdf_file)

    st.success("Document indexed! You can now ask questions about the PDF.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question = st.text_input("Ask a question about the PDF")
    if st.button("Send") and user_question:
        st.session_state.chat_history.append(("User", user_question))

        # TODO: RAG logic
        answer = ""
        st.session_state.chat_history.append(("Bot", answer))

    # Display the conversation history.
    for speaker, message in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {message}")
