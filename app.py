import os
import streamlit as st
from datetime import datetime
from ReportAnalyzer import ReportAnalyzer


def main():
    st.set_page_config(page_title="OrbAid Demo", layout="wide")

    # Initialize session state
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "report_1"
    if "screen" not in st.session_state:
        st.session_state.screen = "upload"

    analyzer = ReportAnalyzer()

    # Upload screen
    if st.session_state.screen == "upload":
        st.title("📄 Upload Sustainability Report")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                try:
                    chunks, text = analyzer.process_pdf(uploaded_file)
                    st.session_state.vectorstore = analyzer.create_vectorstore(
                        chunks, st.session_state.thread_id)
                    answers = analyzer.generate_summary(
                        st.session_state.vectorstore)
                    summarized_report = analyzer.generate_summarized_report(
                        answers)
                    chart_paths = analyzer.generate_charts(
                        answers, st.session_state.thread_id)

                    # Add summary and charts to chat
                    timestamp = datetime.now().strftime("%H:%M")
                    st.session_state.chat_history.append(
                        ("AI", f"**📄 Summary:**\n\n{summarized_report}", timestamp))
                    for path in chart_paths:
                        st.session_state.chat_history.append(
                            ("AI", f"[CHART_IMAGE]{path}", timestamp))

                    st.session_state.screen = "chat"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")

    # Chat screen
    elif st.session_state.screen == "chat":
        st.title("💬 Sustainability Chat")

        # Chat history rendering
        for sender, message, timestamp in st.session_state.chat_history:
            with st.chat_message("user" if sender == "You" else "assistant"):
                if message.startswith("[CHART_IMAGE]"):
                    image_path = message.replace("[CHART_IMAGE]", "")
                    if os.path.exists(image_path):
                        st.image(image_path)
                else:
                    st.markdown(f"{message}")

        # Chat input
        user_prompt = st.chat_input("Ask something about the report...")
        if user_prompt:
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.chat_history.append(
                ("You", user_prompt, timestamp))
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Generating answer..."):
                    try:
                        retriever = st.session_state.vectorstore.as_retriever(
                            search_kwargs={"k": 4})
                        docs = retriever.invoke(user_prompt)
                        context = "\n".join(
                            [doc.page_content for doc in docs]) if docs else "No relevant data found."
                        answer = analyzer.answer_question(user_prompt, context)
                        st.markdown(answer)
                        st.session_state.chat_history.append(
                            ("AI", answer, timestamp))
                    except Exception as e:
                        error_msg = f"Error: {e}"
                        st.markdown(error_msg)
                        st.session_state.chat_history.append(
                            ("AI", error_msg, timestamp))

        # Reset button
        if st.button("🔄 Upload New Report"):
            st.session_state.chat_history = []
            st.session_state.vectorstore = None
            st.session_state.screen = "upload"
            for file in os.listdir("temp"):
                if file.endswith(".png"):
                    os.remove(os.path.join("temp", file))
            st.rerun()


if __name__ == "__main__":
    main()
