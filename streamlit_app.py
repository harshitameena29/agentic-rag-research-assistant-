import os
import requests
import streamlit as st

BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000"
)

st.set_page_config(
    page_title="RAGent",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

header_left, header_right = st.columns([8.5, 1.5])

with header_left:
    st.markdown("## ✦ RAGent")
    st.caption("AI Research Assistant · Ask. Research. Synthesize.")

with header_right:
    st.success("● Online")

st.write("")

left, right = st.columns([3, 7], gap="large")

with left:
    st.markdown("### 📄 Your Document")
    st.caption("Upload a PDF and ask questions about its contents.")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="visible",
    )

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)

        st.success(f"✓ {uploaded_file.name}")
        st.caption(f"{file_size_mb:.2f} MB")
        st.write("**Status:** Ready for research")
    else:
        st.info("Upload a PDF to start researching.")

with right:
    st.markdown("### 💬 Research Assistant")
    st.caption("Ask questions about your uploaded document.")

    if not st.session_state.messages:
        st.info(
            """
            👋 **Welcome to RAGent**

            Upload a PDF on the left and ask me anything about it.

            **I can:**

            🔎 Search your document  
            🧠 Break complex questions into sub-questions  
            🌐 Perform web research when required  
            ✍️ Generate a synthesized answer
            """
        )

    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])

                if message.get("sub_questions"):
                    with st.expander("🧠 View agent research plan"):
                        for i, question_text in enumerate(
                            message["sub_questions"],
                            start=1,
                        ):
                            st.markdown(f"**{i}.** {question_text}")

question = st.chat_input("Ask anything about your document...")

if question:
    if uploaded_file is None:
        st.warning("Please upload a PDF from the left side first.")
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("🤖 Researching your document..."):
            try:
                file_bytes = uploaded_file.getvalue()

                files = {
                    "file": (
                        uploaded_file.name,
                        file_bytes,
                        "application/pdf",
                    )
                }

                data = {
                    "question": question
                }

                response = requests.post(
                     f"{BACKEND_URL}/research",
                    files=files,
                    data=data,
                    timeout=300,
                )

                response.raise_for_status()

                result = response.json()

                report = result.get(
                    "report",
                    "I couldn't generate an answer.",
                )

                sub_questions = result.get(
                    "sub_questions",
                    [],
                )

                st.markdown(report)

                if sub_questions:
                    with st.expander("🧠 View agent research plan"):
                        for i, q in enumerate(
                            sub_questions,
                            start=1,
                        ):
                            st.markdown(f"**{i}.** {q}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": report,
                        "sub_questions": sub_questions,
                    }
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    """
                    ❌ **Could not connect to FastAPI.**

                    Make sure the backend is running:

                    `uvicorn api:app --reload --port 8000`
                    """
                )

            except requests.exceptions.Timeout:
                st.error(
                    """
                    ⏱️ **The request took too long.**

                    Please try again.
                    """
                )

            except requests.exceptions.HTTPError:
                try:
                    error_message = response.json().get(
                        "detail",
                        "Backend error",
                    )
                except Exception:
                    error_message = response.text

                st.error(f"❌ **Backend error:** {error_message}")

            except requests.exceptions.RequestException as e:
                st.error(f"❌ **Request failed:** {e}")

            except Exception as e:
                st.error(f"❌ **Something went wrong:** {e}")