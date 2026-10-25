import json
import uuid

import requests
import streamlit as st

# =====================================
# Configuration
# =====================================

API_BASE = "http://127.0.0.1:8002"

st.set_page_config(
    page_title="Enterprise GenAI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Enterprise GenAI Assistant")


# =====================================
# Session state
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Each browser tab gets its own conversation on the backend
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


def show_sources(sources):
    if sources:
        st.subheader("Sources")
        for source in sources:
            st.write(f"- {source['file']} (Page {source['page']})")


def error_detail(response):
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


# =====================================
# Sidebar: new chat + documents
# =====================================

with st.sidebar:

    if st.button("New chat"):
        try:
            requests.delete(
                f"{API_BASE}/chat/{st.session_state.session_id}",
                timeout=10
            )
        except requests.exceptions.RequestException:
            pass

        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.header("Documents")

    uploaded = st.file_uploader("Add a PDF", type=["pdf"])

    if uploaded and st.button("Add to knowledge base"):
        with st.spinner(f"Indexing {uploaded.name}..."):
            try:
                response = requests.post(
                    f"{API_BASE}/documents",
                    files={
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            "application/pdf"
                        )
                    },
                    timeout=300
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Upload failed: {e}")
            else:
                if response.ok:
                    chunks = response.json()["chunks"]
                    if chunks:
                        st.success(f"Added {uploaded.name} ({chunks} chunks)")
                    else:
                        st.warning(
                            f"{uploaded.name} was added, but no text was found "
                            "in it (it may be a scanned image)."
                        )
                else:
                    st.error(error_detail(response))

    try:
        documents = requests.get(
            f"{API_BASE}/documents",
            timeout=10
        ).json()["documents"]

        for name in documents:
            st.caption(f"📄 {name}")

    except requests.exceptions.RequestException:
        st.caption("Backend not reachable.")


# =====================================
# Previous messages
# =====================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            show_sources(message.get("sources"))


# =====================================
# Ask a question
# =====================================

question = st.chat_input("Ask a question about company documents...")

if question:

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    # Filled in when the "done" event arrives
    result = {"sources": []}

    def token_stream():
        with requests.post(
            f"{API_BASE}/chat/stream",
            json={
                "question": question,
                "session_id": st.session_state.session_id
            },
            stream=True,
            timeout=120
        ) as response:

            if not response.ok:
                raise RuntimeError(
                    f"Backend error {response.status_code}: "
                    f"{error_detail(response)}"
                )

            # chunk_size=None: hand over each line as soon as it arrives
            for line in response.iter_lines(chunk_size=None):
                if not line:
                    continue

                event = json.loads(line)

                if event["type"] == "token":
                    yield event["text"]
                elif event["type"] == "done":
                    result["sources"] = event.get("sources", [])

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(token_stream())

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the backend. "
                "Make sure FastAPI is running on port 8002."
            )
            st.stop()

        except requests.exceptions.Timeout:
            st.error("The backend took too long to respond.")
            st.stop()

        except (requests.exceptions.RequestException, RuntimeError) as e:
            st.error(str(e))
            st.stop()

        show_sources(result["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": result["sources"]
        }
    )