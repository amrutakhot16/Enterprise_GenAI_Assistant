import streamlit as st
import requests
import uuid
# =====================================
# Configuration
# =====================================

API_URL = "http://127.0.0.1:8002/chat"

st.set_page_config(
    page_title="Enterprise GenAI Assistant",
    page_icon="🤖",
    layout="centered"
)

# =====================================
# Title
# =====================================

st.title("🤖 Enterprise GenAI Assistant")


# =====================================
# Initialize chat history
# =====================================

if "messages" not in st.session_state:

    st.session_state.messages = []

# Each browser tab gets its own conversation on the backend
if "session_id" not in st.session_state:

    st.session_state.session_id = str(uuid.uuid4())


if st.sidebar.button("New chat"):

    try:
        requests.delete(
            f"{API_URL}/{st.session_state.session_id}",
            timeout=10
        )
    except requests.exceptions.RequestException:
        pass

    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

# =====================================
# Display previous messages
# =====================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        # Display sources for assistant messages
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            st.subheader("Sources")

            for source in message["sources"]:

                st.write(
                    f"- {source['file']} "
                    f"(Page {source['page']})"
                )


# =====================================
# Chat input
# =====================================

question = st.chat_input(
    "Ask a question about company documents..."
)

# =====================================
# Process question
# =====================================

if question:

    # ---------------------------------
    # Display user question
    # ---------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # ---------------------------------
    # Call FastAPI backend
    # ---------------------------------

    try:

        response = requests.post(
            API_URL,
                        json={
                "question": question,
                "session_id": st.session_state.session_id
            },
            timeout=120
        )


    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to the backend. "
            "Make sure FastAPI is running on port 8002."
        )

        st.stop()


    except requests.exceptions.Timeout:

        st.error(
            "The backend took too long to respond."
        )

        st.stop()


    except requests.exceptions.RequestException as e:

        st.error(
            f"Request failed: {e}"
        )

        st.stop()


    # ---------------------------------
    # Check HTTP response
    # ---------------------------------

    if response.status_code != 200:

        st.error(
            f"Backend error: {response.status_code}"
        )

        st.code(
            response.text
        )

        st.stop()


    # ---------------------------------
    # Parse JSON
    # ---------------------------------

    try:

        data = response.json()

    except ValueError:

        st.error(
            "Backend returned an invalid JSON response."
        )

        st.code(
            response.text
        )

        st.stop()


    # ---------------------------------
    # Get answer
    # ---------------------------------

    answer = data.get(
        "answer",
        ""
    )


    sources = data.get(
        "sources",
        []
    )

    # ---------------------------------
    # Display assistant response
    # ---------------------------------

    with st.chat_message("assistant"):

        st.markdown(answer)


        # ---------------------------------
        # Display sources
        # ---------------------------------

        if sources:

            st.subheader("Sources")

            for source in sources:

                st.write(
                    f"- {source['file']} "
                    f"(Page {source['page']})"
                )


    # ---------------------------------
    # Save assistant response
    # ---------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )