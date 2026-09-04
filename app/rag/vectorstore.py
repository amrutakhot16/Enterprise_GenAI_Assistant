import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DOCUMENTS_PATH = "documents"
VECTORSTORE_PATH = "vectorstore"

def create_vector_store():

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---------------------------------
    # Load existing vector store
    # ---------------------------------

    if os.path.exists(VECTORSTORE_PATH):

        print("Loading existing vector store...")

        return Chroma(
            persist_directory=VECTORSTORE_PATH,
            embedding_function=embeddings
        )

    # ---------------------------------
    # Create vector store
    # ---------------------------------

    print("Creating vector store...")

    documents = []

    for file in os.listdir(DOCUMENTS_PATH):

        if not file.lower().endswith(".pdf"):
            continue

        path = os.path.join(
            DOCUMENTS_PATH,
            file
        )

        print("Loading:", path)

        loader = PyPDFLoader(path)

        loaded_pages = loader.load()

        print(
            f"Loaded PDF pages: {len(loaded_pages)}"
        )

        documents.extend(
            loaded_pages
        )

    # ---------------------------------
    # Split documents
    # ---------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        "Created chunks:",
        len(chunks)
    )

    # ---------------------------------
    # Debug chunks
    # ---------------------------------

    print(
        "\n========== CREATED CHUNKS =========="
    )

    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        source = chunk.metadata.get(
            "source",
            "unknown"
        )

        page = chunk.metadata.get(
            "page",
            None
        )

        page_display = (
            page + 1
            if page is not None
            else "unknown"
        )

        print(
            f"\n--- Chunk {i} ---"
        )

        print(
            f"Source: {source}"
        )

        print(
            f"Page: {page_display}"
        )

        print(
            "Content:"
        )

        print(
            chunk.page_content
        )

    print(
        "\n===================================="
    )

    # ---------------------------------
    # Create Chroma vector store
    # ---------------------------------

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )

    print(
        "\nVector store created successfully."
    )

    return vectorstore