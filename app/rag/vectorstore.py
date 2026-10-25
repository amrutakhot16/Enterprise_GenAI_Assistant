import hashlib
import json
import os
import threading

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCUMENTS_PATH = "documents"
VECTORSTORE_PATH = "vectorstore"
MANIFEST_PATH = os.path.join(VECTORSTORE_PATH, "manifest.json")
MAX_UPLOAD_MB = 20

# Stops two uploads from writing the manifest at the same time
_index_lock = threading.Lock()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
)


# ---------------------------------
# remembers which files are indexed
# ---------------------------------

def _fingerprint(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _load_manifest():
    if not os.path.exists(MANIFEST_PATH):
        return {}
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def _save_manifest(manifest):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------
# Indexing
# ---------------------------------

def remove_file(vectorstore, path):
    existing = vectorstore.get(where={"source": path})
    if existing["ids"]:
        vectorstore.delete(ids=existing["ids"])


def index_file(vectorstore, path):
    # Remove old chunks first so a changed file isn't indexed twice
    remove_file(vectorstore, path)

    pages = PyPDFLoader(path).load()
    chunks = splitter.split_documents(pages)

    if chunks:
        vectorstore.add_documents(chunks)

    return len(chunks)


def sync_documents(vectorstore):
    """Index new or changed PDFs and drop deleted ones."""

    manifest = _load_manifest()
    current = {}

    for name in sorted(os.listdir(DOCUMENTS_PATH)):
        if not name.lower().endswith(".pdf"):
            continue

        path = os.path.join(DOCUMENTS_PATH, name)
        fingerprint = _fingerprint(path)
        current[path] = fingerprint

        if manifest.get(path) != fingerprint:
            count = index_file(vectorstore, path)
            print(f"Indexed {path}: {count} chunks")

    for path in set(manifest) - set(current):
        remove_file(vectorstore, path)
        print(f"Removed {path} from index")

    _save_manifest(current)


# ---------------------------------
# Uploads
# ---------------------------------

def list_documents():
    return sorted(
        name
        for name in os.listdir(DOCUMENTS_PATH)
        if name.lower().endswith(".pdf")
    )


def save_and_index(vectorstore, filename, data):
    """Save an uploaded PDF into documents/ and index it right away."""

    name = os.path.basename(filename or "")

    if not name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    if not data.startswith(b"%PDF"):
        raise ValueError("This file is not a valid PDF.")

    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"PDF is larger than {MAX_UPLOAD_MB} MB.")

    path = os.path.join(DOCUMENTS_PATH, name)

    with open(path, "wb") as f:
        f.write(data)

    with _index_lock:
        try:
            chunks = index_file(vectorstore, path)
        except Exception:
            os.remove(path)
            raise ValueError("Could not read this PDF.")

        manifest = _load_manifest()
        manifest[path] = _fingerprint(path)
        _save_manifest(manifest)

    print(f"Uploaded and indexed {path}: {chunks} chunks")

    return {"file": name, "chunks": chunks}


def create_vector_store():
    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    os.makedirs(VECTORSTORE_PATH, exist_ok=True)

    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    sync_documents(vectorstore)

    return vectorstore