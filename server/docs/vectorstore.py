import os
import time
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)


spec = ServerlessSpec(
    cloud="aws",
    region=PINECONE_ENV
)

existing_indexes = [
    i["name"]
    for i in pc.list_indexes()
]

if PINECONE_INDEX_NAME not in existing_indexes:

    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,   # all-MiniLM-L6-v2 => 384 dimensions
        metric="cosine",
        spec=spec,
    )

    while not pc.describe_index(
        PINECONE_INDEX_NAME
    ).status["ready"]:

        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)


async def load_vectorstore(
    uploaded_files,
    role: str,
    doc_id: str
):

    # Embedding model
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    for file in uploaded_files:

        try:

            # Save uploaded file
            save_path = Path(UPLOAD_DIR) / file.filename

            with open(save_path, "wb") as f:
                f.write(file.file.read())

            print(f"Saved file: {file.filename}")

            # Load PDF
            loader = PyPDFLoader(str(save_path))

            documents = loader.load()

            print(f"Loaded {len(documents)} pages")

            # Split documents
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(
                documents
            )

            print(f"Created {len(chunks)} chunks")

            # Extract text
            texts = [
                chunk.page_content
                for chunk in chunks
            ]

            # Generate IDs
            ids = [
                f"{doc_id}-{i}"
                for i in range(len(chunks))
            ]

            # Metadata
            metadata = [
                {
                    "text": chunk.page_content,
                    "source": file.filename,
                    "doc_id": doc_id,
                    "role": role,
                    "page": chunk.metadata.get(
                        "page",
                        0
                    ),
                }
                for chunk in chunks
            ]

            print("Generating embeddings...")

            embeddings = await asyncio.to_thread(
                embed_model.embed_documents,
                texts
            )

            print(
                f"Generated {len(embeddings)} embeddings"
            )

            # Create vectors
            vectors = list(
                zip(
                    ids,
                    embeddings,
                    metadata
                )
            )

            print("Uploading to Pinecone...")

            with tqdm(
                total=len(vectors),
                desc="Upserting to Pinecone"
            ) as progress:

                index.upsert(vectors=vectors)

                progress.update(len(vectors))

            print(
                f"Upload complete for {file.filename}"
            )

        except Exception as e:

            print(f"Error processing file: {e}")

            raise e

    return True