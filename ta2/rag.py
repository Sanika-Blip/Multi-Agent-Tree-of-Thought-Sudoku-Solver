"""
rag.py  (U3-3 RAG)
------
Chroma vector store (persist_directory=data/chroma) over corpus/*.md,
embedded with Ollama nomic-embed-text. Each '## ' section is one chunk
(10 chunks). Two uses:
  - hint_for_cell(): the Proposer retrieves solving guidance before it
    proposes candidates.
  - answer(): grounded Q&A that must reply "I don't know" when the
    retrieved chunks do not contain the answer.
"""

import glob
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, "corpus")
PERSIST_DIR = os.path.join(HERE, "data", "chroma")
COLLECTION = "sudoku_notes"

GROUNDED_PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: I don't know.

Context:
{context}

Question: {question}
Answer in at most 3 sentences:"""


def load_chunks():
    docs = []
    for path in sorted(glob.glob(os.path.join(CORPUS_DIR, "*.md"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for i, section in enumerate(s for s in text.split("## ") if s.strip()):
            title, _, body = section.partition("\n")
            docs.append(Document(page_content=f"{title.strip()}\n{body.strip()}",
                                 metadata={"chunk_id": i, "title": title.strip(),
                                           "source": os.path.basename(path)}))
    return docs


def get_vectorstore(rebuild: bool = False) -> Chroma:
    vs = Chroma(collection_name=COLLECTION,
                embedding_function=OllamaEmbeddings(model="nomic-embed-text"),
                persist_directory=PERSIST_DIR)
    if rebuild:
        vs.reset_collection()
    if rebuild or not vs.get(limit=1)["ids"]:
        chunks = load_chunks()
        vs.add_documents(chunks, ids=[f"chunk-{d.metadata['chunk_id']}" for d in chunks])
    return vs


def retrieve(vs: Chroma, query: str, k: int = 3):
    return vs.similarity_search(query, k=k)


def answer(vs: Chroma, question: str, k: int = 3, model: str = "qwen2.5:3b"):
    docs = retrieve(vs, question, k)
    context = "\n\n".join(f"[{d.metadata['title']}] {d.page_content}" for d in docs)
    llm = ChatOllama(model=model, temperature=0)
    reply = llm.invoke(GROUNDED_PROMPT.format(context=context, question=question)).content
    return reply.strip(), [d.metadata["title"] for d in docs]


def hint_for_cell(vs: Chroma, k: int = 1) -> str:
    docs = retrieve(vs, "how to propose candidate digits for one empty cell", k)
    return " ".join(d.page_content.split("\n", 1)[-1] for d in docs)
