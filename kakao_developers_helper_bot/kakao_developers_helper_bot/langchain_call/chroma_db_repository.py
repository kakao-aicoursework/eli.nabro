import os

from typing import List
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever


class ChromaDbRepository:
    _collection_name: str
    _persist_dir: str
    _db: Chroma = None
    _retriever: VectorStoreRetriever = None
    _data_dir: str

    def __init__(self, persist_dir: str, data_dir: str) -> None:
        self._collection_name = "kakao_bot"
        self._persist_dir = persist_dir
        self._data_dir = data_dir

    @staticmethod
    def _get_text(file_path: str) -> List[Document]:
        loader = TextLoader
        documents = loader(file_path).load()

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        return text_splitter.split_documents(documents)

    def push_texts(self):
        for root, dirs, files in os.walk(self._data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    text = self._get_text(file_path)
                    Chroma.from_documents(text,
                                          OpenAIEmbeddings(),
                                          persist_directory=self._persist_dir,
                                          collection_name=self._collection_name, )
                    print("SUCCESS: ", file_path)
                except Exception as e:
                    print("FAILED: ", file_path + f"by({e})")

    def query_db(self, query: str, use_retriever: bool = False) -> list[str]:
        if self._db is None:
            self._db = Chroma(
                persist_directory=self._persist_dir,
                embedding_function=OpenAIEmbeddings(),
                collection_name=self._collection_name,
            )
            self._retriever = self._db.as_retriever()

        if use_retriever:
            docs = self._retriever.get_relevant_documents(query)
        else:
            docs = self._db.similarity_search(query)

        str_docs = [doc.page_content for doc in docs]
        return str_docs
