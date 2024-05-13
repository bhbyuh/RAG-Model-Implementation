from langchain_elasticsearch import ElasticsearchStore
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
from config import elastic_url
load_dotenv()

class vectorStore:
    def __init__(self, is_load, index_name):
        self.is_load = is_load
        self.index_name = index_name
        self.qdrant_url = os.getenv("qdrant_url")
        self.qdrant_api_key = os.getenv("qdrant_api")

    def create_vectorstore_qdrant(self, embedding_model, chunks ):
            if not self.is_load:
                vector_db = Qdrant.from_documents(
                    documents = chunks,
                    embedding = embedding_model,
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    collection_name=self.index_name,
                )
                # db = ElasticsearchStore.from_documents(
                #     chunks,
                #     embedding_model,
                #     es_url=elastic_url,
                #     index_name=self.index_name,
                #     )
                #return db
            else:
                qdrant_client = QdrantClient(
                    url=self.qdrant_url, 
                    api_key=self.qdrant_api_key,
                )
                vector_db= Qdrant(qdrant_client, self.index_name, embedding_model)
            return vector_db
