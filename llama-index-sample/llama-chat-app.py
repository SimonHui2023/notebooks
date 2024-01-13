import gradio
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from llama_index import (
    # SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    set_global_service_context,
)
from llama_index.llms.huggingface import HuggingFaceLLM
from sqlalchemy import make_url
# from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores import PGVectorStore
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index import download_loader

import os
import psycopg2
import torch

PG_CONN_STRING = os.getenv("PG_CONN_STRING")
print(PG_CONN_STRING)
DB_NAME = "edb_admin"
TABLE_NAME = "pgvector_sample"


class LlamaChatApp:
    def __init__(self):
        """
        Init llama model and embedding
        """
        # Initialize the model name
        self.model_name = "microsoft/phi-2"
        # Initialize the embedding name
        self.embedding_name = "BAAI/bge-small-en-v1.5"
        self.system_prompt = """
            You are a Q&A assistant.
            Your goal is to answer questions as accurately
            as possible based on the instructions and context provided.
        """
        # This will wrap the default prompts that are internal to llama-index
        self.query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

        self.url = "https://www.enterprisedb.com/docs/pdfs/biganimal/release/biganimal_vrelease_documentation.pdf"
        # Initialize the Llama model
        print("Initialize the Llama model")
        llm = HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.0, "do_sample": False},
            system_prompt=self.system_prompt,
            query_wrapper_prompt=self.query_wrapper_prompt,
            tokenizer_name=self.model_name,
            model_name=self.model_name,
            device_map="auto",
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "use_safetensors": True,
                "trust_remote_code": True
            }
        )
        # Initialize the embeddings
        print("Initialize the embeddings")
        embeddings = LangchainEmbedding(
            HuggingFaceEmbeddings(
                model_name=self.embedding_name,
                # to run model on GPU
                model_kwargs={'device': 'cuda'}
            )
        )
        # Create the service context
        print("Create the service context")
        service_context = ServiceContext.from_defaults(
            chunk_size=1024,
            llm=llm,
            embed_model=embeddings
        )
        # Set the global service context
        print("Set the global service context")
        set_global_service_context(service_context)
        # Get VectorStoreIndex
        self.index = self.load_document()

    def init_pg_vector(self) -> StorageContext:
        """
        Initialize the PostgreSQL vector store and
        return a StorageContext object.
        Returns:
            StorageContext: The initialized StorageContext object.
        """
        # Create a connection to the PostgreSQL database
        connection_string = PG_CONN_STRING
        conn = psycopg2.connect(connection_string)
        conn.autocommit = True
        # Create the vector extension if it doesn't exist
        with conn.cursor() as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Create a URL object from the connection string
        url = make_url(connection_string)
        print(f"{url.host}")
        # Create a PGVectorStore object with the specified parameters
        vector_store = PGVectorStore.from_params(
            database=DB_NAME,
            host=url.host,
            password=url.password,
            port=url.port,
            user=url.username,
            table_name=TABLE_NAME,
            embed_dim=1536,  # embedding dimension
        )
        # Create a StorageContext object with the vector store and return it
        return StorageContext.from_defaults(vector_store=vector_store)

    def load_document(self) -> VectorStoreIndex:
        """
        Load a document and save it to the pg database.
        Args:
            url (str): Remote download url
        Returns:
             VectorStoreIndex: The index of the vector store.
        """
        RemoteReader = download_loader("RemoteReader")

        loader = RemoteReader()
        documents = loader.load_data(url=self.url)

        try:
            storage_context = self.init_pg_vector()
            # Create the vector store index from the documents
            index = VectorStoreIndex.from_documents(
                documents=documents,
                # service_context=service_context,
                storage_context=storage_context,
                show_progress=True
            )
            return index
        except Exception as error:
            print(f"Exception while handling error {str(error)}")
            return None

    def predict(self, input, history):
        try:
            db = self.VectorStoreIndex
            if db:
                query_engine = db.as_query_engine()
                response = query_engine.query(input)
                return str(response)
        except Exception as error:
            print(f"Exception while handling error {str(error)}")
            return None


if __name__ == '__main__':
    chat_ob = LlamaChatApp()
    gradio.ChatInterface(chat_ob.predict).launch(share=True)
