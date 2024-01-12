import gradio
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    set_global_service_context,
)
from llama_index.llms.huggingface import HuggingFaceLLM
from sqlalchemy import make_url
# from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores import PGVectorStore
import os
import psycopg2
import torch
from transformers import BitsAndBytesConfig
from typing import Union

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
# from IPython.display import Markdown, display


HF_TOKEN = os.getenv("HF_TOKEN")
print(HF_TOKEN)
PG_CONN_STRING = os.getenv("PG_CONN_STRING")
print(PG_CONN_STRING)
DB_NAME = "edb_admin"
TABLE_NAME = "pgvector_test"
BNB_CONFIG = BitsAndBytesConfig(
    llm_int8_enable_fp32_cpu_offload=True,
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)


class LlamaChatApp:
    def __init__(self):
        # Initialize the model name
        self.model_name = "meta-llama/Llama-2-7b-chat-hf"

        # Initialize the embedding name
        self.embedding_name = "sentence-transformers/all-MiniLM-L6-v2"

    def init_llama_model(self):
        """
        Init llama model and embedding

        Returns:
            Union[HuggingFaceLLM, HuggingFaceEmbeddings]:
            return module and embedding
        """
        # Initialize the Llama model
        print("Initialize the Llama model")
        llm = HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=2048,
            generate_kwargs={"temperature": 0.0, "do_sample": False},
            tokenizer_name=self.model_name,
            model_name=self.model_name,
            device_map="auto",
            model_kwargs={
                "torch_dtype": torch.float16,
                "load_in_8bit": True,
                "use_auth_token": HF_TOKEN,
                "use_safetensors": True,
                "quantization_config": BNB_CONFIG
            }
        )
        # Initialize the embeddings
        print("Initialize the embeddings")
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_name,
            # to run model on CPU
            model_kwargs={'device': 'cpu'}
        )
        # Create the service context
        print("Create the service context")
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embeddings
        )
        # Set the global service context
        print("Set the global service context")
        set_global_service_context(service_context)

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
            embed_dim=384,  # embedding dimension
        )
        # Create a StorageContext object with the vector store and return it
        return StorageContext.from_defaults(vector_store=vector_store)

    def load_document(self, file_name: str) -> VectorStoreIndex:
        """
        Load a document and save it to the pg database.
        Args:
            file_name (str): The name of the file to retrieve.
        Returns:
             VectorStoreIndex: The index of the vector store.
        """
        # get file extension
        file_extension = file_name.split(".")[-1]
        # Check if file extension is supported
        if file_extension not in ["pdf", "docx", "csv"]:
            print(
              f"Cannot process {file_extension}. "
              "Only 'pdf', 'docx' and 'csv' are supported"
            )
            return None

        # Read the contents of the directory
        directory = os.path.dirname(file_name)
        documents = SimpleDirectoryReader(f"{directory}/").load_data()

        try:
            # Initialize the Llama model and embeddings
            # llm, embeddings = self.init_llama_model()
            # # Create the service context
            # service_context = ServiceContext.from_defaults(
            #     llm=llm,
            #     embed_model=embeddings
            # )
            # # Set the global service context
            # set_global_service_context(service_context)
            # Initialize the PostgreSQL vector storage
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

    def get_answer(
        self,
        fileobj: gradio.File,
        search_query: str
    ) -> Union[str, str]:
        """
        Get answers to user questions from Llama2 using API.

        Args:
            fileobj (gradio.File): Uploaded file.
            search_query (str): User question.
        Returns:
            answer (str): answer from the LLM
            sources (list): answer source from document
        """
        # Initialize default values for answer and sources
        answer, sources = 'could not generate an answer', ''
        # Load document from file
        print(f"loading document from {fileobj.name}")
        db = self.load_document(file_name=fileobj.name)
        if db:
            # Create query engine from loaded document
            query_engine = db.as_query_engine()
            try:
                # Query Llama2 API with the user question
                llm_response = query_engine.query(search_query)
                # Get the response from Llama2 API
                answer = llm_response.response
                # Get the formatted sources from the document
                sources = llm_response.get_formatted_sources()
            except Exception as error:
                print(f"Error while generating answer: {str(error)}")
        # Return the answer and sources
        return answer, sources


def gradio_interface(
    inputs: list = [
                    gradio.File(
                      label="Input file",
                      file_types=[".pdf", ".csv", ".docx"]
                    ),
                    gradio.Textbox(
                      label="your input",
                      lines=3,
                      placeholder="Your search query ..."
                    )
                  ],
    outputs: list = [
                    gradio.Textbox(
                      label="response",
                      lines=6,
                      placeholder="response returned from llama2 ...."
                    ),
                    gradio.Textbox(
                      label="response source",
                      lines=6,
                      placeholder="source to response ..."
                    )
                  ]
  ):
    """
    Render a Gradio interface.

    Args:
        inputs (list): List of input components.
        outputs (list): List of output components.
    """
    # Initialize LlamaChatApp object
    chat_ob = LlamaChatApp()
    try:
        # Initialize the Llama model and embeddings
        chat_ob.init_llama_model()
    except Exception as error:
        print(f"Exception while handling error {str(error)}")
        return None
    # Create a Gradio Interface
    demo = gradio.Interface(
        fn=chat_ob.get_answer,
        inputs=inputs,
        outputs=outputs
    )
    # Launch the interface with sharing enabled
    demo.launch(share=True)


if __name__ == '__main__':
    gradio_interface()
