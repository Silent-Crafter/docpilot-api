import logging
import re
import base64
import json

from docpilot.dspyclasses import MultiHopRAG, configure_llm
from docpilot.utils.llama_utils import load_docs, get_vector_store_index
from docpilot.utils.logger import setup_logging
from config import Config

setup_logging(level=logging.DEBUG)

class ChatBot:
    lm = None
    __index = None
    __image_index = None
    def __init__(self):
        with open("labels/new.json", "r") as f:
            self.mappings = json.loads(f.read())

        if ChatBot.lm is None:
            ChatBot.lm = configure_llm(model=Config.ollama_model, cache=False, base_url=Config.ollama_url)

        if ChatBot.__index is None:
            ChatBot.__index, ChatBot.__image_index = self.__get_index()

        self.__rag = MultiHopRAG(ChatBot.__index, ChatBot.__image_index, num_passages=10)

    def generate(self, prompt):
        resps = self.__rag.forward(question=prompt, stream=True)
        actual_resp = {}
        final_resp = {}
        accumulated = ""
        for resp in resps:
            if resp["type"] == "streaming_answer":
                accumulated += resp["content"]
                resp["content"] = accumulated
                yield resp

            elif resp["type"] == "files":
                print(f"\n\033[0;31m[+] Using files:\033[0m {resp['content']}\n")
                yield resp

            elif resp["type"] == "query":
                print(f"\n\033[0;31m[+] Searching for:\033[0m {resp['content']}\n")
                yield resp

            else:
                yield resp

    def add_file(self, file: str):
        new_mappings = self.__rag.add_new_document(file)
        self.__update_mapping(new_mappings)

    def delete_file(self, filepath: str):
        """Delete a document, its embeddings, and associated images."""
        import os
        filename = os.path.basename(filepath)

        # Delete from vector DB + image files on disk + rebuild indexes
        self.__rag.delete_document(
            filename=filename,
            uri=Config.PG_CONNECTION_URI,
            text_table=Config.embed_table,
            image_table='data_images',
        )

        # Delete the document file itself
        if os.path.isfile(filepath):
            os.remove(filepath)

    def get_history(self):
        return self.__rag.message_history_with_images

    def __update_mapping(self, mapping):
        if not mapping:
            return

        og_mappings: dict = {}
        with open('./labels/new.json', 'r') as f:
            og_mappings: dict = json.loads(f.read())

        og_mappings.update(mapping)

        with open('./labels/new.json', 'w') as f:
            f.write(json.dumps(og_mappings))


    def __get_index(self):
        text_docs, image_docs, mapping = ChatBot.__load_docs()

        self.__update_mapping(mapping)

        return [get_vector_store_index(
                    documents=text_docs,
                    uri=Config.PG_CONNECTION_URI,
                    embeddings_table=Config.embed_table,
                    embed_model=Config.embed_model,
                ),
                get_vector_store_index(
                    documents=image_docs,
                    uri=Config.PG_CONNECTION_URI,
                    embeddings_table='data_images',
                    embed_model=Config.embed_model,
                )]

    @staticmethod
    def __load_docs():
        return load_docs(
            doc_dir=Config.document_dir,
            uri=Config.PG_CONNECTION_URI,
            embedding_table=Config.embed_table,
            use_vlm=True,
        )

