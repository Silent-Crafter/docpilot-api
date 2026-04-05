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
    def __init__(self):
        with open("labels/new.json", "r") as f:
            self.mappings = json.loads(f.read())
        self.lm = configure_llm(model=Config.ollama_model, base_url=Config.ollama_url, cache=False)
        self.__index, self.__image_index = self.__get_index()
        self.__rag = MultiHopRAG(self.__index, self.__image_index, num_passages=10)

    def generate(self, prompt):
        resps = self.__rag.forward(question=prompt, stream=True)
        actual_resp = {}
        final_resp = {}
        accumulated = ""
        for resp in resps:
            if resp["type"] == "answer_with_images":
                actual_resp = resp

                # print(f"\n\033[0;31m [DEBUG] Answer with images: \033[0m \n{actual_resp['content']}\n\n")
                with open('example.html', 'w') as f:
                    f.write("""
                    <html>
                    <head>
                    <style>
                    img { display: block; width: 200px; }
                    </style>
                    </head>
                    <body>
                    """ + actual_resp['content'] + """
                    </body>
                    </html>
                    """)
                yield resp

            elif resp["type"] == "answer":
                final_resp = resp
                yield resp

            elif resp["type"] == "streaming_answer":
                accumulated += resp["content"]
                yield accumulated

            elif resp["type"] == "files":
                print(f"\n\033[0;31m[+] Using files:\033[0m {resp['content']}\n")
                yield resp

            elif resp["type"] == "query":
                print(f"\n\033[0;31m[+] Searching for:\033[0m {resp['content']}\n")
                yield resp


    def add_file(self, file: str):
        self.__rag.add_new_document(file)

    def __get_index(self):
        text_docs, image_docs, mapping = ChatBot.__load_docs()

        with open('./labels/new.json', 'w') as f:
            f.write(json.dumps(mapping))

        return [get_vector_store_index(
                    documents=text_docs,
                    uri=Config.PG_CONNECTION_URI,
                    embeddings_table=Config.embed_table,
                    embed_model=Config.embed_model,
                    reindex=True
                ),
                get_vector_store_index(
                    documents=image_docs,
                    uri=Config.PG_CONNECTION_URI,
                    embeddings_table='data_images',
                    embed_model=Config.embed_model,
                    reindex=True
                )]

    @staticmethod
    def __load_docs():
        return load_docs(
            doc_dir=Config.document_dir,
            uri=Config.PG_CONNECTION_URI,
            embedding_table=Config.embed_table,
            reindex=True
        )

