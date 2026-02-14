import re
import base64
import json

from docpilot.dspyclasses import MultiHopRAG, configure_llm
from docpilot.utils.llama_utils import load_docs, get_vector_store_index
from docpilot.utils.image_utils import mappings_to_llamaindex_document

from config import Config

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

            elif resp["type"] == "files":
                print(f"\n\033[0;31m[+] Using files:\033[0m {resp['content']}\n")
                yield resp

            elif resp["type"] == "query":
                print(f"\n\033[0;31m[+] Searching for:\033[0m {resp['content']}\n")
                yield resp


        # imgs = re.findall(r"<img.*?src=\"(img-[0-9]{20}-\d+\..*?)\".*?/>", final_resp['content'])
        # imgs = re.findall(r"<img.*?src=\"(.*?)\".*?/>", final_resp['content'])
        # b64_imgs = []
        # for img in imgs:
        #     try:
        #         ext = img.split(".")[-1]
        #         file = open('../docpilot/out_images/'+img, 'rb')
        #         b64_imgs.append({"image": img, "b64": f"data:image/{ext};base64, "+base64.b64encode(file.read()).decode('utf-8')})
        #         file.close()
        #     except FileNotFoundError:
        #         b64_imgs.append({"image": img, "b64": " "})
        # print(len(final_resp['content']))
        # for item in b64_imgs:
        #     img = item['image']
        #     b64_img = item['b64']
        #     final_resp['content'] = final_resp['content'].replace(img, b64_img)

        # print(len(final_resp['content']))
        # return final_resp

    def __get_index(self):
        images = mappings_to_llamaindex_document(self.mappings, 'out_images')
        return [get_vector_store_index(
                    documents=ChatBot.__load_docs(), 
                    uri=Config.PG_CONNECTION_URI,
                    embeddings_table=Config.embed_table,
                    embed_model=Config.embed_model
                ),
                get_vector_store_index(
                    documents=images,
                    uri=Config.PG_CONNECTION_URI,
                    embeddings_table='data_images',
                    embed_model=Config.embed_model
                )]

    @staticmethod
    def __load_docs():
        return load_docs(
            doc_dir=Config.document_dir,
            uri=Config.PG_CONNECTION_URI,
            embedding_table=Config.embed_table
        )
