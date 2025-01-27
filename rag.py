import re
import base64
import json

from docpilot.dspyclasses import MultiHopRAG, configure_llm
from docpilot.utils.llama_utils import load_docs, get_vector_store_index

from config import Config

class ChatBot:
    def __init__(self):
        self.lm = configure_llm(model=Config.ollama_model, base_url=Config.ollama_url, cache=False)
        self.__index = ChatBot.__get_index()
        self.__rag = MultiHopRAG(self.__index, num_passages=10, optimized_rag=Config.optimized_rag, place_images=False)
        with open("labels/new.json", "r") as f:
            self.mappings = json.dumps(f.read())

    def generate(self, prompt):
        resps = self.__rag.forward(question=prompt, mapping=self.mappings)
        actual_resp = {}
        final_resp = {}
        for resp in resps:
            if resp["type"] == "answer":
                actual_resp = resp
            print(resp['content'])
            final_resp = resp


        # imgs = re.findall(r"<img.*?src=\"(img-[0-9]{20}-\d+\..*?)\".*?/>", final_resp['content'])
        imgs = re.findall(r"<img.*?src=\"(.*?)\".*?/>", final_resp['content'])
        b64_imgs = []
        for img in imgs:
            try:
                ext = img.split(".")[-1]
                file = open('../docpilot/out_images/'+img, 'rb')
                b64_imgs.append({"image": img, "b64": f"data:image/{ext};base64, "+base64.b64encode(file.read()).decode('utf-8')})
                file.close()
            except FileNotFoundError:
                b64_imgs.append({"image": img, "b64": " "})

        print(actual_resp)
        print(len(final_resp['content']))
        for item in b64_imgs:
            img = item['image']
            b64_img = item['b64']
            final_resp['content'] = final_resp['content'].replace(img, b64_img)

        print(len(final_resp['content']))
        return final_resp

    @staticmethod
    def __get_index():
        return get_vector_store_index(
            documents=ChatBot.__load_docs(),
            uri=Config.PG_CONNECTION_URI,
            embeddings_table=Config.embed_table,
            embed_model=Config.embed_model
        )

    @staticmethod
    def __load_docs():
        return load_docs(
            doc_dir=Config.document_dir,
            uri=Config.PG_CONNECTION_URI,
            embedding_table=Config.embed_table
        )
