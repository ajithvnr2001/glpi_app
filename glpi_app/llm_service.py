import os
# from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from unstructured.partition.html import partition_html
from typing import List, Dict
from langchain_ibm import WatsonxLLM
import requests
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma

class IBMEmebeddings(Embeddings):
    def __init__(self):
        self.watsonx_api_key = os.environ.get("WATSONX_API_KEY")
        self.project_id = os.environ.get("WATSONX_PROJECT_ID")
        self.url = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        if not all([self.watsonx_api_key, self.project_id]):
            raise ValueError("Watsonx.ai API key and project ID must be set in environment variables.")
        self.endpoint_url=f"{self.url}/feature_store/v1/models/sentence-transformers-all-minilm-l6-v2/versions/21/inference"
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.watsonx_api_key}",
        }
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        embeddings = self._get_embeddings([text])
        return embeddings[0]

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
            embeddings = []
            data = [{"input_text": text} for text in texts]
            response = requests.post(self.endpoint_url, headers=self.headers, json={"input_data": data})
            response.raise_for_status()
            results = response.json()["results"]
            for result in results:
                embeddings.append(result["values"])
            return embeddings

class LLMService:
    def __init__(self, model_name: str = "ibm/granite-13b-instruct-v2"):
        self.model_name = model_name
        self.watsonx_api_key = os.environ.get("WATSONX_API_KEY")
        self.project_id = os.environ.get("WATSONX_PROJECT_ID")
        self.url = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        if not all([self.watsonx_api_key, self.project_id]):
            raise ValueError(
                "Watsonx.ai API key and project ID must be set in environment variables."
            )

        self.llm = WatsonxLLM(
            model_id=self.model_name,
            url=self.url,
            apikey=self.watsonx_api_key,
            project_id=self.project_id,
            params={"decoding_method": "sample", "max_new_tokens": 200, "temperature": 0.7},
        )

    def get_embedding_function(self):
        return IBMEmebeddings()

    def create_vectorstore(self, chunks: List[Dict]):
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {key: value for key, value in chunk.items() if key != "text"}
            for chunk in chunks
        ]
        embeddings = self.get_embedding_function()
        db = Chroma.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
        return db

    def query_llm(self, db, query: str):
        qa = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=db.as_retriever()
        )
        result = qa.run(query)
        return result

    def rag_completion(self, documents, query):
        chunks = self.process_documents_to_chunks(documents)
        vector_store = self.create_vectorstore(chunks)
        return self.query_llm(vector_store, query)

    def process_documents_to_chunks(self, documents: List[Dict]) -> List[Dict]:
        chunks = []
        for doc in documents:
            if "content" in doc:
                content = doc["content"]
                elements = partition_html(text=content)
                for element in elements:
                    chunks.append(
                        {
                            "text": str(element),
                            "source_id": doc.get("id"),
                            "source_type": "glpi_ticket",
                        }
                    )
        return chunks

    def complete(self, prompt, context=None):
        if context:
            return self.llm(context + prompt)
        return self.llm(prompt)
