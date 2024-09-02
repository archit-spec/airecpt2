from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import json
import torch
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class VectorDB:
    def __init__(self):
        self.client = QdrantClient(
            
            url="bc80cbd1-33ac-4712-b969-f2145ba40aef.europe-west3-0.gcp.cloud.qdrant.io:6333", 
            port=6333,
            api_key="DeOUSJcHud18mAZzgfVy2f0LsIXkO_Q5bOEH49ABQtlktxpRL5V_8g"
        )
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the model and move it to the appropriate device
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model = self.model.to(self.device)
        
        self.collection_name = "emergency_instructions2"

    def initialize_collection(self):
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.model.get_sentence_embedding_dimension(), distance=models.Distance.COSINE)
        )

    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        points = []
        for i, intent in enumerate(data['intents']):
            tag = intent['tag']
            patterns = intent['patterns']
            responses = intent['responses']

            # Batch encode patterns for efficiency
            embeddings = self.encode_batch(patterns)

            for pattern, embedding in zip(patterns, embeddings):
                points.append(
                    models.PointStruct(
                        id=i,
                        vector=embedding.tolist(),
                        payload={
                            "tag": tag,
                            "pattern": pattern,
                            "responses": responses  # Ensure this matches the key in your JSON
                        }
                    )
                )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    @lru_cache(maxsize=1000)
    def encode_single(self, text):
        return self.model.encode(text, device=self.device)

    def encode_batch(self, texts):
        return [self.encode_single(text) for text in texts]

    def search(self, query, limit=1):
        logger.info(f"Searching for query: {query}")
        query_vector = self.encode_single(query)
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=limit
        )
        if search_result:
            logger.info(f"Search result found for query: {query}")
            top_result = search_result[0]
            return {
                "source": "vector_db",
                "tag": top_result.payload['tag'],
                "response": top_result.payload['responses'][0],  # Access the first response
                "score": top_result.score
            }
        logger.info(f"No search result found for query: {query}")
        return None

    def get_collections(self):
        return self.client.get_collections()
