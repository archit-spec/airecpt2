import json
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="bc80cbd1-33ac-4712-b969-f2145ba40aef.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="DeOUSJcHud18mAZzgfVy2f0LsIXkO_Q5bOEH49ABQtlktxpRL5V_8g",
)

# Create the collection
# qdrant_client.create_collection(
#     collection_name="emergency_instructions2",
#     vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # Adjust size based on your model's output
# )

# Load the JSON data
with open('emergency_instructions.json', 'r') as f:
    data = json.load(f)

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# # Prepare the vectors and metadata
# points = []
# for idx, intent in enumerate(data['intents']):
#     tag = intent['tag']
#     responses = intent['responses']
#     for pattern in intent['patterns']:
#         vector = model.encode(pattern).tolist()
#         points.append(PointStruct(
#             id=idx,
#             vector=vector,
#             payload={
#                 'tag': tag,
#                 'responses': responses
#             }
#         ))

# # Upload the vectors to the Qdrant collection
# qdrant_client.upsert(
#     collection_name="emergency_instructions2",
#     points=points
# )

def encode_single(text):
    return model.encode(text)

def search(query, limit=1):
    query_vector = encode_single(query)
    search_result = qdrant_client.search(
        collection_name="emergency_instructions2",
        query_vector=query_vector.tolist(),
        limit=limit
    )
    return search_result

# Print all the keys in collections
collection_info = qdrant_client.get_collection("emergency_instructions2")
print(f"Number of vectors in the collection: {collection_info.vectors_count}")

print(qdrant_client.get_collections())
print(search("help i cutted my leg"))

a = search("help i cutted my leg")