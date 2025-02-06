import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq
import requests
import json

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("sample-movies")

# Load Sentence Transformer for embeddings
JINA_API_KEY = os.getenv("JINA_API_KEY")


# Generate embeddings
user_query = "How much did Star Wars bring in in box office?"

API_URL = "https://api.jina.ai/v1/embeddings"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {JINA_API_KEY}"
}

data = {
    "model": "jina-clip-v2",  # Specify the embedding model
    "dimensions" : 1024,
    "input": [user_query]
}

response = requests.post(API_URL, json=data, headers=headers)

# Print the embeddings
print(response.json())
query_embeddings = response.json()["data"][0]["embedding"]


# Query Pinecone index
result = index.query(
    vector=query_embeddings,
    top_k=5,
    include_values=False,
    include_metadata=True
)

print(result)

# Extract relevant details
parsed_movies = []
for match in result.get("matches", []):
    metadata = match.get("metadata", {})
    parsed_movies.append({
        "title": metadata.get("title", "Unknown Title"),
        "year": int(metadata.get("year", 0)),
        "summary": metadata.get("summary", "No summary available."),
        "genre": metadata.get("genre", "Unknown Genre"),
        "box_office": f"${int(metadata.get('box-office', 0)):,}",  # Formatting as currency
        "score": round(match.get("score", 0), 4)  # Round similarity score
    })

# Pretty-print extracted results
print(json.dumps(parsed_movies, indent=4))
context = json.dumps(parsed_movies, indent=4)

# Create system prompt
sys_prompt = f"""
Instructions:
- Be helpful and answer questions concisely. If you don't know the answer, say 'I don't know'
- Utilize the context provided for accurate and specific information.
- Incorporate your preexisting knowledge to enhance the depth and relevance of your response.
- Cite your sources
Context: {context}
"""

# Initialize Groq API Client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Generate response with Groq
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_query},
    ]
)

# Print the response
print(response.choices[0].message.content)
