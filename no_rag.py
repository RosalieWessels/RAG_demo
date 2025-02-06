from groq import Groq
import os

user_query = "How much did Barbie bring in in box office?"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Generate response with Groq
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role" : "system", "content" : "Answer questions. Admit if you are not exactly sure."},
        {"role": "user", "content": user_query},
    ]
)

# Print the response
print(response.choices[0].message.content)