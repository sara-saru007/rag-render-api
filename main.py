from fastapi import FastAPI
from pydantic import BaseModel
from utils.rag import load_index, retrieve_context
import openai
import os

app = FastAPI(
    title="RAG Email Reply API",
    description="Auto-generates investor replies using startup-specific documents.",
    version="1.0.0"
)

# Load OpenAI key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.get("/")
def root():
    return {"message": "API is live!"}

class Query(BaseModel):
    startup_id: str
    email_text: str
    manual_prompt: str

@app.post("/generate_reply")
def generate_reply(q: Query):
    index_path = f"startup_configs/{q.startup_id}/index"
    index = load_index(index_path)
    docs = retrieve_context(index, q.email_text)
    context = "\n\n".join([d.page_content for d in docs])

    system_prompt = f"{q.manual_prompt}\n\nUse this context:\n{context}"

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q.email_text}
        ]
    )

    return {"reply": response["choices"][0]["message"]["content"]}

# 👇 Enable local and Render deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
