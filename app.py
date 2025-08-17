import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY env var")

client = OpenAI()
app = FastAPI(title="Tiny RAG Server")

@app.post("/vector-stores")
def create_vector_store(name: str = Form("my_knowledge_base")):
    try:
        vs = client.vector_stores.create(name=name)
        return {"vector_store_id": vs.id, "name": name}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/files")
async def add_file(
    vector_store_id: str = Form(...),
    f: UploadFile = File(...)
):
    try:
        # 1) upload file
        uploaded = client.files.create(file=(f.filename, await f.read()), purpose="assistants")
        # 2) attach to vector store
        client.vector_stores.files.create(vector_store_id=vector_store_id, file_id=uploaded.id)
        return {"file_id": uploaded.id, "filename": f.filename}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/ask")
def ask(
    vector_store_id: str = Form(...),
    question: str = Form(...)
):
    try:
        resp = client.responses.create(
            model="gpt-4.1",  # or your preferred model
            input=question,
            tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
            # include raw search results if you want them:
            # include=["file_search_call.results"]
        )
        # The Responses API returns a structured object; extract the assistant text:
        text_parts = []
        for item in resp.output or []:
            if item.get("type") == "message":
                for c in item["content"]:
                    if c.get("type") == "output_text":
                        text_parts.append(c["text"])
        return JSONResponse({"answer": "\n".join(text_parts)}, status_code=200)
    except Exception as e:
        raise HTTPException(500, str(e))
