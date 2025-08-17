
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from openai import OpenAI

# --- config / client ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    # Do NOT put your key in this file. Add it as an env var on Render.
    raise RuntimeError("Set OPENAI_API_KEY env var on your hosting platform.")

client = OpenAI()  # uses env var
app = FastAPI(title="Brain Library (Tiny RAG Server)")

# ---------- homepage (nice to have) ----------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>Brain Library</title>
        <style>
          body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:720px;margin:40px auto;padding:0 16px}
          input,button{font:inherit;padding:8px}
          label{display:block;margin:8px 0 4px}
          .card{padding:16px;border:1px solid #ddd;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,.04)}
          .muted{color:#666}
          pre{white-space:pre-wrap;background:#f7f7f7;padding:12px;border-radius:8px}
          code{background:#f2f2f2;padding:2px 4px;border-radius:4px}
        </style>
      </head>
      <body>
        <h1>Brain Library ✅</h1>
        <p class="muted">Your server is running. Use the helpers below to test without curl/Postman.</p>

        <div class="card" style="margin:16px 0;">
          <h3>1) Create a Vector Store</h3>
          <label>Name</label>
          <input id="vsName" value="practice_kb"/>
          <button onclick="createVS()">Create</button>
          <pre id="vsOut"></pre>
        </div>

        <div class="card" style="margin:16px 0;">
          <h3>2) Upload File to Vector Store</h3>
          <label>Vector Store ID</label>
          <input id="vsIdForUpload" placeholder="vs_..."/>
          <label>File</label>
          <input id="file" type="file"/>
          <button onclick="upload()">Upload</button>
          <pre id="upOut"></pre>
        </div>

        <div class="card" style="margin:16px 0;">
          <h3>3) Ask a Question</h3>
          <label>Vector Store ID</label>
          <input id="vsId" placeholder="vs_..."/>
          <label>Question</label>
          <input id="q" style="width:100%" placeholder="What are the key takeaways?"/>
          <button onclick="ask()">Ask</button>
          <pre id="askOut"></pre>
        </div>

        <div class="card" style="margin:16px 0;">
          <h3>Status (what files are attached?)</h3>
          <label>Vector Store ID</label>
          <input id="vsStatus" placeholder="vs_..."/>
          <button onclick="status()">Check</button>
          <pre id="stOut"></pre>
        </div>

        <script>
          async function createVS(){
            const fd = new FormData();
            fd.append('name', document.getElementById('vsName').value || 'practice_kb');
            const r = await fetch('/vector-stores', { method:'POST', body: fd });
            document.getElementById('vsOut').textContent = await r.text();
          }
          async function upload(){
            const vs = document.getElementById('vsIdForUpload').value;
            const f = document.getElementById('file').files[0];
            if(!vs || !f){ alert('Need vector store ID and a file'); return; }
            const fd = new FormData();
            fd.append('vector_store_id', vs);
            fd.append('f', f);
            const r = await fetch('/files', { method:'POST', body: fd });
            document.getElementById('upOut').textContent = await r.text();
          }
          async function ask(){
            const vs = document.getElementById('vsId').value;
            const q  = document.getElementById('q').value;
            if(!vs || !q){ alert('Need vector store ID and question'); return; }
            const fd = new FormData();
            fd.append('vector_store_id', vs);
            fd.append('question', q);
            const r = await fetch('/ask', { method:'POST', body: fd });
            document.getElementById('askOut').textContent = await r.text();
          }
          async function status(){
            const vs = document.getElementById('vsStatus').value;
            const fd = new FormData();
            fd.append('vector_store_id', vs);
            const r = await fetch('/status', { method:'POST', body: fd });
            document.getElementById('stOut').textContent = await r.text();
          }
        </script>

        <p class="muted">Endpoints: <code>POST /vector-stores</code>, <code>POST /files</code>, <code>POST /ask</code>, <code>POST /status</code></p>
      </body>
    </html>
    """

# ---------- create vector store ----------
@app.post("/vector-stores")
def create_vector_store(name: str = Form("my_knowledge_base")):
    try:
        vs = client.vector_stores.create(name=name)
        return {"vector_store_id": vs.id, "name": name}
    except Exception as e:
        raise HTTPException(500, str(e))

# ---------- upload file & attach to store ----------
@app.post("/files")
async def add_file(
    vector_store_id: str = Form(...),
    f: UploadFile = File(...)
):
    try:
        # 1) upload file to Files API
        content = await f.read()
        uploaded = client.files.create(
            file=(f.filename, content),
            purpose="assistants"
        )
        # 2) attach to vector store
        client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=uploaded.id
        )
        return {"file_id": uploaded.id, "filename": f.filename, "vector_store_id": vector_store_id}
    except Exception as e:
        raise HTTPException(500, str(e))

# ---------- ask a question against the store ----------
@app.post("/ask")
def ask(
    vector_store_id: str = Form(...),
    question: str = Form(...)
):
    try:
        resp = client.responses.create(
            model="gpt-4.1",
            input=question,
            tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
            # To also get raw citations back, uncomment:
            # include=["file_search_call.results"]
        )

        # Extract assistant text from the structured response
        answer_parts = []
        for item in (resp.output or []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        answer_parts.append(c.get("text", ""))

        answer = "\n".join(answer_parts).strip() or "(No answer text returned.)"
        return JSONResponse({"answer": answer}, status_code=200)
    except Exception as e:
        raise HTTPException(500, str(e))

# ---------- list files + indexing status ----------
@app.post("/status")
def status(vector_store_id: str = Form(...)):
    try:
        vs_files = client.vector_stores.files.list(vector_store_id=vector_store_id)
        return {
            "count": len(vs_files.data),
            "files": [{"id": f.id, "status": f.status, "last_error": getattr(f, "last_error", None)} for f in vs_files.data],
        }
    except Exception as e:
        raise HTTPException(500, str(e))
