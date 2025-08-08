**app.py**
```python
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import os
import openai

app = FastAPI()

class SummarizeRequest(BaseModel):
    url: str
    options: dict

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/summarize.txt")
async def summarize(req: SummarizeRequest, request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token != os.getenv("YTSUM_API_TOKEN"):
        raise HTTPException(status_code=401, detail="Unauthorized")

    prompt = f"Summarize the following YouTube video ({req.url}) in {req.options.get('length','medium')} detail. Language: {req.options.get('language','en')}."
    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        response = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```
