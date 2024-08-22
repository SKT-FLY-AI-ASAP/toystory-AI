import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from gradio_app import process_and_generate

app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

class ContentRequest(BaseModel):
    user_id: int = None
    title: str = None
    image_url: str = None
    prompt: str = None

class GeneratedContent(BaseModel):
    stl_url: str = None
    glb_url: str = None

@app.post("/api/v1/model", response_model=GeneratedContent, status_code=201)
def generate_content(content_req: ContentRequest = None):
    if not content_req.prompt:
        processed_image, s3_urls1, s3_urls2 = process_and_generate(
            input_s3_url=content_req.image_url, 
            title=f"{content_req.user_id}-{content_req.title}"
        )
    else:
        processed_image, s3_urls1, s3_urls2 = process_and_generate(
            input_text=content_req.prompt, 
            title=f"{content_req.user_id}-{content_req.title}"
        )

    return GeneratedContent(stl_url=s3_urls2, glb_url=s3_urls1)