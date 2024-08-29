import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from gradio_app import process_and_generate  # Assuming the previous code is saved in gradio_app.py
import shutil
import os
import logging
import httpx
from datetime import datetime

app = FastAPI()

# 로깅 설정
logging.basicConfig(level=logging.INFO)

class ContentRequest(BaseModel):
    user_id: int = None
    title: str = None
    image_url: str = None
    prompt: str = None

class GeneratedContent(BaseModel):
    png_url: str = None
    bg_png_url: str = None
    glb_url: str = None
    stl_url: str = None
    mp3_url: str = None

class S3Response(BaseModel):
    url: str = None

# 파일 업로드 엔드포인트
@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # 파일을 /tmp/ 경로에 저장
        file_path = f"/tmp/{file.filename}"

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logging.info(f"File saved at {file_path}")
        return {"file_path": file_path}

    except Exception as e:
        logging.error(f"Failed to save file: {str(e)}")
        raise HTTPException(status_code=500, detail="File upload failed")

async def add_to_s3(file: UploadFile, object_name: str):
    SERVER_URL = os.getenv("SERVER_BASE_URL") + "/api/v1/s3"
    files = {"file": file}
    metadata = {"object_name": object_name}

    async with httpx.AsyncClient() as cli:
        response = await cli.post(SERVER_URL, files=files, data=metadata)
        data = S3Response(**response.json())

        return data.url

# 모델 생성 엔드포인트
@app.post("/api/v1/model", response_model=GeneratedContent, status_code=201)
def generate_content(content_req: ContentRequest):
    try:
        # 이미지 URL 또는 프롬프트를 사용하여 콘텐츠 생성
        if content_req.prompt:
            png_path, bg_png_path, glb_path, stl_path, mp3_path = process_and_generate(
                input_text=content_req.prompt,
                title=f"{content_req.user_id}-{content_req.title}"
            )
        elif content_req.image_url:
            png_path, bg_png_path, glb_path, stl_path, mp3_path = process_and_generate(
                input_s3_url=content_req.image_url,
                title=f"{content_req.user_id}-{content_req.title}"
            )
        else:
            raise HTTPException(status_code=400, detail="Either 'prompt' or 'image_url' must be provided in the request")

        # S3 등록
        key_mapping = {
            png_path: "png_url",
            bg_png_path: "bg_png_url",
            glb_path: "glb_url",
            stl_path: "stl_url",
            mp3_path: "mp3_url"
        }
        res = {}
        for file_path in [png_path, bg_png_path, glb_path, stl_path, mp3_path]:
            with open(file_path, 'rb') as file:
                current_time = datetime.now().strftime('%Y%m%dT%H:%M:%S')
                url = add_to_s3(file=file, object_name=f'{current_time}.{os.path.basename(file_path)}')
                key = key_mapping.get(file_path, file_path)  # 기본적으로 file_path를 사용
                res[key] = url

        return JSONResponse(content=res)

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
