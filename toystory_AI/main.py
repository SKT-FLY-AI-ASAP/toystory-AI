import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from gradio_app import process_and_generate  # Assuming the previous code is saved in gradio_app.py
import shutil
import os
import requests
import logging

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
    stl_url: str = None
    glb_url: str = None
    combined_glb_url: str = None

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

def upload_file_to_server(file_path, server_url):
    try:
        with open(file_path, 'rb') as file:
            files = {'file': file}
            response = requests.post(server_url, files=files)
        
        if response.status_code == 200:
            logging.info(f"File {file_path} successfully uploaded to {server_url}")
            return True
        else:
            logging.error(f"Failed to upload {file_path} to {server_url}. Status code: {response.status_code}")
            logging.error(response.text)
            return False
    except Exception as e:
        logging.error(f"Error occurred while uploading file: {str(e)}")
        return False

# 모델 생성 엔드포인트
@app.post("/api/v1/model", response_model=GeneratedContent, status_code=201)
def generate_content(content_req: ContentRequest):
    try:
        # 기본 URL 설정
        server_url = "http://yourserver.com/upload"
        
        # 이미지 URL 또는 프롬프트를 사용하여 콘텐츠 생성
        if content_req.prompt:
            processed_image_path, _, stl_path, _, glb_path, combined_glb_path = process_and_generate(
                input_text=content_req.prompt, 
                title=f"{content_req.user_id}-{content_req.title}"
            )
        elif content_req.image_url:
            processed_image_path, _, stl_path, _, glb_path, combined_glb_path = process_and_generate(
                input_s3_url=content_req.image_url, 
                title=f"{content_req.user_id}-{content_req.title}"
            )
        else:
            raise HTTPException(status_code=400, detail="Either 'prompt' or 'image_url' must be provided in the request")
        
        # 파일을 백엔드로 업로드
        upload_success = all([
            upload_file_to_server(processed_image_path, server_url),
            upload_file_to_server(stl_path, server_url),
            upload_file_to_server(glb_path, server_url),
            upload_file_to_server(combined_glb_path, server_url)
        ])

        if not upload_success:
            raise HTTPException(status_code=500, detail="Failed to upload files to server")

        # URL 생성
        base_url = "http://yourserver.com/files/"
        return GeneratedContent(
            png_url=f"{base_url}{os.path.basename(processed_image_path)}",
            stl_url=f"{base_url}{os.path.basename(stl_path)}",
            glb_url=f"{base_url}{os.path.basename(glb_path)}",
            combined_glb_url=f"{base_url}{os.path.basename(combined_glb_path)}"
        )
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
