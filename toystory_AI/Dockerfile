FROM python:3.9.19

COPY . .

RUN apt-get update

RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]