FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    curl \
    git \
    python3-pip \
    python3-dev \
    libsm6 \
    libegl1-mesa \
    libxrender1 \
    libxext6 \
    libxi6 \
    libgl1-mesa-glx
    # git
    # python3-dev \
    # build-essential \
    # cmake \
    # g++ \
    # && apt-get clean && rm -rf /var/lib/apt/lists/*

# libssl1.1 패키지 다운로드 및 설치
RUN wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.0g-2ubuntu4_amd64.deb && \
    dpkg -i libssl1.1_1.1.0g-2ubuntu4_amd64.deb && \
    apt-get install -f -y

ENV DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1

WORKDIR /workspace

# RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]