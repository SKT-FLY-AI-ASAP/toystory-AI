# 1. Base image: CUDA 12.1과 Ubuntu 20.04를 기반으로 한 이미지 사용
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 2. 기본 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    git \
    build-essential \
    zlib1g-dev \
    libssl-dev \
    libncurses-dev \
    libffi-dev \
    libsqlite3-dev \
    libreadline-dev \
    libbz2-dev \
    liblzma-dev \
    cmake \
    g++ \
    libgl1-mesa-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Python 3.9.19 다운로드 및 설치
RUN wget https://www.python.org/ftp/python/3.9.19/Python-3.9.19.tgz && \
    tar -xf Python-3.9.19.tgz && \
    cd Python-3.9.19 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && rm -rf Python-3.9.19 Python-3.9.19.tgz

# 4. Python 3.9.19을 기본 python3로 설정
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.9 1

# 5. pip, setuptools, wheel 업그레이드
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && \
    pip install --upgrade pip setuptools wheel && \
    rm get-pip.py

# 6. CUDA 지원 PyTorch 설치
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# 7. CMAKE_PREFIX_PATH 환경 변수 설정
ENV CMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"

RUN git clone https://github.com/tatsy/torchmcubes.git && \
    cd torchmcubes && \
    pip install .

# 6. 작업 디렉토리 설정
WORKDIR /app

# 7. 필요한 Python 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 8. 소스 코드 복사
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
