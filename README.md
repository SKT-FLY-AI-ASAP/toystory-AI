# Toystory
## AI

![image](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---
## Toystory 실행
### Installation
- Recommend : python >= 3.9.19, linux 20.04, CUDA >= 12.1(Install CUDA if avilable)
```
conda create --name toystory python==3.9.19
conda activate toystory

# Install the correct version of CUDA
conda install cuda -c nvidia/label/cuda-12.1.0

# Install other dependencies by
pip install -r requirements.txt
```
---
### Trouble Shooting
#### ERROR: Failed to build installable wheels for some pyproject.toml based projects (torchmcubes)
##### Make sure that your environment meets the requirements above
```
git clone https://github.com/tatsy/torchmcubes.git
cd torchmcubes
pip install .
```

#### 그래도 해결되지 않는다면 CUDA 설치 확인
##### 아래 경로에 파일이 있는지 확인하기
```
ls /usr/local/cuda/lib64/libnvToolsExt.so
```
##### 현재 사용 중인 가상 환경의 CUDA 경로에 존재하는지 확인
```
ls /home/asap/anaconda3/envs/test3/lib/libnvToolsExt.so
```
##### CMAKE에 라이브러리 경로 제공
```
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/usr/local/cuda/lib64
```
##### 또는 현재 환경에 맞게 CUDA 경로를 지정하기
```
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:/home/asap/anaconda3/envs/test3/lib
```
#### 그 다음에 다시 torchmcubes 설치 시도
```
pip install .
```
