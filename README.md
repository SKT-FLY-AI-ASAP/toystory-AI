# Toystory
## SKT FLY AI Challenger 5기

![image](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---
## Toystory 실행
### Installation
- Recommend : python >= 3.9.19, linux 20.04, CUDA >= 12.1(Install CUDA if avilable)
```
conda create --name toystory python = 3.9.19
conda activate toystory

# Install the correct version of CUDA
conda install cuda -c nvidia/label/cuda-12.1.0

# Install other dependencies by
pip install -r requirements.txt
```
---
### Trouble Shooting
> ERROR: Failed to build installable wheels for some pyproject.toml based projects (torchmcubes)
>> Make sure that your environment meets the requirements above
```
git clone https://github.com/tatsy/torchmcubes.git
cd torchmcubes
pip install .
```

> 그래도 해결되지 않는다면 CUDA 설치 확인
>> 아래 경로에 파일이 있는지 확인하기
```
ls /usr/local/cuda/lib64/libnvToolsExt.so
```
>> 현재 사용 중인 가상 환경의 CUDA 경로에 존재하는지 확인
```
ls /home/asap/anaconda3/envs/test3/lib/libnvToolsExt.so
```






