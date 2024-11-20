## Diffusion 기반 상품 맞춤형 광고 스토리보드 생성
### Pipeline Overview
![파이프라인](https://github.com/user-attachments/assets/62e42633-c9aa-4cb9-b7a4-d2616cc322e8)
- **사용자 입력**: 품목명과 강조하고 싶은 특징
- **광고 아이디어 생성 과정**
  - 광고 아이디어 생성에 Bllossom-8B 모델 활용
- **텍스트 전처리 과정**
  - Bllossom-8B에서 나온 아웃풋을 전처리 후 번역, 요약하는 과정을 거침
- **스토리보드 이미지 생성 과정**
  - 전처리 완료된 장면별 광고 아이디어 텍스트는 SDXL (Stable Diffusion XL)의 인풋 프롬프트로 입력됨
  - SDXL (Stable Diffusion XL)은 해당 프롬프트를 기반으로 장면 별 스토리보드 이미지를 생성
- **출력**: 프롬프트 (장면 설명), 스토리보드 이미지(6장)가 합쳐져서 하나의 새로운 광고 스토리보드를 생성
## Models and Technologies
### 1. Bllossom-8B
- Llama3를 기반으로 한 한국어-영어 이중 언어 모델
- 한국어 문화와 언어를 고려하여 제작한 데이터를 활용해서 fine-tuning된 모델
- 프롬프트 튜닝 진행
  - LLM에게 역할을 부여하는 PROMPT 와 원하는 명령을 집어넣는 Instruction으로 나눠 프롬프트 튜닝
  - 이미지 생성 모델의 fine-tuning 데이터의 구조와 유사하게 텍스트를 구성하기 위해 상황, 인물 설명, 카메라 구도를 나누고 장면 수를 6개로 고정시킴
### 2. Prompt Preprocessing
- 텍스트 전처리 진행
- 번역: mBART50 모델 사용 → 한국어 문장을 영어로 번역
- 요약: BART 요약 모델 사용 → 번역 문장이 Image Generation 모델에서 처리 가능한 최대 토큰 수인 77토큰을 넘을 경우, 77토큰 이하로 문장을 요약
### 3. SDXL (Stable Diffusion XL)
- 다양한 프롬프트에 대해 고해상도 이미지를 생성하는 성능이 검증된 대표적인 오픈소스 이미지 생성 모델
- 이미지에 추가된 노이즈를 예측하는 Latent Diffusion 방식을 사용해 효율적이고 빠른 이미지 생성 가능
- LoRA Fine-Tuning
  - 스토리보드 이미지 생성에서 실사 이미지보다는 스케치나 웹툰 스타일의 이미지가 적합하다고 판단해 AI hub의 만화.웹툰 생성 데이터와 허깅페이스의 라인스케치 데이터셋 사용해 파인튜닝 진행
  - 이미지 스타일의 일관성 있는 생성을 위해 컬러 이미지들을 모두 흑백 이미지로 변환
  - image caption에 ‘a grayscale sketch of’ 를 추가해서 흑백 스케치 형태의 아웃풋이 나오도록 함
## Demo
### [Demo Video]
[![Demo video](https://img.youtube.com/vi/SSBxPj4Wj2o/0.jpg)](https://www.youtube.com/watch?v=SSBxPj4Wj2o)
### [Demo Images] 
![1](https://github.com/user-attachments/assets/c11aef2b-c235-4f7b-94c8-29ac085e5909)
![2](https://github.com/user-attachments/assets/836f59e9-e389-454a-983f-93379e0ab8b7)
![3](https://github.com/user-attachments/assets/44c5ce6d-3bda-438e-9fdf-ad96fe48e681)

## Project Environment
#### 1. 하드웨어 사양
- GPU: A100, L4, T4
#### 2. 소프트웨어 환경
- CUDA:
- Python:
#### 3. 필수 라이브러리
- requirements.txt에 명시
#### 4. 실행환경
- GPU 환경에서만 실행이 가능합니다.

## Dataset
## Checkpoints
## How to Run
### 1. 환경 설정
- Repository를 clone 합니다.
  ```bash
    git clone https://github.com/justpers/Sketch2Image-and-ImageRetrieval
- Checkpoints에서 가중치를 다운 받아 올바른 경로에 놓습니다.
- 필수 라이브러리를 설치합니다.
  ```bash
    pip install -r requirements.txt
### 2. 

## Hosted and Organized By & Team Members
- 주최/주관: 국민대학교 AI빅데이터융합경영학과 AI빅데이터 분석 학회 D&A,  AI빅데이터융합경영학과 인공지능 학회 X:AI, 소프트웨어학부 웹학술동아리 WINK
- 팀원: 총 4인 [권민지, 김차미(팀장), 노명진, 이지민]
