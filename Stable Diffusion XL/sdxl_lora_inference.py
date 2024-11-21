import os
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, BartForConditionalGeneration, BartTokenizer
import torch
from diffusers import StableDiffusionXLPipeline

# Stable Diffusion 및 번역 모델 로드
model_path = "./sdxl-lora/" # pytorch_lora_weights.safetonsers 경로!
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipe.to("cuda")
pipe.load_lora_weights(model_path)

# 번역 모델
trans_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt", clean_up_tokenization_spaces=True)
trans_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt").to("cuda" if torch.cuda.is_available() else "cpu")

# 영어 요약 모델
device = "cuda" if torch.cuda.is_available() else "cpu"
summary_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
summary_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

# 번역 함수
def translate_prompt(prompt):
    inputs = trans_tokenizer(prompt, return_tensors="pt").to(trans_model.device)
    translated_tokens = trans_model.generate(**inputs)
    translated_prompt = trans_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_prompt

# 영어 요약 함수 (토큰 수 50이상 70이하로 설정)
def summarize_english_text(text, max_length=70, min_length=50):
    inputs = summary_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    summary_ids = summary_model.generate(
        inputs["input_ids"], 
        max_length=max_length, 
        min_length=min_length,  
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    summary = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# 번역 및 요약 통합 함수
def translate_and_summarize(prompt, max_length=70, min_length=50):
    # Step 1: 한국어 → 영어 번역
    translated_text = translate_prompt(prompt)
    print(f"번역된 텍스트: {translated_text}")
    
    # 번역된 텍스트의 토큰 수 계산
    token_count = len(summary_tokenizer.tokenize(translated_text))
    print(f"번역된 텍스트의 토큰 수: {token_count}")
    
    # Step 2: 영어 요약 (70 토큰 초과 시에만)
    if token_count > 70:
        summarized_text = summarize_english_text(translated_text, max_length=max_length, min_length=min_length) 
        print(f"요약된 텍스트: {summarized_text}")
    else:
        summarized_text = translated_text 
    
    return summarized_text


# 이미지 생성 함수
def generate_images(prompts, output_dir):
    generator = torch.manual_seed(42)  # 시드 고정
    images = []
    for i, prompt in enumerate(prompts):
        translated_prompt = "A grayscale sketch of " + translate_and_summarize(prompt)
        image = pipe(translated_prompt, num_inference_steps=20, guidance_scale=7.5).images[0]
        image.save(f"{output_dir}/image_xl{i}.png")
        images.append(image)
    print(f"{len(images)}개의 이미지가 생성되었습니다.")
    return images

# 저장 경로 설정
output_dir = "./inference_outputs/item"
os.makedirs(output_dir, exist_ok=True)

# 프롬프트
prompts = [" "]

# 이미지 생성 호출
generate_images(prompts, output_dir)
