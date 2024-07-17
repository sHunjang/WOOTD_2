from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
import os

# 사전에 학습된 YOLOv8 Object Detection 모델 로드
model = YOLO('TOP&BOTTOM_Detection.pt')  # 학습이 완료된 YOLOv8 모델 경로
model.eval()  # 모델을 평가 모드로 설정

# 이미지 경로 설정
target_image_path = '1.png'
owned_image_path = 'Data_1.png'

# 전처리 함수 정의
preprocess = transforms.Compose([
    transforms.Resize((640, 640)),  # YOLOv8의 입력 크기에 맞게 이미지 조정
    transforms.ToTensor(),         # Tensor로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
])

# 상의 이미지만 추출할 이미지 경로 설정
def extract_top_images(image_path, output_folder):
    image = Image.open(image_path).convert('RGB')  # 이미지 로드 및 RGB 모드로 변환
    results = model(image)  # 객체 탐지 수행

    # 상의 객체만 필터링 (클래스 인덱스가 '0'인 상의 객체만 선택)
    tops = results.pandas().xyxy[0]
    tops = tops[tops['class'] == 0]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 상의 영역 추출 및 저장
    for idx, row in tops.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        top_image = image.crop((x1, y1, x2, y2))
        top_image.save(f'{output_folder}/top_image_{idx}.jpg')  # 상의 이미지를 파일로 저장

# 상의 이미지 추출
extract_top_images(target_image_path, 'target_top_images')
extract_top_images(owned_image_path, 'owned_top_images')

# 코사인 유사도 측정 함수 정의
def compute_cosine_similarity(img1_path, img2_path):
    # 상의 이미지를 불러와서 전처리
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    # 이미지를 전처리
    img1 = preprocess(img1).unsqueeze(0)  # 배치 차원 추가
    img2 = preprocess(img2).unsqueeze(0)  # 배치 차원 추가

    # 이미지 텐서를 모델에 통과시켜 피처 벡터 추출
    with torch.no_grad():
        features1 = model(img1).pred[0][0].numpy().flatten()  # 첫 번째 이미지의 피처 벡터
        features2 = model(img2).pred[0][0].numpy().flatten()  # 두 번째 이미지의 피처 벡터

    # 코사인 유사도 계산
    similarity = cosine_similarity([features1], [features2])[0][0]
    return similarity

# 코사인 유사도 계산 예제
target_top_image_path = 'target_top_images/top_image_0.jpg'
owned_top_image_path = 'owned_top_images/top_image_0.jpg'

# 코사인 유사도 계산
similarity_score = compute_cosine_similarity(target_top_image_path, owned_top_image_path)
print(f'Cosine Similarity Score: {similarity_score}')
