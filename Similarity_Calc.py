import torch
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
from torch.nn.functional import cosine_similarity

# YOLOv8 Classification 모델 로드
model = YOLO('Image_Classification.pt')  # 사전에 학습된 YOLOv8 Classification 모델을 로드
model.eval()

# 이미지 전처리 정의
preprocess = transforms.Compose([
    transforms.Resize((640, 640)),  # YOLOv8의 입력 크기에 맞게 이미지 조정
    transforms.ToTensor(),         # Tensor로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
])

# 이미지 로드 및 전처리
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)  # 배치 차원 추가
    return image

# 희망 패션 스타일의 상의 이미지와 소유 의류의 상의 이미지 경로
target_image = load_image('')
owned_image = load_image('')

# 이미지에서 피처 벡터 추출
with torch.no_grad():
    target_features = model(target_image).logits  # 타겟 패션 스타일의 상의 이미지에서 피처 벡터 추출
    owned_features = model(owned_image).logits    # 소유 의류의 상의 이미지에서 피처 벡터 추출

# 피처 벡터를 1D 텐서로 변환
target_features = target_features.flatten()
owned_features = owned_features.flatten()

# 피처 벡터 간의 코사인 유사도 계산
similarity = cosine_similarity(target_features.unsqueeze(0), owned_features.unsqueeze(0))

print(f'Cosine Similarity between target style and owned clothing: {similarity.item()}')
