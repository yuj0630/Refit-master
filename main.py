import torch
from torchvision import transforms
from PIL import Image

from app import map_class_id_to_name


# 이미지 전처리 함수 정의
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


# 학습된 모델 불러오기
model = torch.load('model_pretrained_False_2nd.pth')

# 이미지 로드 및 전처리
image = Image.open('path_to_image.jpg')
input_tensor = preprocess_image(image)

# 모델 예측
with torch.no_grad():
    model.eval()
    output = model(input_tensor)

# 예측 결과 확인
_, predicted_idx = torch.max(output, 1)
predicted_class = map_class_id_to_name(predicted_idx.item())

print(f'Predicted class: {predicted_class}')
