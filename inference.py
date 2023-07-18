import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

# 딥러닝 모델을 로드합니다.
path_to_efficientnet_model = 'model/model_pretrained_False_2nd.pth'  # efficientnet 모델 파일 경로
path_to_efficientnet_small_model = 'model/model_pretrained_True_efficient.pth'  # resnet 모델 파일 경로

efficientnet_model = torch.load(path_to_efficientnet_model, map_location=torch.device('cpu'))
efficientnet_small_model = torch.load(path_to_efficientnet_small_model, map_location=torch.device('cpu'))

# 이미지 전처리를 위한 변환 함수를 정의합니다.
my_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_efficientnet_label(index):
    # 대분류 카테고리 레이블을 설정하는 함수
    # 예시로서 상의, 하의, 속옷 등의 레이블을 설정합니다.
    labels = ['가방', '기타', '모자', '상의', '속옷', '신발', '아기옷', '하의']
    return labels[index]


def get_efficient_small_label(index):
    # 소분류 카테고리 레이블을 설정하는 함수
    # 예시로서 긴팔, 반팔 등의 레이블을 설정합니다.
    labels = ['가방', '긴바지', '긴팔', '넥타이', '드레스', '모자', '목도리', '반려동물 옷', '반바지', '반팔', '벨트', '브래지어', '비니', '샌들', '셔츠',
              '숏팬츠', '아기옷', '양말', '운동화', '자켓', '챙모자', '치마', '코트', '크롭티', '트렁크', '패딩', '팬티', '플립플롭', '하이힐', '후드']
    return labels[index]


def perform_inference(image):
    # 이미지 전처리
    transformed_image = my_transforms(image).unsqueeze(0).to(torch.device('cpu'))

    efficientnet_output = efficientnet_model(transformed_image)
    small_output = efficientnet_small_model(transformed_image)

    efficientnet_probs = torch.softmax(efficientnet_output, dim=1)[0]
    efficientnet_label_idx = torch.argmax(efficientnet_probs).item()
    efficientnet_label = get_efficientnet_label(efficientnet_label_idx)

    small_probs = torch.softmax(small_output, dim=1)[0]
    small_label_idx = torch.argmax(small_probs).item()
    small_label = get_efficient_small_label(small_label_idx)

    # 추론 결과를 JSON 형식으로 반환
    result = {
        'category_l1': efficientnet_label,
        'category_l2': small_label,
        'confidence': float(efficientnet_probs[efficientnet_label_idx * small_label_idx])
    }
    return result


def build_model(pretrained=False, fine_tune=True, num_classes=8, weights=None):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    model = models.efficientnet_b0(weights=weights)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    # Change the final classification head.
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model
