import json
import os
import torch
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import transforms
from inference import build_model
import base64
from io import BytesIO
import cv2
import numpy as np
import torchvision.transforms as transforms

app = Flask(__name__)

# 모델과 클래스 매핑 파일 경로 설정

# efficientnet 모델 파일 경로
path_to_efficientnet_model = 'model/model_True_384_대분류.pth'

# 상의 소분류 모델 경로
tops_model_path = 'model/model_True_384_소분류_상의.pth'

# 하의 소분류 모델 경로
bottoms_model_path = 'model/model_True_384_소분류_하의.pth'

# 신발 소분류 모델 경로
shoes_model_path = 'model/model_True_384_소분류_신발.pth'

# 클래스 매핑 파일 경로
class_mapping_file = 'class_mapping.json'

# 모델 로드
# efficientnet_model = torch.load(path_to_efficientnet_model, map_location=torch.device('cpu'))

efficientnet_model = build_model(pretrained=True, fine_tune=True, num_classes=5)
efficientnet_checkpoint = torch.load(path_to_efficientnet_model, map_location='cpu')
efficientnet_model.load_state_dict(efficientnet_checkpoint['model_state_dict'])

# 상의 소분류 모델 로드
tops_model = build_model(pretrained=True, fine_tune=True, num_classes=5)
tops_checkpoint = torch.load(tops_model_path, map_location=torch.device('cpu'))
tops_model.load_state_dict(tops_checkpoint['model_state_dict'])

# 하의 소분류 모델 로드
bottoms_model = build_model(pretrained=True, fine_tune=True, num_classes=4)
bottoms_checkpoint = torch.load(bottoms_model_path, map_location=torch.device('cpu'))
bottoms_model.load_state_dict(bottoms_checkpoint['model_state_dict'])

# 신발 소분류 모델 로드
shoes_model = build_model(pretrained=True, fine_tune=True, num_classes=3)
shoes_checkpoint = torch.load(shoes_model_path, map_location=torch.device('cpu'))
shoes_model.load_state_dict(shoes_checkpoint['model_state_dict'])

# 클래스 매핑 파일 로드
class_mapping = {}
if os.path.isfile(class_mapping_file):
    with open(class_mapping_file) as f:
        class_mapping = json.load(f)


def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image)
    input_tensor = input_tensor.unsqueeze(0)  # 차원을 늘려서 4차원 텐서로 변환
    return input_tensor

# efficientnet 모델 예측 함수 정의
def efficientnet_predict(input_tensor):
    with torch.no_grad():
        output = efficientnet_model(input_tensor)
        print(str(output))
        _, predicted_idx = torch.max(output, 1)
        return predicted_idx.item()


def resnet_predict(input_tensor, category):
    with torch.no_grad():
        if category == '상의':
            print("상의!!!")
            output = tops_model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            return predicted_idx.item()
        elif category == '하의':
            print("하의!!!")
            output = bottoms_model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            return predicted_idx.item()
        elif category == '신발':
            print("신발!!!")
            output = shoes_model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            return predicted_idx.item()
        else:
            return -1


# 클래스 ID를 클래스 이름으로 변환하는 함수 정의
def map_class_id_to_name(class_id, sub_class_id, input_tensor, is_efficientnet=True):
    #input_tensor = preprocess_image(input_tensor)
    print("class_id=",class_id)
    print(" is_efficientnet=", is_efficientnet)

    if is_efficientnet:
        class_name = (['가방', '기타', '상의', '신발', '하의'][class_id])
    else:
        if class_id == 0:
            class_name = ['가방']
        elif class_id == 1:
            class_name = ['기타']
        elif class_id == 2:  # '상의'에 해당하는 클래스 ID
            class_name =  ['기타상의', '맨투맨', '셔츠', '아우터', '티셔츠'][sub_class_id]
        elif class_id == 3:  # '신발'에 해당하는 클래스 ID
            class_name = ['운동화', '샌들', '기타신발'][sub_class_id]
        elif class_id == 4:  # '하의'에 해당하는 클래스 ID
            class_name = ['긴바지', '반바지', '숏팬츠', '치마'][sub_class_id]
        else:
            class_name = "Unknown"
    return class_name

#
# def map_resnet_category(input_tensor, labels, category):
#     print("map_resnet_category:labels:",labels)
#     print("map_resnet_category:category:",category)
#     if category == '상의':
#         model = tops_model
#     elif category == '하의':
#         model = bottoms_model
#     elif category == '신발':
#         model = shoes_model
#     else:
#         return "Unknown"
#
#     with torch.no_grad():
#         output = model(input_tensor)
#         _, predicted_idx = torch.max(output, 1)
#         class_name = labels[predicted_idx.item()]
#     return class_name


@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg': 'Try POSTing to the /predict endpoint with an image attachment'})


@app.route('/predict', methods=['POST'])
def predict():
    encoded_data = request.data.decode('utf-8')

    encoded_data = encoded_data.replace("image/jpeg;base64,", "")

    # encoded_data 를 원래의 이미지로 저장
    decoded_data = base64.b64decode(encoded_data)
    image = Image.open(BytesIO(decoded_data))
    print("2")
    input_tensor = preprocess_image(image)
    print("1")
    efficientnet_prediction = efficientnet_predict(input_tensor)
    efficientnet_class_name = map_class_id_to_name(efficientnet_prediction,-1, image, is_efficientnet=True)
    print("efficient_class_name=",efficientnet_class_name)
    print("3")
    resnet_class_name = None
    print("4")
    if efficientnet_class_name in ['상의', '하의', '신발']:
        resnet_prediction = resnet_predict(input_tensor, efficientnet_class_name)
        print("resnet_prediction=",resnet_prediction)
        resnet_class_name = map_class_id_to_name(efficientnet_prediction, resnet_prediction, input_tensor, is_efficientnet=False)
        print("resnet_class_name=",resnet_class_name)
    result = {
        'efficientnet_category': efficientnet_class_name,
        'resnet_category': resnet_class_name
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
