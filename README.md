# Refit-master
Refit 최종본
## ☑️ 프로젝트 소개

- Refit은 헌 옷 재활용 웹사이트 프로젝트이며, 실 사업화를 위해 계속 개선 중입니다.
- AI를 사용한 의류 분류와 매칭, 현 위치별 가까운 옷 분리수거 위치 안내, 자신의 의류 종류에 해당하는 캠페인 추천 기능이 프로젝트의 핵심 기술입니다.
- 프로젝트의 주 사용 연령대층은 20 ~ 40대 청년-중년이며, 특히 30 ~ 40대 주부 분들을 타겟팅으로 합니다.

---

## ☑️ 프로젝트 멤버

- 이상준(팀장) : 프로젝트 총괄 팀장, 전반적인 기획, 캠페인 리스트, AI 연동 페이지 개발, AWS EC2 서버 구축(spring)
- 조시언 : 백엔드 팀장, 마이페이지, 로그인, 구글(네이버) 연동, DB 구축, 백엔드 기술 고문
- 정영운 : AI 팀장, 데이터셋 구축, 학습, 커스텀 딥러닝 모델링, inference 서버 개발, AWS EC2 서버 구축(flask)
- 한종훈 : 백엔드 팀원, 거리별 근처 수거함 추천, kakao map API 연동 시스템 개발,
- 고주형 : 백엔드 팀원, 의류 관련 나눔 게시판, 관리자 페이지 개발,

## ☑️ 상세 정보

- 카테고리
    - 의류/친환경
- 내용 및 기능
    - 옷 정리 관련 커뮤니티 기능
        - 옷 정리 팁 공유
            - 옷 정리 관련 질의 응답
            - 필요한 경우 GPT API 연결
        - 옷 무료 나눔
            - 일반 의류, 반려동물 옷, 아기 옷 등
    - 의류 서비스 관련 위치 정보
        - 짐 보관 장소 정보(여분 자리) 및 위치
        - 코인 빨래방 위치 정보 제공
        - 세탁소 위치 및 리뷰 제공
        - 헌 옷 수거함 위치
    - 옷장 내의 옷을 쉽게 분류하고 정리할 수 있는 기능
        - 옷장이 혼잡해서 옷을 찾기 어려운 경우
        - 옷의 상태를 기록하고 관리하는 것이 어려운 경우
        - 옷을 나누거나 기부하고 싶은 경우 (연계)
    - AI를 통한 의류 이미지 분석 및 매칭
        - 의류 카테고리 분류
        - 세탁 및 보관, 처리 방법, 팁 제공
        - 필요로 하는 사람, 업체 정보 리스트 제공
        - 해당 의류 관련 진행 중인 캠페인
        - 옷장 내의 옷을 쉽게 분류하고 정리할 수 있는 기능(연계)
    - 날씨 관련 의류 정보
        - 세탁하기 좋은 날(세탁 지수 제공)
        - 옷 정리가 적절한 시기 정보
    - 옷 정리 관련 가구 및 생활용품 정보
        - 옷 정리 팁 및 리뷰 공유를 통한 아래 상품 정보 연결 및 제공
            - 옷걸이
            - 선반
            - 행거
            - 습기제거제
            - 의류 스타일러 가전
    - 의류 관련 캠페인 정보 제공
        - 헌 옷 수거 불우 이웃 돕기 캠페인
        - 진행중인 업사이클링 캠페인

---

### ☑️ 구현 기술 내역

Back-End : 거리 기능별 추천 시스템, spring + 타 기술을 통한 AI 기능 탑재

사랑의 옷체통 : 댓글, 옷 등록, 나눔이나 기부 시 적용되는 마일리지 시스템 추가

각각 AI 캠페인, 의류 나눔 게시판, kakao map API 기반 근처 의류수거함 찾기, 캠페인 리스트로 설정

프론트엔드 희망자가 없기에 부트스트랩으로 디자인을 미니멀하게 제작하는 대신 기능 다각화

AI : 기본 옷 + 헌 옷 image classification, 분류, label별로 25만 → 4만장의 데이터셋

spring → flask API 요청 시 inference model을 통한 학습 진행 후 label + 정확도 결과 도출

efficient B0, MobileNet V2, diffusion + inception V3 등 여러 모델 시연 후 최상의 모델 선택

data argumentation, ensemble을 통해 정확도 개선 

이후 캠페인 매칭을 통해 분류된 옷 종류에 맞는 캠페인을 매칭해주는 시스템 개발 +

detection + 정확도를 이용한 헌 옷 구별 기능 제작 예정.

Engineer : Linux + Docker로 백엔드 컨테이너화 이후 flask와 연동, aws ec2를 통해 이미지 저장

이미지 url을 json 형식으로 flask에 POST 하면 inference 후 다시 json 형식으로 ec2로 반환

ec2의 json index를 이용해 매칭 시스템 개발 

---

## ☑️ 프로젝트 상세 진행과정 및 나의 역할

역할 : AI 총괄 팀장 (back-end, AI로 분류)

- 1주차 ~ 2주차

이미지 크롤링 + 이미지 분류와 flask를 이용한 front-end, api 디자인 개발

약 20만개의 데이터셋 중 3만개를 간추려 label별로 분류

- 3주차

대분류 → 소분류 label로 다시 추가 5만장 이미지 크롤링

이미지 분류의 주 사용 모델인 EfficientNet, Resnet 우선 모델 학습 후 추가 시험 모델 모색

모델에 대한 주 이론과 각 모델의 장단점 탐색 이후 알맞는 모델 선택할 예정

back-end 파트를 다른 팀원이 맡게 됨에 따라 front-end가 아닌 flask를 이용한 deep learning inference api 개발 후 spring과 연결 

이후 계속 데이터 재정립 과정

- 4주차

모델 추가 학습, 시범 모델인 만큼 모델 경량화에 초점을 두고 Mobilenet, sqeezenet 모델 추가 시험

이외에 개인적으로 inception V3 모델을 사용해 학습

flask api 개발 착수, linux에서 기존 팀의 코드와 api end-point를 설정한 뒤 일치시킴

데이터 최종본 완료, 약 4만장에 대분류 8 label, 소분류 30 label 분류

- 5주차

flask 개발 환경을 Windows에서 linux로 변경, flask api 역시 linux로 개발

최종 모델 efficientB0 모델 + data augmentation + 특정 이미지 oversampling으로 결정

모델 대분류 모델, 소분류 모델 학습 후 ensemble 진행, 기존 분류 모델 Resnet50을 이용한 전이학습 

back-end와 flask api 개념 명세, AWS EC2에 저장되는 이미지의 url을 받아 inference 진행

후에 json 형식으로 이미지 대분류 + 소분류 + 정확도 결과를 다시 ec2로 response.

- 6주차

spring과 flask 환경을 서로 자유롭게 연결하기 위해 aws 환경에서 flask server 업로드

XAI(SHAP)으로 모델 결과 체크를 하려 했으나 tensorflow, keras만 허용

막판에 모델 문제 파악, 5000장 대분류 모델 + 15000장 소분류 모델 학습 후 local에 추가

이후 모델 개선을 위해 yolov8를 베이스로 이미지 + 영상처리 모델 제작 시작

추가 요구 사항으로 특수 부위 detection을 사용한 결함, 손상 부위 체크 모델 제작 예정

 

---

### ☑️ 이후 진행

- 백엔드 개발 쪽에서는 마일리지 기능을 추가로 구현하려 합니다.
- AI 개발 쪽에서는 영상처리 기능, 특정 오염 데이터 detection 기능을 추가하려 합니다.

---

### ☑️ 프로젝트 성과

### 최종 결과

초기 프로토타입 제작 계획 90% 이상 완성, 추후 프론트엔드 부분을 개선하여 실 운영 예정

### 느낀 점

1. 이전 인턴에서 일부분만 체험하고 느끼지 못했던 많은 과정들을 start-to-end 과정으로 혼자 만들어낸 것이 인상에 깊었습니다.
2. 직접 데이터를 겪었을 때 실전 데이터의 경우엔 수도 없이 많은 경우의 수가 있으며 완벽한 모델이라는 것은 엄청 오랜 시간과 돈을 들여야 한다는 것을 알았습니다.
3. 프로젝트를 하면서 이전에 배웠던 수학 이외의 수학도 공부하고, 다양한 논문과 모델을 보면서 배움의 시선을 많이 키운 것이 좋은 경험이 되었습니다.
4. 두 개 이상의 기능 (classification, detection)을 결합하려는 시도를 해보고, 그만큼 많은 시행착오를 한 것이 재밌는 경험이라 생각합니다.

### 아쉬운 점

1. 이전에 해본 classification, detection 이외에 segmentation 파트를 시도하지 못했던 것이 아쉬움에 남습니다. 
2. 혼자 처음부터 끝까지 했기에 수영복, 재질이 다른 바지, 다양한 디자인 등을 걸러내는 데에 시간이 걸렸으며, 기능을 일부 축소한 것이 다소 아쉽습니다.

---

### ☑️ 레퍼런스

github : https://github.com/yuj0630/REFIT-Back

[Image Classification | Papers With Code](https://paperswithcode.com/task/image-classification)

https://dacon.io/competitions

https://debuggercafe.com/transfer-learning-using-efficientnet-pytorch/

[Home | TensorFlow Hub (tfhub.dev)](https://tfhub.dev/)

[open-mmlab/mmdetection: OpenMMLab Detection Toolbox and Benchmark (github.com)](https://github.com/open-mmlab/mmdetection)

[ultralytics/ultralytics: NEW - YOLOv8 🚀 in PyTorch > ONNX > CoreML > TFLite (github.com)](https://github.com/ultralytics/ultralytics/tree/main)

[DeepLearningExamples/PyTorch/Classification/ConvNets/efficientnet at master · NVIDIA/DeepLearningExamples · GitHub](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet)

---

### ☑️ 참고 자료

[https://www.notion.so/REFIT-243af98333744e3a90cb05431e2fc42f](https://www.notion.so/REFIT-243af98333744e3a90cb05431e2fc42f?pvs=21)

https://github.com/yuj0630/Refit

https://velog.io/@jaehyeong/Flask-%EC%9B%B9-%EC%84%9C%EB%B2%84-AWS-EC2%EC%97%90-%EB%B0%B0%ED%8F%AC%ED%95%98%EA%B8%B0
