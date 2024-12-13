import cv2
import numpy as np
from keras_facenet import FaceNet
import pickle
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
import os

# FaceNet 모델 로드
facenet = FaceNet()

# 얼굴 임베딩을 저장할 디렉토리 설정
embedding_directory = 'embeddings'
os.makedirs(embedding_directory, exist_ok=True)

# 얼굴 검출기 MTCNN 로드
detector = MTCNN()


# 저장된 임베딩과 이름을 로드 (pickle로 저장된 임베딩 파일을 불러오기)
def load_embeddings():
    embeddings = {}
    for filename in os.listdir(embedding_directory):
        if filename.endswith('.pkl'):
            with open(os.path.join(embedding_directory, filename), 'rb') as f:
                embedding = pickle.load(f)
                name = os.path.splitext(filename)[0]  # 파일명에서 확장자 제외한 이름 추출
                embeddings[name] = embedding
    return embeddings


# 얼굴 임베딩 비교 함수 (코사인 유사도 기반)
def compare_faces(embedding1, embedding2):
    # 1D로 변환: 얼굴 임베딩을 1D 배열로 변환
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()

    # 코사인 유사도 계산
    distance = cosine(embedding1, embedding2)
    return distance


# 얼굴 검출 및 이름 매칭
def recognize_face():
    embeddings = load_embeddings()  # 저장된 임베딩 로드

    # 웹캡 캡처 시작
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # 얼굴 검출
        faces = detector.detect_faces(frame)

        # 얼굴이 감지되면 사각형 그리기 및 이름 매칭
        for face in faces:
            x, y, w, h = face['box']
            face_image = frame[y:y + h, x:x + w]
            face_image_resized = cv2.resize(face_image, (160, 160))  # FaceNet 모델에 맞게 크기 조정

            # 얼굴 임베딩 계산
            face_embedding = facenet.embeddings(np.expand_dims(face_image_resized, axis=0))

            # 가장 작은 코사인 거리로 가장 유사한 얼굴 찾기
            min_distance = float('inf')
            recognized_name = "Unknown"
            recognized_percent = 0  # 얼굴 일치율을 저장할 변수

            for name, stored_embedding in embeddings.items():
                distance = compare_faces(face_embedding, stored_embedding)
                if distance < min_distance:
                    min_distance = distance
                    if distance < 0.5:  # 임계값을 설정하여 너무 먼 얼굴을 제외
                        recognized_name = name
                        recognized_percent = max(0, 100 - (distance * 100))  # 퍼센트로 변환

            # 얼굴에 이름과 일치율 표시
            if recognized_name == "Unknown":
                color = (0, 0, 255)  # 빨간색 (Unknown일 때)
            else:
                color = (0, 255, 0)  # 초록색 (인식된 사람)

            # 사각형 그리기
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # 이름과 일치율 표시
            label = f"{recognized_name} ({recognized_percent:.2f}%)"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 실시간 얼굴 검출 결과 표시
        cv2.imshow("Face Recognition", frame)

        # 'q' 키를 눌러 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 얼굴을 분석하고 저장된 얼굴과 매칭하여 이름을 띄우는 함수 호출
recognize_face()
