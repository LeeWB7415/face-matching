import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN
import pickle
import os
import glob

# FaceNet 모델 로드
facenet = FaceNet()

# 얼굴 검출기 MTCNN 로드
detector = MTCNN()

# 얼굴 임베딩을 저장할 디렉토리 설정
embedding_directory = 'embeddings'
os.makedirs(embedding_directory, exist_ok=True)

# 얼굴을 pkl로 저장하는 함수
def save_face_embedding(image_path):
    # 이미지 읽기
    img = cv2.imread(image_path)

    # MTCNN을 사용해 얼굴 검출
    faces = detector.detect_faces(img)

    if len(faces) == 0:
        print(f"No face detected in the image: {image_path}")
        return

    # 첫 번째 얼굴만 처리 (여러 얼굴이 있을 경우 첫 번째 얼굴을 사용)
    x, y, w, h = faces[0]['box']
    face_image = img[y:y + h, x:x + w]

    # FaceNet 모델에 맞게 얼굴 이미지 크기 조정
    face_image_resized = cv2.resize(face_image, (160, 160))

    # 얼굴 임베딩 추출
    face_embedding = facenet.embeddings(np.expand_dims(face_image_resized, axis=0))[0]  # (1, 128) -> (128,)

    # 이미지 파일 이름에서 확장자를 제외한 이름만 추출
    name = os.path.splitext(os.path.basename(image_path))[0]

    # 임베딩을 pickle 파일로 저장
    embedding_path = os.path.join(embedding_directory, f"{name}.pkl")

    with open(embedding_path, 'wb') as f:
        pickle.dump(face_embedding, f)

    print(f"Face embedding for {name} saved at {embedding_path}")

# 이미지 경로에 맞는 모든 .jpg, .png 파일을 찾기
image_paths = glob.glob("../faces/*/*.jpg") + glob.glob("../faces/*/*.png") + glob.glob("../faces/*/*.jpeg") # 현재 디렉토리에서 모든 .jpg와 .png 파일 찾기

# 각 이미지에 대해 얼굴 임베딩 저장
for image_path in image_paths:
    save_face_embedding(image_path)
