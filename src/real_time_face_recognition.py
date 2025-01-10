import cv2 as cv
import numpy as np
from mtcnn.mtcnn import MTCNN
import os
import pickle
from keras_facenet import FaceNet
from unidecode import unidecode

video_camera_id = 0
facenet_SVM_classificator_outfile = '../data/facenet_SVM_classificator_outfile.pkl'
facenet_PCA_outfile = '../data/facenet_PCA_outfile.pkl'
facenet_embeddings_normalizer_outfile = '../data/facenet_embeddings_normalizer_outfile.pkl'
facenet_label_decoder_outfile = '../data/facenet_label_decoder_outfile.pkl'

# INITIALIZE MODELS
detector = MTCNN()
facenet_model = FaceNet()

facenet_SVM_classifier = None
facenet_PCA = None
facenet_embeddings_normalizer = None
facenet_label_decoder = None

with open(facenet_SVM_classificator_outfile, "rb") as in_file:
    facenet_SVM_classifier = pickle.load(in_file)

with open(facenet_PCA_outfile, "rb") as in_file:
    facenet_PCA = pickle.load(in_file)

with open(facenet_embeddings_normalizer_outfile, "rb") as in_file:
    facenet_embeddings_normalizer = pickle.load(in_file)

with open(facenet_label_decoder_outfile, "rb") as in_file:
    facenet_label_decoder = pickle.load(in_file)

cap = cv.VideoCapture(video_camera_id)
while cap.isOpened():
    _, frame = cap.read()
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    faces = detector.detect_faces(np.asarray(rgb_img))
    for face_data in faces:
        x, y, w, h = face_data['box']
        img = rgb_img[y:y + h, x:x + w]
        img = cv.resize(img, (160, 160))
        img = np.expand_dims(img, axis=0)
        embedding = facenet_model.embeddings(img)
        normalized_embedding = facenet_embeddings_normalizer.transform(facenet_PCA.transform(embedding))
        face_name = facenet_SVM_classifier.predict(normalized_embedding)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(frame, str(unidecode(facenet_label_decoder.inverse_transform(face_name)[0])),
                   (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

    cv.imshow("Face Recognition:", frame)
    if cv.waitKey(1) & ord('q') == 27:
        break

cap.release()
cv.destroyAllWindows()
