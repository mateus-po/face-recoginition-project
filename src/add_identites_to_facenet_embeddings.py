import numpy as np
import cv2
from keras_facenet import FaceNet
import os
import pickle

facenet_model = FaceNet()

covered_faces_folder = '../data/additional_identities/covered_faces'
embeddings_by_identity_outfile = '../data/embeddings_by_identity_outfile.pkl'

def get_embedding(model, cropped_face):
    face = cropped_face.astype('float32')
    sample = np.expand_dims(face, axis=0)

    yhat = model.embeddings(sample)
    return yhat[0]


embeddings_by_identity = {}


for identity in os.listdir(covered_faces_folder):
    identity_embeddings = []
    image_names = []
    for image_name in os.listdir(f'{covered_faces_folder}/{identity}'):
        image = cv2.imread(f'{covered_faces_folder}/{identity}/{image_name}')
        image = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        identity_embeddings.append(get_embedding(facenet_model, image))
        image_names.append(image_name)
    embeddings_by_identity[identity] = {'identity_embeddings': identity_embeddings, 'image_names': image_names}
    print(f'Calculating embeddings for identity {identity} finished.')

with open(embeddings_by_identity_outfile, "rb") as in_file:
    embeddings_by_identity.update(pickle.load(in_file))

with open(embeddings_by_identity_outfile, "wb") as out_file:
    pickle.dump(embeddings_by_identity, out_file)
