from deepface import DeepFace
import numpy as np
import cv2
import os
import pickle

covered_faces_folder = '../data/covered_faces_arcface'
embeddings_by_identity_outfile = '../data/embeddings_by_identity_arcface_outfile.pkl'

embeddings_by_identity = {}

for identity in os.listdir(covered_faces_folder):
    identity_embeddings = []
    image_names = []
    for image_name in os.listdir(f'{covered_faces_folder}/{identity}'):
        embedding = DeepFace.represent(
                      img_path=f'{covered_faces_folder}/{identity}/{image_name}', model_name='ArcFace', enforce_detection=False, align=False, detector_backend='skip'
                    )[0]['embedding']
        identity_embeddings.append(np.array(embedding))
        image_names.append(image_name)
    embeddings_by_identity[identity] = {'identity_embeddings': identity_embeddings, 'image_names': image_names}
    print(f'Calculating embeddings for identity {identity} finished.')


with open(embeddings_by_identity_outfile, "wb") as out_file:
    pickle.dump(embeddings_by_identity, out_file)

