# face-recognition-project

Project for identity recognition for faces with covered lower part of the face. 
The implementation is based on the MTCNN model for extracting and covering the face on the image,
FaceNet/ArcFace models for extracting embeddings for each of the covered faces and
SVM classifier for identity classification

# Running the project

## Data preparation
To be able to run the project you need to first download the labeled dataset of faces.
The implementation is based on the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset.
Download the dataset and paste the `img_align_celeba` folder and `identity_CelebA.txt` file from the dataset to the project root folder.

If you want to include additional identities in the classifier, you will need to put the images of such identities in the 
`data/additional_identities/raw_images/<identity>`. Make sure that the `<identity>` folder name
is something distinct from the identities names included in the CelebA dataset (which are numbers 1-10200), as the folder name will be used as an identity label.

## Preparing the images
run `src/prepare_covered_images.py` to extract covered faces from the dataset images.

## Prepare embeddings
Run `prepare_facenet_embeddings.py` to calculate a dataset of face embeddings for the covered images.

## (Optional) Add additional identities embeddings to the dataset
If added some additional identities as mentioned before in the Data Preparation section, you will need to run `add_identites_to_facenet_embeddings.py` so that those embeddings of that additional identities are included in the dataset

## Prepare the SVM classifier
Run `prepare_SVM.py` to prepare the SVM classifier that will learn identities classification on the embeddings dataset created before. This script will also create other files (PCA model, label encoder and embeddings normalizer) that will be needed for the real-time face recognition script

## Run real-time face recognition script
Run `real_time_face_recognition.py` to employ the SVM classifier in the real-time face recognition program that will use the computer camera image to extract face images and it will try to predict the face identity.
