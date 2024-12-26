import os
import csv
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from pathlib import Path

source_images_folder = '../img_align_celeba'
covered_faces_folder = '../data/covered_faces_arcface'
identities_file = '../identity_CelebA.txt'

Path(covered_faces_folder).mkdir(parents=True, exist_ok=True)


def extract_image(image):
    img1 = Image.open(image)
    img1 = img1.convert('RGB')
    pixels = asarray(img1)
    detector = MTCNN()
    f = detector.detect_faces(pixels)
    x1, y1, w, h = f[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2 = abs(x1 + w)
    y2 = abs(y1 + h)
    f[0]['keypoints']['nose'][0] -= x1
    f[0]['keypoints']['nose'][1] -= y1
    store_face = pixels[y1:y2, x1:x2]
    image1 = Image.fromarray(store_face, 'RGB')
    return image1, f[0]


def cover_face(image, mtcnn_data_dict):
    image_array = asarray(image)
    x, y = mtcnn_data_dict['keypoints']['nose']

    image_array[y:, :, :] = 0
    return Image.fromarray(image_array, 'RGB')


def find_identities_with_most_samples(number_of_identities):
    files_by_identity = {}
    with open(identities_file) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ')
        for row in spamreader:
            filename, identity = row
            if identity not in files_by_identity.keys():
                files_by_identity[identity] = []
            files_by_identity[identity].append(filename)
    identities = sorted(files_by_identity.items(), reverse=True, key=lambda x: len(x[1]))[:number_of_identities]
    print(identities[0])

    result = {}
    for identity_with_files in identities:
        result[identity_with_files[0]] = identity_with_files[1]

    print(f'Identity with most images: {identities[0][0]}, images: {len(identities[0][1])}')
    print(f'Identity with fewest images: {identities[-1][0]}, images: {len(identities[-1][1])}')
    return result


def extract_and_cover_faces(number_of_identities):
    for identity, filenames in find_identities_with_most_samples(number_of_identities).items():
        for filename in filenames:
            if not os.path.isdir(covered_faces_folder + '/' + identity):
                os.mkdir(covered_faces_folder + '/' + identity)
            if os.path.isfile(covered_faces_folder + '/' + identity + '/' + filename):
                continue
            try:
                image_extracted, mtcnn_data = extract_image(source_images_folder + '/' + filename)
                covered_image = cover_face(image_extracted, mtcnn_data)

                covered_image = covered_image.resize((112, 112))
                covered_image.save(covered_faces_folder + '/' + identity + '/' + filename)
                print(f'Image {filename} successfully extracted for identity: {identity}')
            except Exception:
                print(f'could not extract face from image: {filename}')

extract_and_cover_faces(1000)



