from numpy import max
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
from random import randint, shuffle
from sklearn.decomposition import PCA

embeddings_by_identity_outfile = '../data/embeddings_by_identity_outfile.pkl'
facenet_SVM_classificator_outfile = '../data/facenet_SVM_classificator_outfile.pkl'
facenet_PCA_outfile = '../data/facenet_PCA_outfile.pkl'
facenet_embeddings_normalizer_outfile = '../data/facenet_embeddings_normalizer_outfile.pkl'
facenet_label_decoder_outfile = '../data/facenet_label_decoder_outfile.pkl'


def prepare_train_test_data(embeddings_by_identity_dict, test_sample_ratio):
    train_x = []
    train_y = []
    train_image_names = []
    test_x = []
    test_y = []
    test_image_names = []

    for identity, data in embeddings_by_identity_dict.items():
        embeddings = data['identity_embeddings']
        image_names = data['image_names']
        test_samples_for_identity = int(len(embeddings) * test_sample_ratio)

        for _ in range(test_samples_for_identity):
            index = randint(0, len(embeddings) - 1)
            test_x.append(embeddings.pop(index))
            test_image_names.append(image_names.pop(index))
            test_y.append(identity)

        for embedding, image_name in zip(embeddings, image_names):
            train_x.append(embedding)
            train_y.append(identity)
            train_image_names.append(image_name)

    zipped = list(zip(train_x, train_y, train_image_names))
    shuffle(zipped)
    train_x, train_y, train_image_names = zip(*zipped)

    zipped = list(zip(test_x, test_y, test_image_names))
    shuffle(zipped)
    test_x, test_y, test_image_names = zip(*zipped)

    return train_x, train_y, train_image_names, test_x, test_y, test_image_names


def find_incorrectly_predicted_images(actual_y, predicted_y, image_names, label_encoder):
    incorrect_predictions = {}
    i = 0
    for actual, predicted in zip(label_encoder.inverse_transform(actual_y), label_encoder.inverse_transform(predicted_y)):
        if actual != predicted:
            if actual not in incorrect_predictions.keys():
                incorrect_predictions[actual] = []
            incorrect_predictions[actual].append(image_names[i])
        i += 1

    return incorrect_predictions


embeddings_by_identity = None
with open(embeddings_by_identity_outfile, "rb") as in_file:
    embeddings_by_identity = pickle.load(in_file)

model = None
num_of_identities_in_classificator = 1000

embeddings_by_identity_subset = {k: embeddings_by_identity[k] for k in list(embeddings_by_identity)[:num_of_identities_in_classificator]}

train_X, train_Y, train_images, test_X, test_Y, test_images = prepare_train_test_data(embeddings_by_identity_subset, 0.2)

pca = PCA(.95)
pca.fit(train_X)

in_encode = Normalizer(norm='l2')
train_X = in_encode.transform(pca.transform(train_X))
test_X = in_encode.transform(pca.transform(test_X))

out_encode = LabelEncoder()
out_encode.fit(train_Y)
train_Y = out_encode.transform(train_Y)
test_Y = out_encode.transform(test_Y)

model = SVC(kernel='rbf', C=0.7, probability=True)
model.fit(train_X, train_Y)

predict_train = model.predict(train_X)
predict_test = model.predict(test_X)

probability = model.predict_proba(test_X)
confidence = max(probability)

acc_train = accuracy_score(train_Y, predict_train)
acc_test = accuracy_score(test_Y, predict_test)

print(f'Number of identities: {num_of_identities_in_classificator}, Accuracy train: {acc_train}, Accuracy test: {acc_test}, Number of dimestions: {train_X[0].shape[0]}')

with open(facenet_SVM_classificator_outfile, "wb") as out_file:
    pickle.dump(model, out_file)

with open(facenet_PCA_outfile, "wb") as out_file:
    pickle.dump(pca, out_file)

with open(facenet_embeddings_normalizer_outfile, "wb") as out_file:
    pickle.dump(in_encode, out_file)

with open(facenet_label_decoder_outfile, "wb") as out_file:
    pickle.dump(out_encode, out_file)
