import os

import numpy as np
from PIL import Image, ImageOps


def load_img(file_name):
    # open the image
    img = Image.open(file_name)
    # convert to grayscale to have a single dimension
    gray_scale = ImageOps.grayscale(img)
    # convert to numpy
    arr = np.asarray(gray_scale, dtype='float64')
    # convert to a single vector
    return arr.flatten()


def load_train_data(train_path):
    truth_file = f'{train_path}/truth.dsv'

    dataset = np.genfromtxt(truth_file, delimiter=':', dtype=np.dtype('<U12'), encoding='utf-8')
    files, labels = np.split(dataset, 2, 1)
    # flatten to a single vector
    labels = labels.flatten()
    files = files.flatten()
    # load images
    images = np.array([load_img(f'{train_path}/{x}') for x in files])
    return images, labels


class kNN:
    def __init__(self, values, labels):
        self.values = values
        self.labels = labels

    def __euclidean_distance(self, X):
        euclid = [np.sqrt(np.sum((self.values - X[i]) ** 2, axis=1)) for i in range(len(X))]
        return np.array(euclid)

    def predict(self, X, k):
        dists = self.__euclidean_distance(X)

        knn = np.argsort(dists)[:, :k]
        y_knn = self.labels[knn]

        max_votes = [max(y_knn[i], key=list(y_knn[i]).count) for i in range(len(X))]
        return max_votes


def run_kNN(train_path, test_paths, k):
    # set default k
    k = k if k else 2

    images, labels = load_train_data(train_path)
    model = kNN(images, labels)

    to_predict_images = np.array([load_img(file) for file in test_paths])

    return model.predict(to_predict_images, k)


def run_bayes(train_path, test_files):
    # TODO program bayes
    return run_kNN(train_path, test_files, None)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Learn and Classify image data.')
    parser.add_argument('train_path', help='Path to the training data')
    parser.add_argument('test_path', help='Path to the testing data')
    parser.add_argument('-k', dest='K', type=int,
                        help='run K-NN classifier, if K is 0 the code may decide about proper K by itself')
    parser.add_argument('-o', dest='name',
                        help='name (with path) of the output dsv file with the results')
    parser.add_argument('-b', action='store_true', help='run Naive Bayes classifier')

    args = parser.parse_args()

    test_files = [x for x in os.listdir(args.test_path) if not x.endswith('dsv')]
    test_paths = [f'{args.test_path}/{x}' for x in test_files]
    if args.b:
        predicted = run_bayes(args.train_path, test_paths)
    else:
        predicted = run_kNN(args.train_path, test_paths, args.K)

    output_string = ''
    for i in range(len(test_files)):
        output_string += f'{test_files[i]}:{predicted[i]}\n'

    with open(args.name, 'w') as f:
        f.write(output_string)
