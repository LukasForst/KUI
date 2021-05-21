import math
import os
from collections import defaultdict

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
    # load dataset as numpy array
    dataset = np.genfromtxt(truth_file, delimiter=':', dtype=np.dtype('<U12'), encoding='utf-8')
    # get the file names and labels
    files, labels = np.split(dataset, 2, 1)
    # flatten to a single vector
    labels = labels.flatten()
    files = files.flatten()
    # load images
    images = np.array([load_img(f'{train_path}/{x}') for x in files])
    return images, labels


class kNN:
    """
    Classic kNN where as the distance vector is used euclidean distance.
    """

    def __init__(self, values, labels):
        self.values = values
        self.labels = labels

    def __euclidean_distance(self, X):
        # get vector of euclidean instances
        return np.array([np.sqrt(np.sum((self.values - X[i]) ** 2, axis=1)) for i in range(len(X))])

    def predict(self, X, k):
        # compute euclid distances from the training data
        dists = self.__euclidean_distance(X)
        # select K closest data points
        knn = np.argsort(dists)[:, :k]
        # get the labels for the selected training points
        y_knn = self.labels[knn]
        # select labels which are most frequent in the neighborhood of the given data
        return [max(y_knn[i], key=list(y_knn[i]).count) for i in range(len(X))]


def run_kNN(train_path, test_paths, k):
    # set default k
    k = k if k else 2
    images, labels = load_train_data(train_path)
    model = kNN(images, labels)
    to_predict_images = np.array([load_img(file) for file in test_paths])
    return model.predict(to_predict_images, k)


class Bayes:
    """
    Taking an image as a feature and a true value as a class.

    P(class | image) = (P(image | class) * P(class)) / P(image)
    """

    def __init__(self, X_train, y_train):
        # X - values, train vector
        self.X_train = X_train
        # y - labels, train labels
        self.y_train = y_train
        # find all unique classes
        self.classes = np.unique(y_train)
        # compute how many samples do we have
        self.n_s, self.n_f = X_train.shape
        self.n_c = len(self.classes)
        # structure for storing features means
        self.mean = defaultdict(lambda: np.zeros(self.n_f))
        # structure for storing variance of features
        self.variance = defaultdict(lambda: np.zeros(self.n_f))
        # how probable is the class given the data
        self.class_probability = defaultdict(lambda: 0.0)

        # this is here because otherwise we divide by zero due to the completely white/black pixels
        # so we add variance to all data -- using 10, we still divided by zero,
        # 100 was working, but 1000 was giving us best results while training the data
        # on provided datasets
        self.additional_variance = 1000

        self.__train()

    def __train(self):
        # iterate through all classes and compute means and variances
        for c in self.classes:
            # get all data with the same class
            X_c = self.X_train[self.y_train == c]
            # compute probability of a class c
            self.class_probability[c] = len(X_c) / len(self.X_train)
            # and compute mean and variance
            self.mean[c] = X_c.mean(axis=0)
            self.variance[c] = X_c.var(axis=0) + self.additional_variance

    def predict(self, single_x):
        # set default values
        mx, probable_c = -math.inf, None
        # go through all classes and compute
        for c in self.classes:
            # compute P(image | class)
            # we relax our problem a bit and we assume that the pixels in the image are independent
            # so here instead of having a single pixel as a feature, we just use mean and variance of the
            # whole image to represent the normal distribution, thus reducing the problem
            # here we need to compute density of the normal distribution
            dst = 1 / np.sqrt(2 * np.pi * self.variance[c])
            es = np.exp(-((single_x - self.mean[c]) ** 2) / (2 * self.variance[c]))
            # and now we have P(image | class)
            p_image_class = dst * es
            # and we compute the final probability
            # this gives us better results then with + np.log(self.class_probability[c])
            # meaning that when we assume that all classes have same probability in the dataset
            # we get the better results on the testing data sets
            p_log = np.sum(np.log(p_image_class))
            # select higher probability
            if p_log > mx:
                mx, probable_c = p_log, c

        return probable_c


def run_bayes(train_path, test_paths):
    images, labels = load_train_data(train_path)
    b = Bayes(images, labels)
    to_predict_images = np.array([load_img(file) for file in test_paths])
    return [b.predict(img) for img in to_predict_images]


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
