from tqdm import tqdm
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import os

def load_data(IMG_SIZE):

    DAISY_DIR = 'flowers/daisy'
    SUNFLOWER_DIR = 'flowers/sunflower'
    TULIP_DIR = 'flowers/tulip'
    DANDI_DIR = 'flowers/dandelion'
    ROSE_DIR = 'flowers/rose'

    X = []
    Z = []

    "Load the data set into X array and labels in Z array"
    def make_train_data(flower_type, DIR):
        for img in tqdm(os.listdir(DIR)):
            label = flower_type
            path = os.path.join(DIR, img)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            X.append(np.array(img))
            Z.append(str(label))

    make_train_data('Daisy', DAISY_DIR)
    print(len(X))

    make_train_data('Sunflower', SUNFLOWER_DIR)
    print(len(X))

    make_train_data('Tulip', TULIP_DIR)
    print(len(X))

    make_train_data('Dandelion', DANDI_DIR)
    print(len(X))

    make_train_data('Rose', ROSE_DIR)
    print(len(X))

    """ Encodes the label. Encoding is categorical, each label is represented as binary array for length 5 """

    le = LabelEncoder()
    Y = le.fit_transform(Z)
    Y = to_categorical(Y, 5)
    X = np.array(X)
    X = X / 255

    return X, Y
