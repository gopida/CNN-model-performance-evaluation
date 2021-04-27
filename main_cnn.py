import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.models import Sequential
import tensorflow as tf
import random as rn
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.nasnet import NASNetLarge
from keras.layers import Dropout
from sklearn.model_selection import RandomizedSearchCV
from data_helper import load_data
import sys
from util import plot_confusion_matrix, getlabel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

IMG_SIZE = 64  # Resolution of the image

#X=np.load('/floyd/input/my_data/x.npy')
#Y=np.load('/floyd/input/my_data/y.npy')

X, Y = load_data(IMG_SIZE)

""" Split training and testing data. 75% of the data is used for training and 25% for testing """
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)

""" Data Augmentation to increase the relevant data in the data set """

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

best_num_of_hidden_layers = 3
best_activation = 'relu'
best_dropout_rate =  0.7
best_neurons = 256
best_optimizer = 'Adam'
best_freeze_layers = 75
best_epochs = 1
best_batch_size = 64

# comment/uncomment next few lines to select the pre-trained model of your choice as your base model
# base_model = MobileNet(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='max')
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='max')
# base_model = VGG19(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='max')
# base_model = NASNetLarge(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='max')

model = Sequential()
model.add(base_model)
model.add(BatchNormalization())
for i in range(best_num_of_hidden_layers):
    model.add(Dense(best_neurons, activation=best_activation))
    model.add(Dropout(best_dropout_rate))
    model.add(BatchNormalization())
model.add(Dense(5, activation='softmax'))
for layer in base_model.layers[best_freeze_layers:]:
    layer.trainable=True
for layer in base_model.layers[0:best_freeze_layers]:
    layer.trainable=False

model.compile(optimizer=best_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(datagen.flow(x_train,y_train, batch_size=best_batch_size),epochs = best_epochs,verbose = 1, steps_per_epoch=x_train.shape[0] // best_batch_size)

# Perform testing on the model with high rank
y_pred = model.predict_classes(x_test)

m = len(y_test)

y_test_rev = []
for i in range(m):
    y_test_rev.append(np.argmax(y_test[i]))

print(y_test_rev)
print(y_pred)

# Compute confusion matrix, precision and recall
cf_matrix = confusion_matrix(y_test_rev, y_pred)
print(cf_matrix)
rc_score = recall_score(y_test_rev, y_pred, average=None)
pr_score = precision_score(y_test_rev, y_pred, average=None)

print(rc_score)
print(pr_score)

# Display Confusion Matrix
class_label = ['Daisy', 'Sunflower', 'Tulip', 'Dandelion', 'Rose']
plot_confusion_matrix(cf_matrix, class_label)
plt.show()


# Generate heat map
df_cm = pd.DataFrame(cf_matrix, range(5), range(5))
#plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16})
plt.show()

# Evaluating the incorrect decision
test_size = len(y_test)
sample_vis_count = 0

for i in range(0, test_size):
    if y_test_rev[i] != y_pred[i]:
        img = x_test[i]
        plt.subplot(4, 3, sample_vis_count+1)
        plt.title('Actual label: %s , Prediction: %s' % (getlabel(y_test_rev[i]), getlabel(y_pred[i])), fontsize=8)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        sample_vis_count += 1

    if sample_vis_count == 12:
        plt.show()
        break
