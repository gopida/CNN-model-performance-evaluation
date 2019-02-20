import dill
dill.load_session('iris_data.db')

le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,17)
X=np.array(X)
X=X/255

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=64, input_shape=(IMG_SIZE,IMG_SIZE,3), kernel_size=(3,3), padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='valid'))
model.add(Activation('relu'))

# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='valid'))
model.add(Activation('relu'))

# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))


# 3rd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='valid'))
model.add(Activation('relu'))

# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

# 4th Convolutional Layer
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization())


# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.5))
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.5))
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.5))
model.add(BatchNormalization())

# Output Layer
model.add(Dense(17))
model.add(Activation('softmax'))

model.summary()

# (4) Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# (5) Train
history = model.fit(x=x_train, y=y_train, batch_size=5, epochs=1, verbose=1, validation_split=0.2, shuffle=True)

# batch_size=3
# epochs=5
# no_itr_per_epoch=len(X_train)//batch_size
# val_steps=len(X_val)//batch_size
#
# history = model.fit(x=np.array(X_train), y=np.array(Y_train), batch_size=batch_size, epochs=epochs,
#           verbose=1, callbacks=None, validation_split=0.5,
#           validation_data=(np.array(X_val), np.array(Y_val)),
#           shuffle=True, class_weight=None,
#           sample_weight=None, initial_epoch=0)
