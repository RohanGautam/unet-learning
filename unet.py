import tensorflow as tf
import pickle

from constants import *
import parse_data
# building the model c1,p1, etc are all labelled in the image

inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
inputs_normalized = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

'''Contraction path'''
# feature dimension: 16, kernel : 3x3, he_normal to initialize the weights
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(inputs_normalized)
# add a droupout to prevent overfitting
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
# c1's output is now 128x128x16. we have to reduce information via maxpooling now.
# a 2x2 square will make the result 128/2 = 64x64x16
p1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(c1)
# now have feature dimensionality (# of features/ dimensions) = 32
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)

# repeat till P4
p2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(c2)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
c3 = tf.keras.layers.Dropout(0.1)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)

p3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(c3)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
c4 = tf.keras.layers.Dropout(0.1)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)

p4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same")(c4)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
c5 = tf.keras.layers.Dropout(0.1)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)


'''Expansive path:'''
# conv2dtranspose used because we are backtracking from shape of output of convolutional layer to shape of input layer(the image). Basically, upscaling instead of downscaling. We use a 2x2 kernel size in this step.
u6 = tf.keras.layers.Conv2DTranspose(128, (2,2),strides=(2,2), activation="relu", kernel_initializer="he_normal", padding="same")(c5)
u6 = tf.keras.layers.concatenate([u6,c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
c6 = tf.keras.layers.Dropout(0.1)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2,2),strides=(2,2), activation="relu", kernel_initializer="he_normal", padding="same")(c6)
u7 = tf.keras.layers.concatenate([u7,c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
c7 = tf.keras.layers.Dropout(0.1)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2,2),strides=(2,2), activation="relu", kernel_initializer="he_normal", padding="same")(c7)
u8 = tf.keras.layers.concatenate([u8,c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2,2),strides=(2,2), activation="relu", kernel_initializer="he_normal", padding="same")(c8)
u9 = tf.keras.layers.concatenate([u9,c1])
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

outputs = tf.keras.layers.Conv2D(1,(1,1), activation="sigmoid")(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

#-------------------#
# Checkpoint: to save model at periodic intervals
# can save model to train later, ar can stop training when a certain event(defined by you) occours
# In tf : *ModelCallback, *EarlyStopping (when you dont know number of epochs to train for)
#         * TensorBoard (visualisation tool, to see hists of loss etc)
# chekpoint:
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_neuclei.h5', verbose=1, save_best_only=True)

callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
]

# get the data
# X_train, Y_train, X_test = parse_data.getData()
X_train, Y_train, X_test = pickle.load(open('parsed_data/X_train.dat','rb')), pickle.load(open('parsed_data/Y_train.dat','rb')), pickle.load(open('parsed_data/X_test.dat','rb'))

# results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)
#######
