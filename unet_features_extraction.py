import matplotlib.pyplot as plt
import soundfile as sf
#from PIL import Image
from scipy import signal
import scipy as sp
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping

def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['accuracy'])
    encoder=Model(input=inputs, output=drop5)
    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model, encoder



a07_bic = np.load('a07_bic_128.npy')
a08_bic = np.load('a08_bic_128.npy')
a09_bic = np.load('a09_bic_128.npy')
a10_bic = np.load('a10_bic_128.npy')
a11_bic = np.load('a11_bic_128.npy')
a12_bic = np.load('a12_bic_128.npy')
a13_bic = np.load('a13_bic_128.npy')
a14_bic = np.load('a14_bic_128.npy')
a15_bic = np.load('a15_bic_128.npy')
a16_bic = np.load('a16_bic_128.npy')
a17_bic = np.load('a17_bic_128.npy')
a18_bic = np.load('a18_bic_128.npy')
a19_bic = np.load('a19_bic_128.npy')
#a17_bic = np.load('npy_arrays/asv_unet_128/a17_bic_128.npy')

bona_asv = np.load('bona_bic_eval_128.npy')


a07_bic = np.load('npy_arrays/asv_unet_128/a07_bic_128.npy')
a08_bic = np.load('npy_arrays/asv_unet_128/a08_bic_128.npy')
a09_bic = np.load('npy_arrays/asv_unet_128/a09_bic_128.npy')
a10_bic = np.load('npy_arrays/asv_unet_128/a10_bic_128.npy')
a11_bic = np.load('a11_bic_128.npy')
a12_bic = np.load('a12_bic_128.npy')
a13_bic = np.load('npy_arrays/asv_unet_128/a13_bic_128.npy')
a14_bic = np.load('npy_arrays/asv_unet_128/a14_bic_128.npy')
a15_bic = np.load('npy_arrays/asv_unet_128/a15_bic_128.npy')
a16_bic = np.load('npy_arrays/asv_unet_128/a16_bic_128.npy')
a17_bic = np.load('npy_arrays/asv_unet_128/a17_bic_128.npy')
a18_bic = np.load('npy_arrays/asv_unet_128/a18_bic_128.npy')
a19_bic = np.load('npy_arrays/asv_unet_128/a19_bic_128.npy')
bona_asv=np.load('bona_bic_eval_128.npy')

# preprocessing (modules)

a07=np.abs(a07_bic)
a08=np.abs(a08_bic)
a09=np.abs(a09_bic)
a10=np.abs(a10_bic)
a11=np.abs(a11_bic)
a12=np.abs(a12_bic)
a13=np.abs(a13_bic)
a14=np.abs(a14_bic)
a15=np.abs(a15_bic)
a16=np.abs(a16_bic)
a17=np.abs(a17_bic)
a18=np.abs(a18_bic)
a19=np.abs(a19_bic)
bona=np.abs(bona_asv)

a07 = np.reshape(a07, (len(a07), 128, 128, 1))
a08 = np.reshape(a08, (len(a08), 128, 128, 1))
a09 = np.reshape(a09, (len(a09), 128, 128, 1))
a10 = np.reshape(a10, (len(a10), 128, 128, 1))
a11 = np.reshape(a11, (len(a11), 128, 128, 1))
a12 = np.reshape(a12, (len(a12), 128, 128, 1))
a13 = np.reshape(a13, (len(a13), 128, 128, 1))
a14 = np.reshape(a14, (len(a14), 128, 128, 1))
a15 = np.reshape(a15, (len(a15), 128, 128, 1))
a16 = np.reshape(a16, (len(a16), 128, 128, 1))
a17 = np.reshape(a17, (len(a17), 128, 128, 1))
a18 = np.reshape(a18, (len(a18), 128, 128, 1))
a19 = np.reshape(a19, (len(a19), 128, 128, 1))
bona = np.reshape(bona, (len(bona), 128, 128, 1))

a07_bic=[]
a08_bic=[]
a09_bic=[]
a10_bic=[]
a11_bic=[]
a12_bic=[]
a13_bic=[]
a14_bic=[]
a15_bic=[]
a16_bic=[]
a17_bic=[]
a18_bic=[]
a19_bic=[]
bona_asv=[]

# UNET training a07

x_train07=a07[0:3000]
x_vali07=a07[3000:4000]

model07, encoder07=unet('weigths/unet_phases.ckpt', (128, 128, 1))

checkpoint = ModelCheckpoint("weigths/uneta07.ckpt", monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
early = EarlyStopping(monitor='loss', min_delta=0, patience=5,
                      verbose=1, mode='auto')


history07=model07.fit(x_train07, x_train07,
                epochs=100,
                batch_size=64,
                shuffle=True,
                validation_data=(x_vali07, x_vali07),
                callbacks=[ checkpoint, early])


plt.figure()
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.plot(history07.history['loss'])
plt.plot(history07.history['val_loss'])
plt.title("model loss functions")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["loss", "validation loss"])

#plt.show()
plt.savefig('unet_fe_a07_loss_modules.png')


# UNET training a10

x_train10=a10[0:3000]
x_vali10=a10[3000:4000]

model10, encoder10=unet('weigths/unet_phases.ckpt', (128, 128, 1))

checkpoint = ModelCheckpoint("weigths/uneta10.ckpt", monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
early = EarlyStopping(monitor='loss', min_delta=0, patience=5,
                      verbose=1, mode='auto')


history10=model10.fit(x_train10, x_train10,
                epochs=100,
                batch_size=64,
                shuffle=True,
                validation_data=(x_vali10, x_vali10),
                callbacks=[ checkpoint, early])

plt.figure()
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.plot(history10.history['loss'])
plt.plot(history10.history['val_loss'])
plt.title("model loss functions")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["loss", "validation loss"])

#plt.show()
plt.savefig('unet_fe_a10_loss_modules.png')




# UNET training a15

x_train15=a15[0:3000]
x_vali15=a15[3000:4000]

model15, encoder15=unet('weigths/unet_phases.ckpt', (128, 128, 1))

checkpoint = ModelCheckpoint("weigths/uneta15.ckpt", monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
early = EarlyStopping(monitor='loss', min_delta=0, patience=5,
                      verbose=1, mode='auto')


history15=model15.fit(x_train15, x_train15,
                epochs=100,
                batch_size=64,
                shuffle=True,
                validation_data=(x_vali15, x_vali15),
                callbacks=[ checkpoint, early])



plt.figure()
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.plot(history15.history['loss'])
plt.plot(history15.history['val_loss'])
plt.title("model loss functions")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["loss", "validation loss"])

#plt.show()
plt.savefig('unet_fe_a15_loss_modules.png')

# UNET training a16

x_train16=a16[0:3000]
x_vali16=a16[3000:4000]

model16, encoder16=unet('weigths/unet_phases.ckpt', (128, 128, 1))

checkpoint = ModelCheckpoint("weigths/uneta16.ckpt", monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
early = EarlyStopping(monitor='loss', min_delta=0, patience=5,
                      verbose=1, mode='auto')


history16=model16.fit(x_train16, x_train16,
                epochs=100,
                batch_size=64,
                shuffle=True,
                validation_data=(x_vali16, x_vali16),
                callbacks=[ checkpoint, early])

plt.figure()
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.plot(history16.history['loss'])
plt.plot(history16.history['val_loss'])
plt.title("model loss functions")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["loss", "validation loss"])

#plt.show()
plt.savefig('unet_fe_a16_loss_modules.png')

#  UNET_training_a17

x_train17=a17[0:3000]
x_vali17=a17[3000:4000]

model17, encoder17=unet('weigths/unet_phases.ckpt', (128, 128, 1))

checkpoint = ModelCheckpoint("weigths/uneta17.ckpt", monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
early = EarlyStopping(monitor='loss', min_delta=0, patience=5,
                      verbose=1, mode='auto')


history17=model17.fit(x_train17, x_train17,
                epochs=100,
                batch_size=64,
                shuffle=True,
                validation_data=(x_vali17, x_vali17),
                callbacks=[ checkpoint, early])


plt.figure()
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.plot(history17.history['loss'])
plt.plot(history17.history['val_loss'])
plt.title("model loss functions")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["loss", "validation loss"])

#plt.show()
plt.savefig('unet_fe_a17_loss_modules.png')






gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

#  TESTS

model07, encoder07=unet('weigths/uneta07.ckpt', (128, 128, 1))
model10, encoder10=unet('weigths/uneta10.ckpt', (128, 128, 1))
model15, encoder15=unet('weigths/uneta15.ckpt', (128, 128, 1))
model16, encoder16=unet('weigths/uneta16.ckpt', (128, 128, 1))
model17, encoder17=unet('weigths/uneta17.ckpt', (128, 128, 1))


# model07

a07_sfocate_07=model07.predict(a07)
a08_sfocate_07=model07.predict(a08)
a09_sfocate_07=model07.predict(a09)
a10_sfocate_07=model07.predict(a10)
a11_sfocate_07=model07.predict(a11)
a12_sfocate_07=model07.predict(a12)
a13_sfocate_07=model07.predict(a13)
a14_sfocate_07=model07.predict(a14)
a15_sfocate_07=model07.predict(a15)
a16_sfocate_07=model07.predict(a16)
a17_sfocate_07=model07.predict(a17)
a18_sfocate_07=model07.predict(a18)
a19_sfocate_07=model07.predict(a19)
bona_sfocate_07=model07.predict(bona)

"""
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(a07[i].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(a07_sfocate_07[i].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
#plt.show()
plt.savefig('unet_a07_modules_07.png')
"""


a07_compressed_07=encoder07.predict(a07)
a08_compressed_07=encoder07.predict(a08)
a09_compressed_07=encoder07.predict(a09)
a10_compressed_07=encoder07.predict(a10)
a11_compressed_07=encoder07.predict(a11)
a12_compressed_07=encoder07.predict(a12)
a13_compressed_07=encoder07.predict(a13)
a14_compressed_07=encoder07.predict(a14)
a15_compressed_07=encoder07.predict(a15)
a16_compressed_07=encoder07.predict(a16)
a17_compressed_07=encoder07.predict(a17)
a18_compressed_07=encoder07.predict(a18)
a19_compressed_07=encoder07.predict(a19)
bona_compressed_07=encoder07.predict(bona)

a07_sfocate_compressed_07=encoder07.predict(a07_sfocate_07)
a08_sfocate_compressed_07=encoder07.predict(a08_sfocate_07)
a09_sfocate_compressed_07=encoder07.predict(a09_sfocate_07)
a10_sfocate_compressed_07=encoder07.predict(a10_sfocate_07)
a11_sfocate_compressed_07=encoder07.predict(a11_sfocate_07)
a12_sfocate_compressed_07=encoder07.predict(a12_sfocate_07)
a13_sfocate_compressed_07=encoder07.predict(a13_sfocate_07)
a14_sfocate_compressed_07=encoder07.predict(a14_sfocate_07)
a15_sfocate_compressed_07=encoder07.predict(a15_sfocate_07)
a16_sfocate_compressed_07=encoder07.predict(a16_sfocate_07)
a17_sfocate_compressed_07=encoder07.predict(a17_sfocate_07)
a18_sfocate_compressed_07=encoder07.predict(a18_sfocate_07)
a19_sfocate_compressed_07=encoder07.predict(a19_sfocate_07)
bona_sfocate_compressed_07=encoder07.predict(bona_sfocate_07)


mse_a07_07=np.zeros(len(a07))
mse_a08_07=np.zeros(len(a08))
mse_a09_07=np.zeros(len(a09))
mse_a10_07=np.zeros(len(a10))
mse_a11_07=np.zeros(len(a11))
mse_a12_07=np.zeros(len(a12))
mse_a13_07=np.zeros(len(a13))
mse_a14_07=np.zeros(len(a14))
mse_a15_07=np.zeros(len(a15))
mse_a16_07=np.zeros(len(a16))
mse_a17_07=np.zeros(len(a17))
mse_a18_07=np.zeros(len(a18))
mse_a19_07=np.zeros(len(a19))
mse_bona_07=np.zeros(len(bona))


for i in range(len(a07)):
    mse_a07_07[i] = (np.square(a07_compressed_07[i] -
                               a07_sfocate_compressed_07[i])).mean()

for i in range(len(a08)):
    mse_a08_07[i] = (np.square(a08_compressed_07[i] -
                               a08_sfocate_compressed_07[i])).mean()

for i in range(len(a09)):
    mse_a09_07[i] = (np.square(a09_compressed_07[i] -
                               a09_sfocate_compressed_07[i])).mean()



for i in range(len(a10)):
    mse_a10_07[i] = (np.square(a10_compressed_07[i]
                               - a10_sfocate_compressed_07[i])).mean()

for i in range(len(a11)):
    mse_a11_07[i] = (np.square(a11_compressed_07[i]
                               - a11_sfocate_compressed_07[i])).mean()

for i in range(len(a12)):
    mse_a12_07[i] = (np.square(a12_compressed_07[i]
                               - a12_sfocate_compressed_07[i])).mean()

for i in range(len(a13)):
    mse_a13_07[i] = (np.square(a13_compressed_07[i]
                               - a13_sfocate_compressed_07[i])).mean()

for i in range(len(a14)):
    mse_a14_07[i] = (np.square(a14_compressed_07[i]
                               - a14_sfocate_compressed_07[i])).mean()

for i in range(len(a15)):
    mse_a15_07[i] = (np.square(a15_compressed_07[i] - a15_sfocate_compressed_07[i])).mean()

for i in range(len(a16)):
    mse_a16_07[i] = (np.square(a16_compressed_07[i] - a16_sfocate_compressed_07[i])).mean()

for i in range(len(a17)):
    mse_a17_07[i] = (np.square(a17_compressed_07[i]
                               - a17_sfocate_compressed_07[i])).mean()

for i in range(len(a18)):
    mse_a18_07[i] = (np.square(a18_compressed_07[i]
                               - a18_sfocate_compressed_07[i])).mean()

for i in range(len(a19)):
    mse_a19_07[i] = (np.square(a19_compressed_07[i]
                               - a19_sfocate_compressed_07[i])).mean()

for i in range(len(bona)):
    mse_bona_07[i] = (np.square(bona_compressed_07[i] -
                                bona_sfocate_compressed_07[i])).mean()


a07_sfocate_07=[]
a08_sfocate_07=[]
a09_sfocate_07=[]
a10_sfocate_07=[]
a11_sfocate_07=[]
a12_sfocate_07=[]
a13_sfocate_07=[]
a14_sfocate_07=[]
a15_sfocate_07=[]
a16_sfocate_07=[]
a17_sfocate_07=[]
a18_sfocate_07=[]
a19_sfocate_07=[]
bona_sfocate_07=[]

a07_compressed_07=[]
a08_compressed_07=[]
a09_compressed_07=[]
a10_compressed_07=[]
a11_compressed_07=[]
a12_compressed_07=[]
a13_compressed_07=[]
a14_compressed_07=[]
a15_compressed_07=[]
a16_compressed_07=[]
a17_compressed_07=[]
a18_compressed_07=[]
a19_compressed_07=[]
bona_compressed_07=[]

a07_sfocate_compressed_07=[]
a08_sfocate_compressed_07=[]
a09_sfocate_compressed_07=[]
a10_sfocate_compressed_07=[]
a11_sfocate_compressed_07=[]
a12_sfocate_compressed_07=[]
a13_sfocate_compressed_07=[]
a14_sfocate_compressed_07=[]
a15_sfocate_compressed_07=[]
a16_sfocate_compressed_07=[]
a17_sfocate_compressed_07=[]
a18_sfocate_compressed_07=[]
a19_sfocate_compressed_07=[]
bona_sfocate_compressed_07=[]



# model10

a07_sfocate_10=model10.predict(a07)
a08_sfocate_10=model10.predict(a08)
a09_sfocate_10=model10.predict(a09)
a10_sfocate_10=model10.predict(a10)
a11_sfocate_10=model10.predict(a11)
a12_sfocate_10=model10.predict(a12)
a13_sfocate_10=model10.predict(a13)
a14_sfocate_10=model10.predict(a14)
a15_sfocate_10=model10.predict(a15)
a16_sfocate_10=model10.predict(a16)
a17_sfocate_10=model10.predict(a17)
a18_sfocate_10=model10.predict(a18)
a19_sfocate_10=model10.predict(a19)
bona_sfocate_10=model10.predict(bona)


a07_compressed_10=encoder10.predict(a07)
a08_compressed_10=encoder10.predict(a08)
a09_compressed_10=encoder10.predict(a09)
a10_compressed_10=encoder10.predict(a10)
a11_compressed_10=encoder10.predict(a11)
a12_compressed_10=encoder10.predict(a12)
a13_compressed_10=encoder10.predict(a13)
a14_compressed_10=encoder10.predict(a14)
a15_compressed_10=encoder10.predict(a15)
a16_compressed_10=encoder10.predict(a16)
a17_compressed_10=encoder10.predict(a17)
a18_compressed_10=encoder10.predict(a18)
a19_compressed_10=encoder10.predict(a19)
bona_compressed_10=encoder10.predict(bona)

a07_sfocate_compressed_10=encoder10.predict(a07_sfocate_10)
a08_sfocate_compressed_10=encoder10.predict(a08_sfocate_10)
a09_sfocate_compressed_10=encoder10.predict(a09_sfocate_10)
a10_sfocate_compressed_10=encoder10.predict(a10_sfocate_10)
a11_sfocate_compressed_10=encoder10.predict(a11_sfocate_10)
a12_sfocate_compressed_10=encoder10.predict(a12_sfocate_10)
a13_sfocate_compressed_10=encoder10.predict(a13_sfocate_10)
a14_sfocate_compressed_10=encoder10.predict(a14_sfocate_10)
a15_sfocate_compressed_10=encoder10.predict(a15_sfocate_10)
a16_sfocate_compressed_10=encoder10.predict(a16_sfocate_10)
a17_sfocate_compressed_10=encoder10.predict(a17_sfocate_10)
a18_sfocate_compressed_10=encoder10.predict(a18_sfocate_10)
a19_sfocate_compressed_10=encoder10.predict(a19_sfocate_10)
bona_sfocate_compressed_10=encoder10.predict(bona_sfocate_10)


mse_a07_10=np.zeros(len(a07))
mse_a08_10=np.zeros(len(a08))
mse_a09_10=np.zeros(len(a09))
mse_a10_10=np.zeros(len(a10))
mse_a11_10=np.zeros(len(a11))
mse_a12_10=np.zeros(len(a12))
mse_a13_10=np.zeros(len(a13))
mse_a14_10=np.zeros(len(a14))
mse_a15_10=np.zeros(len(a15))
mse_a16_10=np.zeros(len(a16))
mse_a17_10=np.zeros(len(a17))
mse_a18_10=np.zeros(len(a18))
mse_a19_10=np.zeros(len(a19))
mse_bona_10=np.zeros(len(bona))


for i in range(len(a07)):
    mse_a07_10[i] = (np.square(a07_compressed_10[i]
                               - a07_sfocate_compressed_10[i])).mean()

for i in range(len(a08)):
    mse_a08_10[i] = (np.square(a08_compressed_10[i]
                               - a08_sfocate_compressed_10[i])).mean()
for i in range(len(a09)):
    mse_a09_10[i] = (np.square(a09_compressed_10[i]
                               - a09_sfocate_compressed_10[i])).mean()

for i in range(len(a10)):
    mse_a10_10[i] = (np.square(a10_compressed_10[i]
                               - a10_sfocate_compressed_10[i])).mean()

for i in range(len(a11)):
    mse_a11_10[i] = (np.square(a11_compressed_10[i]
                               - a11_sfocate_compressed_10[i])).mean()

for i in range(len(a12)):
    mse_a12_10[i] = (np.square(a12_compressed_10[i]
                               - a12_sfocate_compressed_10[i])).mean()

for i in range(len(a13)):
    mse_a13_10[i] = (np.square(a13_compressed_10[i]
                               - a13_sfocate_compressed_10[i])).mean()


for i in range(len(a14)):
    mse_a14_10[i] = (np.square(a14_compressed_10[i]
                               - a14_sfocate_compressed_10[i])).mean()



for i in range(len(a15)):
    mse_a15_10[i] = (np.square(a15_compressed_10[i] - a15_sfocate_compressed_10[i])).mean()

for i in range(len(a16)):
    mse_a16_10[i] = (np.square(a16_compressed_10[i] - a16_sfocate_compressed_10[i])).mean()

for i in range(len(a17)):
    mse_a17_10[i] = (np.square(a17_compressed_10[i]
                               - a17_sfocate_compressed_10[i])).mean()

for i in range(len(a18)):
    mse_a18_10[i] = (np.square(a18_compressed_10[i]
                               - a18_sfocate_compressed_10[i])).mean()

for i in range(len(a19)):
    mse_a19_10[i] = (np.square(a19_compressed_10[i]
                               - a19_sfocate_compressed_10[i])).mean()

for i in range(len(bona)):
    mse_bona_10[i] = (np.square(bona_compressed_10[i]
                                - bona_sfocate_compressed_10[i])).mean()


a07_sfocate_10=[]
a08_sfocate_10=[]
a09_sfocate_10=[]
a10_sfocate_10=[]
a11_sfocate_10=[]
a12_sfocate_10=[]
a13_sfocate_10=[]
a14_sfocate_10=[]
a15_sfocate_10=[]
a16_sfocate_10=[]
a17_sfocate_10=[]
a18_sfocate_10=[]
a19_sfocate_10=[]
bona_sfocate_10=[]

a07_compressed_10=[]
a08_compressed_10=[]
a09_compressed_10=[]
a10_compressed_10=[]
a11_compressed_10=[]
a12_compressed_10=[]
a13_compressed_10=[]
a14_compressed_10=[]
a15_compressed_10=[]
a16_compressed_10=[]
a17_compressed_10=[]
bona_compressed_10=[]

a07_sfocate_compressed_10=[]
a08_sfocate_compressed_10=[]
a09_sfocate_compressed_10=[]
a10_sfocate_compressed_10=[]
a11_sfocate_compressed_10=[]
a12_sfocate_compressed_10=[]
a13_sfocate_compressed_10=[]
a14_sfocate_compressed_10=[]
a15_sfocate_compressed_10=[]
a16_sfocate_compressed_10=[]
a17_sfocate_compressed_10=[]
a18_sfocate_compressed_10=[]
a19_sfocate_compressed_10=[]
bona_sfocate_compressed_10=[]



# model15

a07_sfocate_15=model15.predict(a07)
a08_sfocate_15=model15.predict(a08)
a09_sfocate_15=model15.predict(a09)
a10_sfocate_15=model15.predict(a10)
a11_sfocate_15=model15.predict(a11)
a12_sfocate_15=model15.predict(a12)
a13_sfocate_15=model15.predict(a13)
a14_sfocate_15=model15.predict(a14)
a15_sfocate_15=model15.predict(a15)
a16_sfocate_15=model15.predict(a16)
a17_sfocate_15=model15.predict(a17)
a18_sfocate_15=model15.predict(a18)
a19_sfocate_15=model15.predict(a19)
bona_sfocate_15=model15.predict(bona)

bona_compressed_15=encoder15.predict(bona)
a07_compressed_15=encoder15.predict(a07)
a08_compressed_15=encoder15.predict(a08)
a09_compressed_15=encoder15.predict(a09)
a10_compressed_15=encoder15.predict(a10)
a11_compressed_15=encoder15.predict(a11)
a12_compressed_15=encoder15.predict(a12)
a13_compressed_15=encoder15.predict(a13)
a14_compressed_15=encoder15.predict(a14)
a15_compressed_15=encoder15.predict(a15)
a16_compressed_15=encoder15.predict(a16)
a17_compressed_15=encoder15.predict(a17)
a18_compressed_15=encoder15.predict(a18)
a19_compressed_15=encoder15.predict(a19)

bona_sfocate_compressed_15=encoder15.predict(bona_sfocate_15)
a07_sfocate_compressed_15=encoder15.predict(a07_sfocate_15)
a08_sfocate_compressed_15=encoder15.predict(a08_sfocate_15)
a09_sfocate_compressed_15=encoder15.predict(a09_sfocate_15)
a10_sfocate_compressed_15=encoder15.predict(a10_sfocate_15)
a11_sfocate_compressed_15=encoder15.predict(a11_sfocate_15)
a12_sfocate_compressed_15=encoder15.predict(a12_sfocate_15)
a13_sfocate_compressed_15=encoder15.predict(a13_sfocate_15)
a14_sfocate_compressed_15=encoder15.predict(a14_sfocate_15)
a15_sfocate_compressed_15=encoder15.predict(a15_sfocate_15)
a16_sfocate_compressed_15=encoder15.predict(a16_sfocate_15)
a17_sfocate_compressed_15=encoder15.predict(a17_sfocate_15)
a18_sfocate_compressed_15=encoder15.predict(a18_sfocate_15)
a19_sfocate_compressed_15=encoder15.predict(a19_sfocate_15)

mse_bona_15=np.zeros(len(bona))
mse_a07_15=np.zeros(len(a07))
mse_a08_15=np.zeros(len(a08))
mse_a09_15=np.zeros(len(a09))
mse_a10_15=np.zeros(len(a10))
mse_a11_15=np.zeros(len(a11))
mse_a12_15=np.zeros(len(a12))
mse_a13_15=np.zeros(len(a13))
mse_a14_15=np.zeros(len(a14))
mse_a15_15=np.zeros(len(a15))
mse_a16_15=np.zeros(len(a16))
mse_a17_15=np.zeros(len(a17))
mse_a18_15=np.zeros(len(a18))
mse_a19_15=np.zeros(len(a19))



for i in range(len(a07)):
    mse_a07_15[i] = (np.square(a07_compressed_15[i]
                               - a07_sfocate_compressed_15[i])).mean()

for i in range(len(a08)):
    mse_a08_15[i] = (np.square(a08_compressed_15[i]
                               - a08_sfocate_compressed_15[i])).mean()

for i in range(len(a09)):
    mse_a09_15[i] = (np.square(a09_compressed_15[i]
                               - a09_sfocate_compressed_15[i])).mean()

for i in range(len(a10)):
    mse_a10_15[i] = (np.square(a10_compressed_15[i]
                               - a10_sfocate_compressed_15[i])).mean()

for i in range(len(a11)):
    mse_a11_15[i] = (np.square(a11_compressed_15[i]
                               - a11_sfocate_compressed_15[i])).mean()

for i in range(len(a12)):
    mse_a12_15[i] = (np.square(a12_compressed_15[i]
                               - a12_sfocate_compressed_15[i])).mean()

for i in range(len(a13)):
    mse_a13_15[i] = (np.square(a13_compressed_15[i]
                               - a13_sfocate_compressed_15[i])).mean()

for i in range(len(a14)):
    mse_a14_15[i] = (np.square(a14_compressed_15[i]
                               - a14_sfocate_compressed_15[i])).mean()

for i in range(len(a15)):
    mse_a15_15[i] = (np.square(a15_compressed_15[i] - a15_sfocate_compressed_15[i])).mean()

for i in range(len(a16)):
    mse_a16_15[i] = (np.square(a16_compressed_15[i] - a16_sfocate_compressed_15[i])).mean()

for i in range(len(a17)):
    mse_a17_15[i] = (np.square(a17_compressed_15[i]
                               - a17_sfocate_compressed_15[i])).mean()

for i in range(len(a18)):
    mse_a18_15[i] = (np.square(a18_compressed_15[i]
                               - a18_sfocate_compressed_15[i])).mean()

for i in range(len(a19)):
    mse_a19_15[i] = (np.square(a19_compressed_15[i]
                               - a19_sfocate_compressed_15[i])).mean()


for i in range(len(bona)):
    mse_bona_15[i] = (np.square(bona_compressed_15[i]
                                - bona_sfocate_compressed_15[i])).mean()

bona_sfocate_15=[]
a07_sfocate_15=[]
a08_sfocate_15=[]
a09_sfocate_15=[]
a10_sfocate_15=[]
a11_sfocate_15=[]
a12_sfocate_15=[]
a13_sfocate_15=[]
a14_sfocate_15=[]
a15_sfocate_15=[]
a16_sfocate_15=[]
a17_sfocate_15=[]
a18_sfocate_15=[]
a19_sfocate_15=[]

bona_compressed_15=[]
a07_compressed_15=[]
a08_compressed_15=[]
a09_compressed_15=[]
a10_compressed_15=[]
a11_compressed_15=[]
a12_compressed_15=[]
a13_compressed_15=[]
a14_compressed_15=[]
a15_compressed_15=[]
a16_compressed_15=[]
a17_compressed_15=[]
a18_compressed_15=[]
a19_compressed_15=[]



bona_sfocate_compressed_15=[]
a07_sfocate_compressed_15=[]
a08_sfocate_compressed_15=[]
a09_sfocate_compressed_15=[]
a10_sfocate_compressed_15=[]
a11_sfocate_compressed_15=[]
a12_sfocate_compressed_15=[]
a13_sfocate_compressed_15=[]
a14_sfocate_compressed_15=[]
a15_sfocate_compressed_15=[]
a16_sfocate_compressed_15=[]
a17_sfocate_compressed_15=[]
a18_sfocate_compressed_15=[]
a19_sfocate_compressed_15=[]




# model16

bona_sfocate_16=model16.predict(bona)
a07_sfocate_16=model16.predict(a07)
a08_sfocate_16=model16.predict(a08)
a09_sfocate_16=model16.predict(a09)
a10_sfocate_16=model16.predict(a10)
a11_sfocate_16=model16.predict(a11)
a12_sfocate_16=model16.predict(a12)
a13_sfocate_16=model16.predict(a13)
a14_sfocate_16=model16.predict(a14)
a15_sfocate_16=model16.predict(a15)
a16_sfocate_16=model16.predict(a16)
a17_sfocate_16=model16.predict(a17)
a18_sfocate_16=model16.predict(a18)
a19_sfocate_16=model16.predict(a19)

bona_compressed_16=encoder16.predict(bona)
a07_compressed_16=encoder16.predict(a07)
a08_compressed_16=encoder16.predict(a08)
a09_compressed_16=encoder16.predict(a09)
a10_compressed_16=encoder16.predict(a10)
a11_compressed_16=encoder16.predict(a11)
a12_compressed_16=encoder16.predict(a12)
a13_compressed_16=encoder16.predict(a13)
a14_compressed_16=encoder16.predict(a14)
a15_compressed_16=encoder16.predict(a15)
a16_compressed_16=encoder16.predict(a16)
a17_compressed_16=encoder16.predict(a17)
a18_compressed_16=encoder16.predict(a18)
a19_compressed_16=encoder16.predict(a19)


bona_sfocate_compressed_16=encoder16.predict(bona_sfocate_16)
a07_sfocate_compressed_16=encoder16.predict(a07_sfocate_16)
a08_sfocate_compressed_16=encoder16.predict(a08_sfocate_16)
a09_sfocate_compressed_16=encoder16.predict(a09_sfocate_16)
a10_sfocate_compressed_16=encoder16.predict(a10_sfocate_16)
a11_sfocate_compressed_16=encoder16.predict(a11_sfocate_16)
a12_sfocate_compressed_16=encoder16.predict(a12_sfocate_16)
a13_sfocate_compressed_16=encoder16.predict(a13_sfocate_16)
a14_sfocate_compressed_16=encoder16.predict(a14_sfocate_16)
a15_sfocate_compressed_16=encoder16.predict(a15_sfocate_16)
a16_sfocate_compressed_16=encoder16.predict(a16_sfocate_16)
a17_sfocate_compressed_16=encoder16.predict(a17_sfocate_16)
a18_sfocate_compressed_16=encoder16.predict(a18_sfocate_16)
a19_sfocate_compressed_16=encoder16.predict(a19_sfocate_16)


mse_bona_16=np.zeros(len(bona))
mse_a07_16=np.zeros(len(a07))
mse_a08_16=np.zeros(len(a08))
mse_a09_16=np.zeros(len(a09))
mse_a10_16=np.zeros(len(a10))
mse_a11_16=np.zeros(len(a11))
mse_a12_16=np.zeros(len(a12))
mse_a13_16=np.zeros(len(a13))
mse_a14_16=np.zeros(len(a14))
mse_a15_16=np.zeros(len(a15))
mse_a16_16=np.zeros(len(a16))
mse_a17_16=np.zeros(len(a17))
mse_a18_16=np.zeros(len(a18))
mse_a19_16=np.zeros(len(a19))



for i in range(len(a07)):
    mse_a07_16[i] = (np.square(a07_compressed_16[i]
                               - a07_sfocate_compressed_16[i])).mean()

for i in range(len(a08)):
    mse_a08_16[i] = (np.square(a08_compressed_16[i]
                               - a08_sfocate_compressed_16[i])).mean()

for i in range(len(a09)):
    mse_a09_16[i] = (np.square(a09_compressed_16[i]
                               - a09_sfocate_compressed_16[i])).mean()

for i in range(len(a10)):
    mse_a10_16[i] = (np.square(a10_compressed_16[i]
                               - a10_sfocate_compressed_16[i])).mean()

for i in range(len(a11)):
    mse_a11_16[i] = (np.square(a11_compressed_16[i]
                               - a11_sfocate_compressed_16[i])).mean()

for i in range(len(a12)):
    mse_a12_16[i] = (np.square(a12_compressed_16[i]
                               - a12_sfocate_compressed_16[i])).mean()

for i in range(len(a13)):
    mse_a13_16[i] = (np.square(a13_compressed_16[i]
                               - a13_sfocate_compressed_16[i])).mean()

for i in range(len(a14)):
    mse_a14_16[i] = (np.square(a14_compressed_16[i]
                               - a14_sfocate_compressed_16[i])).mean()

for i in range(len(a15)):
    mse_a15_16[i] = (np.square(a15_compressed_16[i] - a15_sfocate_compressed_16[i])).mean()

for i in range(len(a16)):
    mse_a16_16[i] = (np.square(a16_compressed_16[i] - a16_sfocate_compressed_16[i])).mean()

for i in range(len(a17)):
    mse_a17_16[i] = (np.square(a17_compressed_16[i]
                               - a17_sfocate_compressed_16[i])).mean()

for i in range(len(a18)):
    mse_a18_16[i] = (np.square(a18_compressed_16[i]
                               - a18_sfocate_compressed_16[i])).mean()

for i in range(len(a19)):
    mse_a19_16[i] = (np.square(a19_compressed_16[i]
                               - a19_sfocate_compressed_16[i])).mean()

for i in range(len(bona)):
    mse_bona_16[i] = (np.square(bona_compressed_16[i]
                                - bona_sfocate_compressed_16[i])).mean()

bona_sfocate_16=[]
a07_sfocate_16=[]
a08_sfocate_16=[]
a09_sfocate_16=[]
a10_sfocate_16=[]
a11_sfocate_16=[]
a12_sfocate_16=[]
a13_sfocate_16=[]
a14_sfocate_16=[]
a15_sfocate_16=[]
a16_sfocate_16=[]
a17_sfocate_16=[]
a18_sfocate_16=[]
a19_sfocate_16=[]

bona_compressed_16=[]
a07_compressed_16=[]
a08_compressed_16=[]
a09_compressed_16=[]
a10_compressed_16=[]
a11_compressed_16=[]
a12_compressed_16=[]
a13_compressed_16=[]
a14_compressed_16=[]
a15_compressed_16=[]
a16_compressed_16=[]
a17_compressed_16=[]
a18_compressed_16=[]
a19_compressed_16=[]

bona_sfocate_compressed_16=[]
a07_sfocate_compressed_16=[]
a08_sfocate_compressed_16=[]
a09_sfocate_compressed_16=[]
a10_sfocate_compressed_16=[]
a11_sfocate_compressed_16=[]
a12_sfocate_compressed_16=[]
a13_sfocate_compressed_16=[]
a14_sfocate_compressed_16=[]
a15_sfocate_compressed_16=[]
a16_sfocate_compressed_16=[]
a17_sfocate_compressed_16=[]
a18_sfocate_compressed_16=[]
a19_sfocate_compressed_16=[]


# model17

bona_sfocate_17=model17.predict(bona)
a07_sfocate_17=model17.predict(a07)
a08_sfocate_17=model17.predict(a08)
a09_sfocate_17=model17.predict(a09)
a10_sfocate_17=model17.predict(a10)
a11_sfocate_17=model17.predict(a11)
a12_sfocate_17=model17.predict(a12)
a13_sfocate_17=model17.predict(a13)
a14_sfocate_17=model17.predict(a14)
a15_sfocate_17=model17.predict(a15)
a16_sfocate_17=model17.predict(a16)
a17_sfocate_17=model17.predict(a17)
a18_sfocate_17=model17.predict(a18)
a19_sfocate_17=model17.predict(a19)


bona_compressed_17=encoder17.predict(bona)
a07_compressed_17=encoder17.predict(a07)
a08_compressed_17=encoder17.predict(a08)
a09_compressed_17=encoder17.predict(a09)
a10_compressed_17=encoder17.predict(a10)
a11_compressed_17=encoder17.predict(a11)
a12_compressed_17=encoder17.predict(a12)
a13_compressed_17=encoder17.predict(a13)
a14_compressed_17=encoder17.predict(a14)
a15_compressed_17=encoder17.predict(a15)
a16_compressed_17=encoder17.predict(a16)
a17_compressed_17=encoder17.predict(a17)
a18_compressed_17=encoder17.predict(a18)
a19_compressed_17=encoder17.predict(a19)

bona_sfocate_compressed_17=encoder17.predict(bona_sfocate_17)
a07_sfocate_compressed_17=encoder17.predict(a07_sfocate_17)
a08_sfocate_compressed_17=encoder17.predict(a08_sfocate_17)
a09_sfocate_compressed_17=encoder17.predict(a09_sfocate_17)
a10_sfocate_compressed_17=encoder17.predict(a10_sfocate_17)
a11_sfocate_compressed_17=encoder17.predict(a11_sfocate_17)
a12_sfocate_compressed_17=encoder17.predict(a12_sfocate_17)
a13_sfocate_compressed_17=encoder17.predict(a13_sfocate_17)
a14_sfocate_compressed_17=encoder17.predict(a14_sfocate_17)
a15_sfocate_compressed_17=encoder17.predict(a15_sfocate_17)
a16_sfocate_compressed_17=encoder17.predict(a16_sfocate_17)
a17_sfocate_compressed_17=encoder17.predict(a17_sfocate_17)
a18_sfocate_compressed_17=encoder17.predict(a18_sfocate_17)
a19_sfocate_compressed_17=encoder17.predict(a19_sfocate_17)

mse_bona_17=np.zeros(len(bona))
mse_a07_17=np.zeros(len(a07))
mse_a08_17=np.zeros(len(a08))
mse_a09_17=np.zeros(len(a09))
mse_a10_17=np.zeros(len(a10))
mse_a11_17=np.zeros(len(a11))
mse_a12_17=np.zeros(len(a12))
mse_a13_17=np.zeros(len(a13))
mse_a14_17=np.zeros(len(a14))
mse_a15_17=np.zeros(len(a15))
mse_a16_17=np.zeros(len(a16))
mse_a17_17=np.zeros(len(a17))
mse_a18_17=np.zeros(len(a18))
mse_a19_17=np.zeros(len(a19))



for i in range(len(a07)):
    mse_a07_17[i] = (np.square(a07_compressed_17[i]
                               - a07_sfocate_compressed_17[i])).mean()

for i in range(len(a08)):
    mse_a08_17[i] = (np.square(a08_compressed_17[i]
                               - a08_sfocate_compressed_17[i])).mean()

for i in range(len(a09)):
    mse_a09_17[i] = (np.square(a09_compressed_17[i]
                               - a09_sfocate_compressed_17[i])).mean()

for i in range(len(a10)):
    mse_a10_17[i] = (np.square(a10_compressed_17[i]
                               - a10_sfocate_compressed_17[i])).mean()

for i in range(len(a11)):
    mse_a11_17[i] = (np.square(a11_compressed_17[i]
                               - a11_sfocate_compressed_17[i])).mean()

for i in range(len(a12)):
    mse_a12_17[i] = (np.square(a12_compressed_17[i]
                               - a12_sfocate_compressed_17[i])).mean()

for i in range(len(a13)):
    mse_a13_17[i] = (np.square(a13_compressed_17[i]
                               - a13_sfocate_compressed_17[i])).mean()

for i in range(len(a14)):
    mse_a14_17[i] = (np.square(a14_compressed_17[i]
                               - a14_sfocate_compressed_17[i])).mean()

for i in range(len(a15)):
    mse_a15_17[i] = (np.square(a15_compressed_17[i] - a15_sfocate_compressed_17[i])).mean()

for i in range(len(a16)):
    mse_a16_17[i] = (np.square(a16_compressed_17[i] - a16_sfocate_compressed_17[i])).mean()

for i in range(len(a17)):
    mse_a17_17[i] = (np.square(a17_compressed_17[i]
                               - a17_sfocate_compressed_17[i])).mean()

for i in range(len(a18)):
    mse_a18_17[i] = (np.square(a18_compressed_17[i]
                               - a18_sfocate_compressed_17[i])).mean()

for i in range(len(a19)):
    mse_a19_17[i] = (np.square(a19_compressed_17[i]
                               - a19_sfocate_compressed_17[i])).mean()

for i in range(len(bona)):
    mse_bona_17[i] = (np.square(bona_compressed_17[i]
                                - bona_sfocate_compressed_17[i])).mean()


bona_sfocate_17=[]
a07_sfocate_17=[]
a08_sfocate_17=[]
a09_sfocate_17=[]
a10_sfocate_17=[]
a11_sfocate_17=[]
a12_sfocate_17=[]
a13_sfocate_17=[]
a14_sfocate_17=[]
a15_sfocate_17=[]
a16_sfocate_17=[]
a17_sfocate_17=[]
a18_sfocate_17=[]
a19_sfocate_17=[]

bona_compressed_17=[]
a07_compressed_17=[]
a08_compressed_17=[]
a09_compressed_17=[]
a10_compressed_17=[]
a11_compressed_17=[]
a12_compressed_17=[]
a13_compressed_17=[]
a14_compressed_17=[]
a15_compressed_17=[]
a16_compressed_17=[]
a17_compressed_17=[]
a18_compressed_17=[]
a19_compressed_17=[]

bona_sfocate_compressed_17=[]
a07_sfocate_compressed_17=[]
a08_sfocate_compressed_17=[]
a09_sfocate_compressed_17=[]
a10_sfocate_compressed_17=[]
a11_sfocate_compressed_17=[]
a12_sfocate_compressed_17=[]
a13_sfocate_compressed_17=[]
a14_sfocate_compressed_17=[]
a15_sfocate_compressed_17=[]
a16_sfocate_compressed_17=[]
a17_sfocate_compressed_17=[]
a18_sfocate_compressed_17=[]
a19_sfocate_compressed_17=[]




features_bona=np.zeros((len(bona), 5))
features_a07=np.zeros((len(a07), 5))
features_a08=np.zeros((len(a08), 5))
features_a09=np.zeros((len(a09), 5))
features_a10=np.zeros((len(a10), 5))
features_a11=np.zeros((len(a11), 5))
features_a12=np.zeros((len(a12), 5))
features_a13=np.zeros((len(a13), 5))
features_a14=np.zeros((len(a14), 5))
features_a15=np.zeros((len(a15), 5))
features_a16=np.zeros((len(a16), 5))
features_a17=np.zeros((len(a17), 5))
features_a18=np.zeros((len(a18), 5))
features_a19=np.zeros((len(a19), 5))


for j in range(len(a07)):
    features_a07[j, 0] = mse_a07_07[j]
    features_a07[j, 1] = mse_a07_10[j]
    features_a07[j, 2] = mse_a07_15[j]
    features_a07[j, 3] = mse_a07_16[j]
    features_a07[j, 4] = mse_a07_17[j]

for j in range(len(a08)):
    features_a08[j, 0] = mse_a08_07[j]
    features_a08[j, 1] = mse_a08_10[j]
    features_a08[j, 2] = mse_a08_15[j]
    features_a08[j, 3] = mse_a08_16[j]
    features_a08[j, 4] = mse_a08_17[j]

for j in range(len(a09)):
    features_a09[j, 0] = mse_a09_07[j]
    features_a09[j, 1] = mse_a09_10[j]
    features_a09[j, 2] = mse_a09_15[j]
    features_a09[j, 3] = mse_a09_16[j]
    features_a09[j, 4] = mse_a09_17[j]


for j in range(len(a10)):
    features_a10[j, 0] = mse_a10_07[j]
    features_a10[j, 1] = mse_a10_10[j]
    features_a10[j, 2] = mse_a10_15[j]
    features_a10[j, 3] = mse_a10_16[j]
    features_a10[j, 4] = mse_a10_17[j]

for j in range(len(a11)):
    features_a11[j, 0] = mse_a11_07[j]
    features_a11[j, 1] = mse_a11_10[j]
    features_a11[j, 2] = mse_a11_15[j]
    features_a11[j, 3] = mse_a11_16[j]
    features_a11[j, 4] = mse_a11_17[j]

for j in range(len(a12)):
    features_a12[j, 0] = mse_a12_07[j]
    features_a12[j, 1] = mse_a12_10[j]
    features_a12[j, 2] = mse_a12_15[j]
    features_a12[j, 3] = mse_a12_16[j]
    features_a12[j, 4] = mse_a12_17[j]

for j in range(len(a13)):
    features_a13[j, 0] = mse_a13_07[j]
    features_a13[j, 1] = mse_a13_10[j]
    features_a13[j, 2] = mse_a13_15[j]
    features_a13[j, 3] = mse_a13_16[j]
    features_a13[j, 4] = mse_a13_17[j]

for j in range(len(a14)):
    features_a14[j, 0] = mse_a14_07[j]
    features_a14[j, 1] = mse_a14_10[j]
    features_a14[j, 2] = mse_a14_15[j]
    features_a14[j, 3] = mse_a14_16[j]
    features_a14[j, 4] = mse_a14_17[j]


for j in range(len(a15)):
    features_a15[j, 0] = mse_a15_07[j]
    features_a15[j, 1] = mse_a15_10[j]
    features_a15[j, 2] = mse_a15_15[j]
    features_a15[j, 3] = mse_a15_16[j]
    features_a15[j, 4] = mse_a15_17[j]



for j in range(len(a16)):
    features_a16[j, 0] = mse_a16_07[j]
    features_a16[j, 1] = mse_a16_10[j]
    features_a16[j, 2] = mse_a16_15[j]
    features_a16[j, 3] = mse_a16_16[j]
    features_a16[j, 4] = mse_a16_17[j]


for j in range(len(a17)):
    features_a17[j, 0] = mse_a17_07[j]
    features_a17[j, 1] = mse_a17_10[j]
    features_a17[j, 2] = mse_a17_15[j]
    features_a17[j, 3] = mse_a17_16[j]
    features_a17[j, 4] = mse_a17_17[j]

for j in range(len(a18)):
    features_a18[j, 0] = mse_a18_07[j]
    features_a18[j, 1] = mse_a18_10[j]
    features_a18[j, 2] = mse_a18_15[j]
    features_a18[j, 3] = mse_a18_16[j]
    features_a18[j, 4] = mse_a18_17[j]

for j in range(len(a19)):
    features_a19[j, 0] = mse_a19_07[j]
    features_a19[j, 1] = mse_a19_10[j]
    features_a19[j, 2] = mse_a19_15[j]
    features_a19[j, 3] = mse_a19_16[j]
    features_a19[j, 4] = mse_a19_17[j]


for j in range(len(bona)):
    features_bona[j, 0] = mse_bona_07[j]
    features_bona[j, 1] = mse_bona_10[j]
    features_bona[j, 2] = mse_bona_15[j]
    features_bona[j, 3] = mse_bona_16[j]
    features_bona[j, 4] = mse_bona_17[j]


np.save('features_a07.npy', features_a07)
np.save('features_a10.npy', features_a10)
np.save('features_a15.npy', features_a15)
np.save('features_a16.npy', features_a16)
np.save('features_a17.npy', features_a17)
np.save('features_bona.npy', features_bona)

np.save('features_a08.npy', features_a08)
np.save('features_a09.npy', features_a09)
np.save('features_a11.npy', features_a11)
np.save('features_a12.npy', features_a12)
np.save('features_a13.npy', features_a13)
np.save('features_a14.npy', features_a14)
np.save('features_a18.npy', features_a18)
np.save('features_a19.npy', features_a19)



