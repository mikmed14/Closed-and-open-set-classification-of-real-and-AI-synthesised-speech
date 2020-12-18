import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
import pandas as pd


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


bona=np.abs(bona_asv)
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





# bonafide vs all fake

train_images=np.concatenate((a07[0:270], a08[0:270], a09[0:270], a10[0:270],
                        a11[0:270], a12[0:270], a13[0:270], a14[0:270],
                        a15[0:270], a16[0:270], a17[0:270], a18[0:270],
                        a19[0:270], bona[0:3510]))

train_labels=np.zeros(len(train_images))

for i in range(270*13):
    train_labels[i]=1

vali_images=np.concatenate((a07[270:385], a08[270:385], a09[270:385], a10[270:385],
                        a11[270:385], a12[270:385], a13[270:385], a14[270:385],
                        a15[270:385], a16[270:385], a17[270:385], a18[270:385],
                        a19[270:385], bona[3510:5005]))

vali_labels=np.zeros(len(vali_images))

for i in range(115*13):
    vali_labels[i]=1

test_images=np.concatenate((a07[385:560], a08[385:560], a09[385:560],
                            a10[385:560], a11[385:560], a12[385:560],
                            a13[385:560], a14[385:560], a15[385:560],
                            a16[385:560], a17[385:560], a18[385:560],
                            a19[385:560], bona[5005:7280]))


test_labels=np.zeros(len(test_images))
for i in range(175*13):
    test_labels[i]=1




class_names=['bonafide', 'fake']


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)






model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3),  activation='relu', input_shape=(128, 128, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
#model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
#model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(16, activation='relu'))#64
model.add(keras.layers.Dense(2))

checkpoint = ModelCheckpoint("weigths_cnn_bonavspoof.ckpt", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.00001)

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              #loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=200,
                    validation_data=(vali_images, vali_labels),
                    callbacks=[checkpoint, reduce_lr]

                    )

plt.figure()
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss function")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["training loss", "validation_loss"])

#plt.show()
plt.savefig('neural_loss_bonavspoof.png')

model.load_weights('weigths_cnn_bonavspoof.ckpt')

test_loss, test_acc = model.evaluate(test_images,  test_labels,
                                     verbose=2)


probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])

predictions_proba = probability_model.predict(test_images)
predictions=np.zeros(len(predictions_proba))
for i in range(len(predictions)):
    predictions[i]=np.argmax(predictions_proba[i])


cm=confusion_matrix(test_labels, predictions, normalize='true')

sum=0.0
sum=cm[0,0]+cm[1,1]


accuracy=sum/2

print('accuracy=' , accuracy)


cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, binary classification bonafide vs fake, Neural Networks')
plt.savefig('cm_neural_bonavspoof.png')

# MULTICLASS

train_images=np.concatenate((a07[0:3000], a08[0:3000], a09[0:3000],
                             a10[0:3000], a11[0:3000], a12[0:3000],
                             a13[0:3000], a14[0:3000], a15[0:3000],
                             a16[0:3000], a17[0:3000], a18[0:3000],
                             a19[0:3000], bona[0:3000]))

train_labels_07=np.zeros(3000)
for i in range(len(train_labels_07)):
    train_labels_07[i]=1

train_labels_08=np.zeros(3000)
for i in range(len(train_labels_08)):
    train_labels_08[i]=2

train_labels_09=np.zeros(3000)
for i in range(len(train_labels_09)):
    train_labels_09[i]=3

train_labels_10=np.zeros(3000)
for i in range(len(train_labels_10)):
    train_labels_10[i]=4

train_labels_11=np.zeros(3000)
for i in range(len(train_labels_11)):
    train_labels_11[i]=5

train_labels_12=np.zeros(3000)
for i in range(len(train_labels_12)):
    train_labels_12[i]=6

train_labels_13=np.zeros(3000)
for i in range(len(train_labels_13)):
    train_labels_13[i]=7

train_labels_14=np.zeros(3000)
for i in range(len(train_labels_14)):
    train_labels_14[i]=8

train_labels_15=np.zeros(3000)
for i in range(len(train_labels_15)):
    train_labels_15[i]=9


train_labels_16=np.zeros(3000)
for i in range(len(train_labels_16)):
    train_labels_16[i]=10

train_labels_17=np.zeros(3000)
for i in range(len(train_labels_17)):
    train_labels_17[i]=11

train_labels_18=np.zeros(3000)
for i in range(len(train_labels_18)):
    train_labels_18[i]=12

train_labels_19=np.zeros(3000)
for i in range(len(train_labels_19)):
    train_labels_19[i]=13

train_labels_bona=np.zeros(3000)
for i in range(len(train_labels_bona)):
    train_labels_bona[i]=0

train_labels=np.concatenate((train_labels_07, train_labels_08,
                             train_labels_09, train_labels_10,
                             train_labels_11, train_labels_12,
                             train_labels_13, train_labels_14,
                             train_labels_15, train_labels_16,
                             train_labels_17, train_labels_18,
                             train_labels_19, train_labels_bona))
train_labels=train_labels.astype(int)

vali_images=np.concatenate((a07[3000:4000], a08[3000:4000],
                            a09[3000:4000], a10[3000:4000],
                            a11[3000:4000], a12[3000:4000], a13[3000:4000],
                            a14[3000:4000], a15[3000:4000],
                            a16[3000:4000], a17[3000:4000],
                            a18[3000:4000], a19[3000:4000],
                            bona[3000:4000]))


vali_labels_07=np.zeros(1000)
for i in range(len(vali_labels_07)):
    vali_labels_07[i]=1

vali_labels_08=np.zeros(1000)
for i in range(len(vali_labels_08)):
    vali_labels_08[i]=2

vali_labels_09=np.zeros(1000)
for i in range(len(vali_labels_09)):
    vali_labels_09[i]=3

vali_labels_10=np.zeros(1000)
for i in range(len(vali_labels_10)):
    vali_labels_10[i]=4

vali_labels_11=np.zeros(1000)
for i in range(len(vali_labels_11)):
    vali_labels_11[i]=5

vali_labels_12=np.zeros(1000)
for i in range(len(vali_labels_12)):
    vali_labels_12[i]=6

vali_labels_13=np.zeros(1000)
for i in range(len(vali_labels_13)):
    vali_labels_13[i]=7

vali_labels_14=np.zeros(1000)
for i in range(len(vali_labels_14)):
    vali_labels_14[i]=8

vali_labels_15=np.zeros(1000)
for i in range(len(vali_labels_15)):
    vali_labels_15[i]=9


vali_labels_16=np.zeros(1000)
for i in range(len(vali_labels_16)):
    vali_labels_16[i]=10

vali_labels_17=np.zeros(1000)
for i in range(len(vali_labels_17)):
    vali_labels_17[i]=11

vali_labels_18=np.zeros(1000)
for i in range(len(vali_labels_18)):
    vali_labels_18[i]=12

vali_labels_19=np.zeros(1000)
for i in range(len(vali_labels_19)):
    vali_labels_19[i]=13

vali_labels_bona=np.zeros(1000)
for i in range(len(vali_labels_bona)):
    vali_labels_bona[i]=0

vali_labels=np.concatenate((vali_labels_07, vali_labels_08,
                             vali_labels_09, vali_labels_10,
                             vali_labels_11, vali_labels_12,
                             vali_labels_13, vali_labels_14,
                             vali_labels_15, vali_labels_16,
                             vali_labels_17, vali_labels_18,
                             vali_labels_19, vali_labels_bona))

vali_labels=vali_labels.astype(int)

test_images=np.concatenate((a07[4000:4900], a08[4000:4900],
                            a09[4000:4900], a10[4000:4900],
                            a11[4000:4900], a12[4000:4900],
                            a13[4000:4900], a14[4000:4900],
                            a15[4000:4900], a16[4000:4900],
                            a17[4000:4900], a18[4000:4900],
                            a19[4000:4900], bona[4000:4900]))


k=900

test_labels_07 = np.zeros(k)
for i in range(len(test_labels_07)):
    test_labels_07[i] = 1

test_labels_08 = np.zeros(k)
for i in range(len(test_labels_08)):
    test_labels_08[i] = 2

test_labels_09 = np.zeros(k)
for i in range(len(test_labels_09)):
    test_labels_09[i] = 3

test_labels_10 = np.zeros(k)
for i in range(len(test_labels_10)):
    test_labels_10[i] = 4

test_labels_11 = np.zeros(k)
for i in range(len(test_labels_11)):
    test_labels_11[i] = 5

test_labels_12 = np.zeros(k)
for i in range(len(test_labels_12)):
    test_labels_12[i] = 6

test_labels_13 = np.zeros(k)
for i in range(len(test_labels_13)):
    test_labels_13[i] = 7

test_labels_14 = np.zeros(k)
for i in range(len(test_labels_14)):
    test_labels_14[i] = 8

test_labels_15 = np.zeros(k)
for i in range(len(test_labels_15)):
    test_labels_15[i] = 9

test_labels_16 = np.zeros(k)
for i in range(len(test_labels_16)):
    test_labels_16[i] = 10

test_labels_17 = np.zeros(k)
for i in range(len(test_labels_17)):
    test_labels_17[i] = 11

test_labels_18 = np.zeros(k)
for i in range(len(test_labels_18)):
    test_labels_18[i] = 12

test_labels_19 = np.zeros(k)
for i in range(len(test_labels_19)):
    test_labels_19[i] = 13

test_labels_bona = np.zeros(k)
for i in range(len(test_labels_bona)):
    test_labels_bona[i] = 0



test_labels=np.concatenate((test_labels_07, test_labels_08,
                            test_labels_09, test_labels_10,
                            test_labels_11, test_labels_12,
                            test_labels_13, test_labels_14,
                            test_labels_15, test_labels_16,
                            test_labels_17, test_labels_18,
                            test_labels_19, test_labels_bona))

test_labels=test_labels.astype(int)

class_names=['bonafide', 'a07', 'a08', 'a09', 'a10', 'a11', 'a12',
             'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19']


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3),  activation='relu', input_shape=(128, 128, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
#model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
#model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(14))

checkpoint = ModelCheckpoint("weigths_cnn_multi.ckpt", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.00001)

model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=200,
                    validation_data=(vali_images, vali_labels),
                    callbacks=[checkpoint, reduce_lr]

                    )

plt.figure()
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss function")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["training loss", "validation_loss"])


plt.savefig('neural_loss_multi.png')



model.load_weights('weigths_cnn_multi.ckpt')

test_loss, test_acc = model.evaluate(test_images,  test_labels,
                                     verbose=2)


probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])

predictions_proba = probability_model.predict(test_images)
predictions=np.zeros(len(predictions_proba))
for i in range(len(predictions)):
    predictions[i]=np.argmax(predictions_proba[i])


cm=confusion_matrix(test_labels, predictions, normalize='true')

#sum=cm[0,0]+cm[1,1]
sum=0
for i in range(14):
    sum=sum+cm[i,i]


accuracy=sum/14

print('accuracy=' , accuracy)

cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(15,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, Multilabel classification, Neural Networks')
plt.savefig('cm_neural_multi.png')





# binary classification bonafide vs each one of the fakes

train_images=np.concatenate((a19[0:3000], bona[0:3000]))
train_labels=np.zeros(len(train_images))
for i in range(3000):
    train_labels[i]=1

vali_images=np.concatenate((a19[3000:4000], bona[3000:4000]))
vali_labels=np.zeros(len(vali_images))
for i in range(1000):
    vali_labels[i]=1

test_images=np.concatenate((a19[4000:4900], bona[4000:4900]))
test_labels=np.zeros(len(test_images))
for i in range(900):
    test_labels[i]=1


class_names=['bonafide', 'a19']

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3),  activation='relu', input_shape=(128, 128, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(2))



model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              #loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=75,
                    validation_data=(vali_images, vali_labels),
                    shuffle=True
                    )
'''
plt.figure()
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss function")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["training loss", "validation_loss"])

#plt.show()
plt.savefig('neural_loss_a16.png')
'''

test_loss, test_acc = model.evaluate(test_images,  test_labels,
                                     verbose=2)


probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])

predictions_proba = probability_model.predict(test_images)
predictions=np.zeros(len(predictions_proba))
for i in range(len(predictions)):
    predictions[i]=np.argmax(predictions_proba[i])


cm=confusion_matrix(test_labels, predictions, normalize='true')

sum=cm[0,0]+cm[1,1]


accuracy=sum/2

print('accuracy=' , accuracy)

cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, Binary classification bonafide vs a19, Neural Networks')
plt.savefig('cm_neural_a19.png')


