import scipy as sp
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


a07_bic = np.load('npy_arrays/asv_unet_128/a07_bic_128.npy')
a08_bic = np.load('npy_arrays/asv_unet_128/a08_bic_128.npy')
a09_bic = np.load('npy_arrays/asv_unet_128/a09_bic_128.npy')
a10_bic = np.load('npy_arrays/asv_unet_128/a10_bic_128.npy')
#a11_bic = np.load('npy_arrays/asv_unet_128/a11_bic_128.npy')
#a12_bic = np.load('npy_arrays/asv_unet_128/a12_bic_128.npy')
a11_bic = np.load('a11_bic_128.npy')
a12_bic = np.load('a12_bic_128.npy')
a13_bic = np.load('npy_arrays/asv_unet_128/a13_bic_128.npy')
a14_bic = np.load('npy_arrays/asv_unet_128/a14_bic_128.npy')
a15_bic = np.load('npy_arrays/asv_unet_128/a15_bic_128.npy')
a16_bic = np.load('npy_arrays/asv_unet_128/a16_bic_128.npy')
a17_bic = np.load('npy_arrays/asv_unet_128/a17_bic_128.npy')
a18_bic = np.load('npy_arrays/asv_unet_128/a18_bic_128.npy')
a19_bic = np.load('npy_arrays/asv_unet_128/a19_bic_128.npy')
#bona_asv = np.load('npy_arrays/asv_unet_128/auto_training_bic.npy')
bona_asv=np.load('bona_bic_eval_128.npy')


bona_asv_modules=np.abs(bona_asv)
a07_modules=np.abs(a07_bic)
a08_modules=np.abs(a08_bic)
a09_modules=np.abs(a09_bic)
a10_modules=np.abs(a10_bic)
a11_modules=np.abs(a11_bic)
a12_modules=np.abs(a12_bic)
a13_modules=np.abs(a13_bic)
a14_modules=np.abs(a14_bic)
a15_modules=np.abs(a15_bic)
a16_modules=np.abs(a16_bic)
a17_modules=np.abs(a17_bic)
a18_modules=np.abs(a18_bic)
a19_modules=np.abs(a19_bic)


bona_asv_phases=np.angle(bona_asv)
a07_phases=np.angle(a07_bic)
a08_phases=np.angle(a08_bic)
a09_phases=np.angle(a09_bic)
a10_phases=np.angle(a10_bic)
a11_phases=np.angle(a11_bic)
a12_phases=np.angle(a12_bic)
a13_phases=np.angle(a13_bic)
a14_phases=np.angle(a14_bic)
a15_phases=np.angle(a15_bic)
a16_phases=np.angle(a16_bic)
a17_phases=np.angle(a17_bic)
a18_phases=np.angle(a18_bic)
a19_phases=np.angle(a19_bic)

bona_asv_phases=(bona_asv_phases+np.pi)/(2*np.pi)
a07_phases=(a07_phases+np.pi)/(2*np.pi)
a08_phases=(a08_phases+np.pi)/(2*np.pi)
a09_phases=(a09_phases+np.pi)/(2*np.pi)
a10_phases=(a10_phases+np.pi)/(2*np.pi)
a11_phases=(a11_phases+np.pi)/(2*np.pi)
a12_phases=(a12_phases+np.pi)/(2*np.pi)
a13_phases=(a13_phases+np.pi)/(2*np.pi)
a14_phases=(a14_phases+np.pi)/(2*np.pi)
a15_phases=(a15_phases+np.pi)/(2*np.pi)
a16_phases=(a16_phases+np.pi)/(2*np.pi)
a17_phases=(a17_phases+np.pi)/(2*np.pi)
a18_phases=(a18_phases+np.pi)/(2*np.pi)
a19_phases=(a19_phases+np.pi)/(2*np.pi)


features_bona_asv=np.zeros((len(bona_asv), 8))
features_a07=np.zeros((len(a07_bic), 8))
features_a08=np.zeros((len(a08_bic), 8))
features_a09=np.zeros((len(a09_bic), 8))
features_a10=np.zeros((len(a10_bic), 8))
features_a11=np.zeros((len(a11_bic), 8))
features_a12=np.zeros((len(a12_bic), 8))
features_a13=np.zeros((len(a13_bic), 8))
features_a14=np.zeros((len(a14_bic), 8))
features_a15=np.zeros((len(a15_bic), 8))
features_a16=np.zeros((len(a16_bic), 8))
features_a17=np.zeros((len(a17_bic), 8))
features_a18=np.zeros((len(a18_bic), 8))
features_a19=np.zeros((len(a19_bic), 8))



for i in range(len(bona_asv_modules)):

    features_bona_asv[i, 0]=bona_asv_modules[i].mean()
    features_bona_asv[i, 1] =np.var(bona_asv_modules[i])
    features_bona_asv[i, 2] =skew(bona_asv_modules[i], None)
    features_bona_asv[i, 3] =kurtosis(bona_asv_modules[i], None)
    features_bona_asv[i, 4] =bona_asv_phases[i].mean()
    features_bona_asv[i, 5] =np.var(bona_asv_phases[i])
    features_bona_asv[i, 6] =skew(bona_asv_phases[i], None)
    features_bona_asv[i, 7] =kurtosis(bona_asv_phases[i], None)

for i in range(len(a07_modules)):
    features_a07[i, 0] = a07_modules[i].mean()
    features_a07[i, 1] = np.var(a07_modules[i])
    features_a07[i, 2] = skew(a07_modules[i], None)
    features_a07[i, 3] = kurtosis(a07_modules[i], None)
    features_a07[i, 4] = a07_phases[i].mean()
    features_a07[i, 5] = np.var(a07_phases[i])
    features_a07[i, 6] = skew(a07_phases[i], None)
    features_a07[i, 7] = kurtosis(a07_phases[i], None)



for i in range(len(a08_modules)):
    features_a08[i, 0] = a08_modules[i].mean()
    features_a08[i, 1] = np.var(a08_modules[i])
    features_a08[i, 2] = skew(a08_modules[i], None)
    features_a08[i, 3] = kurtosis(a08_modules[i], None)
    features_a08[i, 4] = a08_phases[i].mean()
    features_a08[i, 5] = np.var(a08_phases[i])
    features_a08[i, 6] = skew(a08_phases[i], None)
    features_a08[i, 7] = kurtosis(a08_phases[i], None)


for i in range(len(a09_modules)):
    features_a09[i, 0] = a09_modules[i].mean()
    features_a09[i, 1] = np.var(a09_modules[i])
    features_a09[i, 2] = skew(a09_modules[i], None)
    features_a09[i, 3] = kurtosis(a09_modules[i], None)
    features_a09[i, 4] = a09_phases[i].mean()
    features_a09[i, 5] = np.var(a09_phases[i])
    features_a09[i, 6] = skew(a09_phases[i], None)
    features_a09[i, 7] = kurtosis(a09_phases[i], None)




for i in range(len(a10_modules)):
    features_a10[i, 0] = a10_modules[i].mean()
    features_a10[i, 1] = np.var(a10_modules[i])
    features_a10[i, 2] = skew(a10_modules[i], None)
    features_a10[i, 3] = kurtosis(a10_modules[i], None)
    features_a10[i, 4] = a10_phases[i].mean()
    features_a10[i, 5] = np.var(a10_phases[i])
    features_a10[i, 6] = skew(a10_phases[i], None)
    features_a10[i, 7] = kurtosis(a10_phases[i], None)




for i in range(len(a11_modules)):
    features_a11[i, 0] = a11_modules[i].mean()
    features_a11[i, 1] = np.var(a11_modules[i])
    features_a11[i, 2] = skew(a11_modules[i], None)
    features_a11[i, 3] = kurtosis(a11_modules[i], None)
    features_a11[i, 4] = a11_phases[i].mean()
    features_a11[i, 5] = np.var(a11_phases[i])
    features_a11[i, 6] = skew(a11_phases[i], None)
    features_a11[i, 7] = kurtosis(a11_phases[i], None)



for i in range(len(a12_modules)):
    features_a12[i, 0] = a12_modules[i].mean()
    features_a12[i, 1] = np.var(a12_modules[i])
    features_a12[i, 2] = skew(a12_modules[i], None)
    features_a12[i, 3] = kurtosis(a12_modules[i], None)
    features_a12[i, 4] = a12_phases[i].mean()
    features_a12[i, 5] = np.var(a12_phases[i])
    features_a12[i, 6] = skew(a12_phases[i], None)
    features_a12[i, 7] = kurtosis(a12_phases[i], None)



for i in range(len(a13_modules)):
    features_a13[i, 0] = a13_modules[i].mean()
    features_a13[i, 1] = np.var(a13_modules[i])
    features_a13[i, 2] = skew(a13_modules[i], None)
    features_a13[i, 3] = kurtosis(a13_modules[i], None)
    features_a13[i, 4] = a13_phases[i].mean()
    features_a13[i, 5] = np.var(a13_phases[i])
    features_a13[i, 6] = skew(a13_phases[i], None)
    features_a13[i, 7] = kurtosis(a13_phases[i], None)


for i in range(len(a14_modules)):
    features_a14[i, 0] = a14_modules[i].mean()
    features_a14[i, 1] = np.var(a14_modules[i])
    features_a14[i, 2] = skew(a14_modules[i], None)
    features_a14[i, 3] = kurtosis(a14_modules[i], None)
    features_a14[i, 4] = a14_phases[i].mean()
    features_a14[i, 5] = np.var(a14_phases[i])
    features_a14[i, 6] = skew(a14_phases[i], None)
    features_a14[i, 7] = kurtosis(a14_phases[i], None)


for i in range(len(a15_modules)):
    features_a15[i, 0] = a15_modules[i].mean()
    features_a15[i, 1] = np.var(a15_modules[i])
    features_a15[i, 2] = skew(a15_modules[i], None)
    features_a15[i, 3] = kurtosis(a15_modules[i], None)
    features_a15[i, 4] = a15_phases[i].mean()
    features_a15[i, 5] = np.var(a15_phases[i])
    features_a15[i, 6] = skew(a15_phases[i], None)
    features_a15[i, 7] = kurtosis(a15_phases[i], None)



for i in range(len(a16_modules)):
    features_a16[i, 0] = a16_modules[i].mean()
    features_a16[i, 1] = np.var(a16_modules[i])
    features_a16[i, 2] = skew(a16_modules[i], None)
    features_a16[i, 3] = kurtosis(a16_modules[i], None)
    features_a16[i, 4] = a16_phases[i].mean()
    features_a16[i, 5] = np.var(a16_phases[i])
    features_a16[i, 6] = skew(a16_phases[i], None)
    features_a16[i, 7] = kurtosis(a16_phases[i], None)


for i in range(len(a17_modules)):
    features_a17[i, 0] = a17_modules[i].mean()
    features_a17[i, 1] = np.var(a17_modules[i])
    features_a17[i, 2] = skew(a17_modules[i], None)
    features_a17[i, 3] = kurtosis(a17_modules[i], None)
    features_a17[i, 4] = a17_phases[i].mean()
    features_a17[i, 5] = np.var(a17_phases[i])
    features_a17[i, 6] = skew(a17_phases[i], None)
    features_a17[i, 7] = kurtosis(a17_phases[i], None)



for i in range(len(a18_modules)):
    features_a18[i, 0] = a18_modules[i].mean()
    features_a18[i, 1] = np.var(a18_modules[i])
    features_a18[i, 2] = skew(a18_modules[i], None)
    features_a18[i, 3] = kurtosis(a18_modules[i], None)
    features_a18[i, 4] = a18_phases[i].mean()
    features_a18[i, 5] = np.var(a18_phases[i])
    features_a18[i, 6] = skew(a18_phases[i], None)
    features_a18[i, 7] = kurtosis(a18_phases[i], None)



for i in range(len(a19_modules)):
    features_a19[i, 0] = a19_modules[i].mean()
    features_a19[i, 1] = np.var(a19_modules[i])
    features_a19[i, 2] = skew(a19_modules[i], None)
    features_a19[i, 3] = kurtosis(a19_modules[i], None)
    features_a19[i, 4] = a19_phases[i].mean()
    features_a19[i, 5] = np.var(a19_phases[i])
    features_a19[i, 6] = skew(a19_phases[i], None)
    features_a19[i, 7] = kurtosis(a19_phases[i], None)




np.save('faric_features_bona', features_bona_asv)
np.save('faric_features_a07', features_a07)
np.save('faric_features_a08', features_a08)
np.save('faric_features_a09', features_a09)
np.save('faric_features_a10', features_a10)
np.save('faric_features_a11', features_a11)
np.save('faric_features_a12', features_a12)
np.save('faric_features_a13', features_a13)
np.save('faric_features_a14', features_a14)
np.save('faric_features_a15', features_a15)
np.save('faric_features_a16', features_a16)
np.save('faric_features_a17', features_a17)
np.save('faric_features_a18', features_a18)
np.save('faric_features_a19', features_a19)

features_bona_asv=np.load('faric_features_bona.npy')
features_a07=np.load('faric_features_a07.npy')
features_a08=np.load('faric_features_a08.npy')
features_a09=np.load('faric_features_a09.npy')
features_a10=np.load('faric_features_a10.npy')
features_a11=np.load('faric_features_a11.npy')
features_a12=np.load('faric_features_a12.npy')
features_a13=np.load('faric_features_a13.npy')
features_a14=np.load('faric_features_a14.npy')
features_a15=np.load('faric_features_a15.npy')
features_a16=np.load('faric_features_a16.npy')
features_a17=np.load('faric_features_a17.npy')
features_a18=np.load('faric_features_a18.npy')
features_a19=np.load('faric_features_a19.npy')

#global classification



train_features=np.concatenate((features_a07[0:380],
                               features_a08[0:380],
                               features_a09[0:380],
                               features_a10[0:380],
                               features_a11[0:380],
                               features_a12[0:380],
                               features_a13[0:380],
                               features_a14[0:380],
                               features_a15[0:380],
                               features_a16[0:380],
                               features_a17[0:380],
                               features_a18[0:380],
                               features_a19[0:380],
                               features_bona_asv[0:5000]))

train_labels=np.zeros(len(train_features))

for i in range(380*13):
    train_labels[i]=1


test_features=np.concatenate((features_a07[380:560],
                              features_a08[380:560],
                              features_a09[380:560],
                              features_a10[380:560],
                              features_a11[380:560],
                              features_a12[380:560],
                              features_a13[380:560],
                              features_a14[380:560],
                              features_a15[380:560],
                              features_a16[380:560],
                              features_a17[380:560],
                              features_a18[380:560],
                              features_a19[380:560],
                              features_bona_asv[5000:7000]))
test_labels=np.zeros(len(test_features))
for i in range(180*13):
    test_labels[i]=1

model=svm.SVC()
model.fit(train_features, train_labels)
predictions=model.predict(test_features)

class_names=['bonafide', 'fake']

disp = plot_confusion_matrix(model, test_features, test_labels,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=None)
disp.ax_.set_title('Confusion matrix')

#print(title)
#print(disp.confusion_matrix)

plt.savefig('conf_mat_svm.png')



true_positive = 0.0
true_negative = 0.0
false_positive = 0.0
false_negative = 0.0

for i in range(0, len(test_labels)):
    if test_labels[i] == 0 and predictions[i] == test_labels[i]:
        true_negative = true_negative + 1
    if test_labels[i] == 1 and predictions[i] == test_labels[i]:
        true_positive = true_positive + 1
    if test_labels[i] == 0 and predictions[i] != test_labels[i]:
        false_negative = false_negative + 1
    if test_labels[i] == 1 and predictions[i] != test_labels[i]:
        false_positive = false_positive + 1

print('\ntrue negative:', true_negative)
print('\ntrue positive:', true_positive)
print('\nfalse negative:', false_negative)
print('\nfalse positive:', false_positive)


accuracy=(true_positive+true_negative)/(len(test_features))
print('accuracy:', accuracy)



######################################################################


# logistic regression bonafide
a=385
b=230

train_features_bona=np.concatenate((features_a07[0:a], features_a08[0:a],
                                    features_a09[0:a], features_a10[0:a],
                                    features_a11[0:a], features_a12[0:a],
                                    features_a13[0:a], features_a14[0:a],
                                    features_a15[0:a], features_a16[0:a],
                                    features_a17[0:a], features_a18[0:a],
                                    features_a19[0:a], features_bona_asv[0:5000] ))
train_labels_bona=np.zeros(len(train_features_bona))
for i in range(a*13):
    train_labels_bona[i]=1



model_bona=LogisticRegression(random_state=0, max_iter=10000)
model_bona.fit(train_features_bona, train_labels_bona)

# logistic regression a07
train_features_a07=np.concatenate((features_bona_asv[0:b], features_a08[0:b],
                                    features_a09[0:b], features_a10[0:b],
                                    features_a11[0:b], features_a12[0:b],
                                    features_a13[0:b], features_a14[0:b],
                                    features_a15[0:b], features_a16[0:b],
                                    features_a17[0:b], features_a18[0:b],
                                    features_a19[0:b], features_a07[0:3000] ))

train_labels_a07=np.zeros(len(train_features_a07))
for i in range(b*13):
    train_labels_a07[i]=1

model_a07=LogisticRegression(random_state=0, max_iter=10000)
model_a07.fit(train_features_a07, train_labels_a07)


# logistic regression a08
train_features_a08=np.concatenate((features_bona_asv[0:b], features_a07[0:b],
                                    features_a09[0:b], features_a10[0:b],
                                    features_a11[0:b], features_a12[0:b],
                                    features_a13[0:b], features_a14[0:b],
                                    features_a15[0:b], features_a16[0:b],
                                    features_a17[0:b], features_a18[0:b],
                                    features_a19[0:b], features_a08[0:3000] ))

train_labels_a08=np.zeros(len(train_features_a08))
for i in range(b*13):
    train_labels_a08[i]=1

model_a08=LogisticRegression(random_state=0, max_iter=10000)
model_a08.fit(train_features_a08, train_labels_a08)


# logistic regression a09
train_features_a09=np.concatenate((features_bona_asv[0:b], features_a07[0:b],
                                    features_a08[0:b], features_a10[0:b],
                                    features_a11[0:b], features_a12[0:b],
                                    features_a13[0:b], features_a14[0:b],
                                    features_a15[0:b], features_a16[0:b],
                                    features_a17[0:b], features_a18[0:b],
                                    features_a19[0:b], features_a09[0:3000] ))

train_labels_a09=np.zeros(len(train_features_a09))
for i in range(b*13):
    train_labels_a09[i]=1


model_a09=LogisticRegression(random_state=0, max_iter=10000)
model_a09.fit(train_features_a09, train_labels_a09)

# logistic regression a10
train_features_a10=np.concatenate((features_bona_asv[0:b], features_a07[0:b],
                                    features_a08[0:b], features_a09[0:b],
                                    features_a11[0:b], features_a12[0:b],
                                    features_a13[0:b], features_a14[0:b],
                                    features_a15[0:b], features_a16[0:b],
                                    features_a17[0:b], features_a18[0:b],
                                    features_a19[0:b], features_a10[0:3000] ))

train_labels_a10=np.zeros(len(train_features_a10))
for i in range(b*13):
    train_labels_a10[i]=1


model_a10=LogisticRegression(random_state=0, max_iter=10000)
model_a10.fit(train_features_a10, train_labels_a10)

# logistic regression a11
train_features_a11=np.concatenate((features_bona_asv[0:b], features_a07[0:b],
                                    features_a08[0:b], features_a09[0:b],
                                    features_a10[0:b], features_a12[0:b],
                                    features_a13[0:b], features_a14[0:b],
                                    features_a15[0:b], features_a16[0:b],
                                    features_a17[0:b], features_a18[0:b],
                                    features_a19[0:b], features_a11[0:3000] ))

train_labels_a11=np.zeros(len(train_features_a11))
for i in range(b*13):
    train_labels_a11[i]=1

model_a11=LogisticRegression(random_state=0, max_iter=10000)
model_a11.fit(train_features_a11, train_labels_a11)



# logistic regression a12
train_features_a12=np.concatenate((features_bona_asv[0:b], features_a07[0:b],
                                    features_a08[0:b], features_a09[0:b],
                                    features_a10[0:b], features_a11[0:b],
                                    features_a13[0:b], features_a14[0:b],
                                    features_a15[0:b], features_a16[0:b],
                                    features_a17[0:b], features_a18[0:b],
                                    features_a19[0:b], features_a12[0:3000] ))

train_labels_a12=np.zeros(len(train_features_a12))
for i in range(b*13):
    train_labels_a12[i]=1

model_a12=LogisticRegression(random_state=0, max_iter=10000)
model_a12.fit(train_features_a12, train_labels_a12)


# logistic regression a13
train_features_a13=np.concatenate((features_bona_asv[0:b], features_a07[0:b],
                                    features_a08[0:b], features_a09[0:b],
                                    features_a10[0:b], features_a11[0:b],
                                    features_a12[0:b], features_a14[0:b],
                                    features_a15[0:b], features_a16[0:b],
                                    features_a17[0:b], features_a18[0:b],
                                    features_a19[0:b], features_a13[0:3000] ))

train_labels_a13=np.zeros(len(train_features_a13))
for i in range(b*13):
    train_labels_a13[i]=1

model_a13=LogisticRegression(random_state=0, max_iter=10000)
model_a13.fit(train_features_a13, train_labels_a13)

# logistic regression a14
train_features_a14=np.concatenate((features_bona_asv[0:b], features_a07[0:b],
                                    features_a08[0:b], features_a09[0:b],
                                    features_a10[0:b], features_a11[0:b],
                                    features_a12[0:b], features_a13[0:b],
                                    features_a15[0:b], features_a16[0:b],
                                    features_a17[0:b], features_a18[0:b],
                                    features_a19[0:b], features_a14[0:3000] ))

train_labels_a14=np.zeros(len(train_features_a14))
for i in range(b*13):
    train_labels_a14[i]=1

model_a14=LogisticRegression(random_state=0, max_iter=10000)
model_a14.fit(train_features_a14, train_labels_a14)

# logistic regression a15
train_features_a15=np.concatenate((features_bona_asv[0:b], features_a07[0:b],
                                    features_a08[0:b], features_a09[0:b],
                                    features_a10[0:b], features_a11[0:b],
                                    features_a12[0:b], features_a13[0:b],
                                    features_a14[0:b], features_a16[0:b],
                                    features_a17[0:b], features_a18[0:b],
                                    features_a19[0:b], features_a15[0:3000] ))

train_labels_a15=np.zeros(len(train_features_a15))
for i in range(b*13):
    train_labels_a15[i]=1

model_a15=LogisticRegression(random_state=0, max_iter=10000)
model_a15.fit(train_features_a15, train_labels_a15)


# logistic regression a16
train_features_a16=np.concatenate((features_bona_asv[0:b], features_a07[0:b],
                                    features_a08[0:b], features_a09[0:b],
                                    features_a10[0:b], features_a11[0:b],
                                    features_a12[0:b], features_a13[0:b],
                                    features_a14[0:b], features_a15[0:b],
                                    features_a17[0:b], features_a18[0:b],
                                    features_a19[0:b], features_a16[0:3000] ))

train_labels_a16=np.zeros(len(train_features_a16))
for i in range(b*13):
    train_labels_a16[i]=1

model_a16=LogisticRegression(random_state=0, max_iter=10000)
model_a16.fit(train_features_a16, train_labels_a16)

# logistic regression a17
train_features_a17=np.concatenate((features_bona_asv[0:b], features_a07[0:b],
                                    features_a08[0:b], features_a09[0:b],
                                    features_a10[0:b], features_a11[0:b],
                                    features_a12[0:b], features_a13[0:b],
                                    features_a14[0:b], features_a15[0:b],
                                    features_a16[0:b], features_a18[0:b],
                                    features_a19[0:b], features_a17[0:3000] ))

train_labels_a17=np.zeros(len(train_features_a17))
for i in range(b*13):
    train_labels_a17[i]=1

model_a17=LogisticRegression(random_state=0, max_iter=10000)
model_a17.fit(train_features_a17, train_labels_a17)

# logistic regression a18
train_features_a18=np.concatenate((features_bona_asv[0:b], features_a07[0:b],
                                    features_a08[0:b], features_a09[0:b],
                                    features_a10[0:b], features_a11[0:b],
                                    features_a12[0:b], features_a13[0:b],
                                    features_a14[0:b], features_a15[0:b],
                                    features_a16[0:b], features_a17[0:b],
                                    features_a19[0:b], features_a18[0:3000] ))

train_labels_a18=np.zeros(len(train_features_a18))
for i in range(b*13):
    train_labels_a18[i]=1

model_a18=LogisticRegression(random_state=0, max_iter=10000)
model_a18.fit(train_features_a18, train_labels_a18)


# logistic regression a19
train_features_a19=np.concatenate((features_bona_asv[0:b], features_a07[0:b],
                                    features_a08[0:b], features_a09[0:b],
                                    features_a10[0:b], features_a11[0:b],
                                    features_a12[0:b], features_a13[0:b],
                                    features_a14[0:b], features_a15[0:b],
                                    features_a16[0:b], features_a17[0:b],
                                    features_a18[0:b], features_a19[0:3000] ))

train_labels_a19=np.zeros(len(train_features_a19))
for i in range(b*13):
    train_labels_a19[i]=1

model_a19=LogisticRegression(random_state=0, max_iter=10000)
model_a19.fit(train_features_a19, train_labels_a19)





test_features=np.concatenate((features_a07[3000:len(features_a07)], features_a08[3000:len(features_a08)],
                                    features_a09[3000:len(features_a09)], features_a10[3000:len(features_a10)],
                                    features_a11[3000:len(features_a11)], features_a12[3000:len(features_a12)],
                                    features_a13[3000:len(features_a13)], features_a14[3000:len(features_a14)],
                                    features_a15[3000:len(features_a15)], features_a16[3000:len(features_a16)],
                                    features_a17[3000:len(features_a17)], features_a18[3000:len(features_a18)],
                                    features_a19[3000:len(features_a19)],
                              features_bona_asv[5000:len(features_bona_asv)] ))

k=len(features_a07)-3000
test_labels=np.zeros(len(test_features))
for i in range(0, k):
    test_labels[i]=7
    test_labels[i+k]=8
    test_labels[i+(2*k)]=9
    test_labels[i + (3 * k)] = 10
    test_labels[i + (4 * k)] = 11
    test_labels[i + (5 * k)] = 12
    test_labels[i + (6 * k)] = 13
    test_labels[i + (7 * k)] = 14
    test_labels[i + (8 * k)] = 15
    test_labels[i + (9 * k)] = 16
    test_labels[i + (10 * k)] = 17
    test_labels[i + (11 * k)] = 18
    test_labels[i + (12 * k)] = 19


predictions_bona=model_bona.predict(test_features)
predictions_a07=model_a07.predict(test_features)
predictions_a08=model_a08.predict(test_features)
predictions_a09=model_a09.predict(test_features)
predictions_a10=model_a10.predict(test_features)
predictions_a11=model_a11.predict(test_features)
predictions_a12=model_a12.predict(test_features)
predictions_a13=model_a13.predict(test_features)
predictions_a14=model_a14.predict(test_features)
predictions_a15=model_a15.predict(test_features)
predictions_a16=model_a16.predict(test_features)
predictions_a17=model_a17.predict(test_features)
predictions_a18=model_a18.predict(test_features)
predictions_a19=model_a19.predict(test_features)

predictions_proba_bona=model_bona.predict_proba(test_features)
predictions_proba_a07=model_a07.predict_proba(test_features)
predictions_proba_a08=model_a08.predict_proba(test_features)
predictions_proba_a09=model_a09.predict_proba(test_features)
predictions_proba_a10=model_a10.predict_proba(test_features)
predictions_proba_a11=model_a11.predict_proba(test_features)
predictions_proba_a12=model_a12.predict_proba(test_features)
predictions_proba_a13=model_a13.predict_proba(test_features)
predictions_proba_a14=model_a14.predict_proba(test_features)
predictions_proba_a15=model_a15.predict_proba(test_features)
predictions_proba_a16=model_a16.predict_proba(test_features)
predictions_proba_a17=model_a17.predict_proba(test_features)
predictions_proba_a18=model_a18.predict_proba(test_features)
predictions_proba_a19=model_a19.predict_proba(test_features)


predictions_tot=np.zeros((len(test_features), 14))

predictions_tot[:, 0]=predictions_bona
predictions_tot[:, 1]=predictions_a07
predictions_tot[:, 2]=predictions_a08
predictions_tot[:, 3]=predictions_a09
predictions_tot[:, 4]=predictions_a10
predictions_tot[:, 5]=predictions_a11
predictions_tot[:, 6]=predictions_a12
predictions_tot[:, 7]=predictions_a13
predictions_tot[:, 8]=predictions_a14
predictions_tot[:, 9]=predictions_a15
predictions_tot[:, 10]=predictions_a16
predictions_tot[:, 11]=predictions_a17
predictions_tot[:, 12]=predictions_a18
predictions_tot[:, 13]=predictions_a19

matrix=np.zeros((14, 14))
classes=np.zeros(len(test_features))
more_than_one_class=np.zeros(len(test_features))
for i in range(len(test_features)):
    classes[i] = -1


for i in range(len(test_features)):
    for j in range(14):
        if ( predictions_tot[i, j] == 0 ):
            if (classes[i]!=-1):
                more_than_one_class[i]=1

            if (j==0):
                classes[i]=j
            else :
                classes[i]=j+6

unknown=0

for i in range(len(classes)):
    if (classes[i]==-1):
        unknown=unknown+1










zero=0
for i in range(len(predictions)):
    if (predictions[i]==0.0):
        zero=zero+1






#####################################################################

# logistic ovr



train_features=np.concatenate((features_a07[0:3000], features_a08[0:3000],
                                    features_a09[0:3000], features_a10[0:3000],
                                    features_a11[0:3000], features_a12[0:3000],
                                    features_a13[0:3000], features_a14[0:3000],
                                    features_a15[0:3000], features_a16[0:3000],
                                    features_a17[0:3000], features_a18[0:3000],
                                    features_a19[0:3000], features_bona_asv[0:5000] ))

train_labels_07=np.zeros(3000)
for i in range(len(train_labels_07)):
    train_labels_07[i]=7

train_labels_08=np.zeros(3000)
for i in range(len(train_labels_08)):
    train_labels_08[i]=8

train_labels_09=np.zeros(3000)
for i in range(len(train_labels_09)):
    train_labels_09[i]=9

train_labels_10=np.zeros(3000)
for i in range(len(train_labels_10)):
    train_labels_10[i]=10

train_labels_11=np.zeros(3000)
for i in range(len(train_labels_11)):
    train_labels_11[i]=11

train_labels_12=np.zeros(3000)
for i in range(len(train_labels_12)):
    train_labels_12[i]=12

train_labels_13=np.zeros(3000)
for i in range(len(train_labels_13)):
    train_labels_13[i]=13

train_labels_14=np.zeros(3000)
for i in range(len(train_labels_14)):
    train_labels_14[i]=14

train_labels_15=np.zeros(3000)
for i in range(len(train_labels_15)):
    train_labels_15[i]=15


train_labels_16=np.zeros(3000)
for i in range(len(train_labels_16)):
    train_labels_16[i]=16

train_labels_17=np.zeros(3000)
for i in range(len(train_labels_17)):
    train_labels_17[i]=17

train_labels_18=np.zeros(3000)
for i in range(len(train_labels_18)):
    train_labels_18[i]=18

train_labels_19=np.zeros(3000)
for i in range(len(train_labels_19)):
    train_labels_19[i]=19

train_labels_bona=np.zeros(5000)
for i in range(len(train_labels_bona)):
    train_labels_bona[i]=0

train_labels=np.concatenate((train_labels_07, train_labels_08,
                             train_labels_09, train_labels_10,
                             train_labels_11, train_labels_12,
                             train_labels_13, train_labels_14,
                             train_labels_15, train_labels_16,
                             train_labels_17, train_labels_18,
                             train_labels_19, train_labels_bona))

test_features=np.concatenate((features_a07[3000:len(features_a07)], features_a08[3000:len(features_a08)],
                                    features_a09[3000:len(features_a09)], features_a10[3000:len(features_a10)],
                                    features_a11[3000:len(features_a11)], features_a12[3000:len(features_a12)],
                                    features_a13[3000:len(features_a13)], features_a14[3000:len(features_a14)],
                                    features_a15[3000:len(features_a15)], features_a16[3000:len(features_a16)],
                                    features_a17[3000:len(features_a17)], features_a18[3000:len(features_a18)],
                                    features_a19[3000:len(features_a19)],
                              features_bona_asv[5000:len(features_bona_asv)] ))
k=len(features_a07)-3000
j=len(features_bona_asv)-5000

test_labels_07 = np.zeros(k)
for i in range(len(test_labels_07)):
    test_labels_07[i] = 7

test_labels_08 = np.zeros(k)
for i in range(len(test_labels_08)):
    test_labels_08[i] = 8

test_labels_09 = np.zeros(k)
for i in range(len(test_labels_09)):
    test_labels_09[i] = 9

test_labels_10 = np.zeros(k)
for i in range(len(test_labels_10)):
    test_labels_10[i] = 10

test_labels_11 = np.zeros(k)
for i in range(len(test_labels_11)):
    test_labels_11[i] = 11

test_labels_12 = np.zeros(k)
for i in range(len(test_labels_12)):
    test_labels_12[i] = 12

test_labels_13 = np.zeros(k)
for i in range(len(test_labels_13)):
    test_labels_13[i] = 13

test_labels_14 = np.zeros(k)
for i in range(len(test_labels_14)):
    test_labels_14[i] = 14

test_labels_15 = np.zeros(k)
for i in range(len(test_labels_15)):
    test_labels_15[i] = 15

test_labels_16 = np.zeros(k)
for i in range(len(test_labels_16)):
    test_labels_16[i] = 16

test_labels_17 = np.zeros(k)
for i in range(len(test_labels_17)):
    test_labels_17[i] = 17

test_labels_18 = np.zeros(k)
for i in range(len(test_labels_18)):
    test_labels_18[i] = 18

test_labels_19 = np.zeros(k)
for i in range(len(test_labels_19)):
    test_labels_19[i] = 19

test_labels_bona = np.zeros(j)
for i in range(len(test_labels_bona)):
    test_labels_bona[i] = 0



test_labels=np.concatenate((test_labels_07, test_labels_08,
                            test_labels_09, test_labels_10,
                            test_labels_11, test_labels_12,
                            test_labels_13, test_labels_14,
                            test_labels_15, test_labels_16,
                            test_labels_17, test_labels_18,
                            test_labels_19, test_labels_bona))

model = LogisticRegression(random_state=0, multi_class='ovr', max_iter=10000)
model.fit(train_features, train_labels)
predictions=model.predict(test_features)
#predictions_proba=model.predict_proba(test_features)

cm=confusion_matrix(test_labels, predictions)
score=model.score(train_features, train_labels)

sum=0
for i in range(14):
    sum=sum+ cm[i,i]

accuracy=sum/len(test_features)



class_names=['bonafide', 'a07', 'a08', 'a09', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19']



disp = plot_confusion_matrix(model, test_features, test_labels,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
disp.ax_.set_title('Confusion matrix')



plt.savefig('conf_mat_log_ovr.png')



def plot_confusion_matrix_2(cm,
                      target_names,
                      title='Confusion matrix',
                      cmap=None,
                      normalize=True):

    FONT_SIZE = 8

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8*2, 6*2))    # 8, 6
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90, fontsize=FONT_SIZE)
        plt.yticks(tick_marks, target_names, fontsize=FONT_SIZE)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                    horizontalalignment="center",
                    fontsize=FONT_SIZE,
                    color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    fontsize=FONT_SIZE,
                    color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('confusion_matrix.png')