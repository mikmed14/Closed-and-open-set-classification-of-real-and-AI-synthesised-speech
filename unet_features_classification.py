import scipy as sp
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM
from sklearn import tree
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

features_bona=np.load('features_bona.npy')
features_a07=np.load('features_a07.npy')
features_a08=np.load('features_a08.npy')
features_a09=np.load('features_a09.npy')
features_a10=np.load('features_a10.npy')
features_a11=np.load('features_a11.npy')
features_a12=np.load('features_a12.npy')
features_a13=np.load('features_a13.npy')
features_a14=np.load('features_a14.npy')
features_a15=np.load('features_a15.npy')
features_a16=np.load('features_a16.npy')
features_a17=np.load('features_a17.npy')
features_a18=np.load('features_a18.npy')
features_a19=np.load('features_a19.npy')


scaler = MinMaxScaler()
scaler.fit(features_bona)
features_bona_asv=scaler.transform(features_bona)

scaler.fit(features_a07)
features_a07=scaler.transform(features_a07)

scaler.fit(features_a08)
features_a08=scaler.transform(features_a08)

scaler.fit(features_a09)
features_a09=scaler.transform(features_a09)

scaler.fit(features_a10)
features_a10=scaler.transform(features_a10)

scaler.fit(features_a11)
features_a11=scaler.transform(features_a11)

scaler.fit(features_a12)
features_a12=scaler.transform(features_a12)

scaler.fit(features_a13)
features_a13=scaler.transform(features_a13)

scaler.fit(features_a14)
features_a14=scaler.transform(features_a14)

scaler.fit(features_a15)
features_a15=scaler.transform(features_a15)

scaler.fit(features_a16)
features_a16=scaler.transform(features_a16)

scaler.fit(features_a17)
features_a17=scaler.transform(features_a17)

scaler.fit(features_a18)
features_a18=scaler.transform(features_a18)

scaler.fit(features_a19)
features_a19=scaler.transform(features_a19)


# MULTICLASS CLASSIFIER

train_features=np.concatenate((features_a07[0:3000],
                               features_a08[0:3000],
                               features_a09[0:3000],
                                     features_a10[0:3000],
                               features_a11[0:3000],
                               features_a12[0:3000],
                               features_a13[0:3000],
                               features_a14[0:3000],
                                     features_a15[0:3000],
                                     features_a16[0:3000],
                                     features_a17[0:3000],
                               features_a18[0:3000],
                               features_a19[0:3000],
                                     features_bona_asv[0:3000] ))

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

train_labels_bona=np.zeros(3000)
for i in range(len(train_labels_bona)):
    train_labels_bona[i]=0

train_labels=np.concatenate((train_labels_07, train_labels_08,
                             train_labels_09,
                              train_labels_10, train_labels_11,
                             train_labels_12, train_labels_13,
                             train_labels_14,

                             train_labels_15, train_labels_16,
                             train_labels_17, train_labels_18,
                             train_labels_19,
                              train_labels_bona))

test_features=np.concatenate((features_a07[3000:len(features_a07)],
                              features_a08[3000:len(features_a08)],
                              features_a09[3000:len(features_a09)],
                                     features_a10[3000:len(features_a10)],
                              features_a11[3000:len(features_a11)],
                              features_a12[3000:len(features_a12)],
                              features_a13[3000:len(features_a13)],
                              features_a14[3000:len(features_a14)],

                                    features_a15[3000:len(features_a15)],
                              features_a16[3000:len(features_a16)],
                                    features_a17[3000:len(features_a17)],
                              features_a18[3000:len(features_a18)],
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
                            test_labels_09,
                             test_labels_10,test_labels_11,
                            test_labels_12, test_labels_13,
                            test_labels_14,

                            test_labels_15, test_labels_16,
                            test_labels_17, test_labels_18,
                            test_labels_19,
                             test_labels_bona))

class_names=['bonafide', 'a07', 'a08', 'a09',  'a10', 'a11', 'a12', 'a13', 'a14',
              'a15', 'a16', 'a17', 'a18', 'a19']


# 1) svm
model=svm.SVC()
model.fit(train_features, train_labels)
predictions=model.predict(test_features)
cm=confusion_matrix(test_labels, predictions, normalize='true')

sum=0.0
for i in range(14):
    sum=sum+ cm[i,i]

accuracy=sum/14

print('accuracy=', accuracy)


cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(15,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, multilabel classification, SVM')
plt.savefig('cm_ufeat_multi_svm.png')

# 2) logistic ovr
model = LogisticRegression(random_state=0, multi_class='ovr', max_iter=10000)
model.fit(train_features, train_labels)
predictions=model.predict(test_features)
cm=confusion_matrix(test_labels, predictions, normalize='true')

sum=0.0
for i in range(14):
    sum=sum+ cm[i,i]

accuracy=sum/14

print('accuracy=', accuracy)


cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(15,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, multilabel classification, Logistic regression')
plt.savefig('cm_ufeat_multi_logovr.png')

#3) random forest

model=RandomForestClassifier(n_estimators=5000, min_samples_split=2,
                              min_samples_leaf=1, min_weight_fraction_leaf=0.0,

                              verbose=0, n_jobs=-1, max_samples=None,
                              max_depth=None, max_leaf_nodes=None)
model.fit(train_features, train_labels)
predictions=model.predict(test_features)
cm=confusion_matrix(test_labels, predictions, normalize='true')

sum=0.0
for i in range(14):
    sum=sum+ cm[i,i]

accuracy=sum/14

print('accuracy=', accuracy)


cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(15,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, multilabel classification, Random forest')
plt.savefig('cm_ufeat_multi_forest.png')


# BINARY CLASSIFICATION BONAFIDE VS ALL FAKE

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
                               features_bona[0:4940]))
train_labels=np.zeros(len(train_features))
for i in range(4940):
    train_labels[i]=1

test_features=np.concatenate((features_a07[1000:1150],
                              features_a08[1000:1150],
                              features_a09[1000:1150],
                              features_a10[1000:1150],
                              features_a11[1000:1150],
                              features_a12[1000:1150],
                              features_a13[1000:1150],
                              features_a14[1000:1150],
                              features_a15[1000:1150],
                              features_a16[1000:1150],
                              features_a17[1000:1150],
                              features_a18[1000:1150],
                              features_a19[1000:1150],
                              features_bona[5000:6950]))
test_labels=np.zeros(len(test_features))
for i in range(1950):
    test_labels[i]=1


class_names=['bonafide', 'fake']

# 1) svm
model=svm.SVC()
model.fit(train_features, train_labels)
predictions=model.predict(test_features)
cm=confusion_matrix(test_labels, predictions, normalize='true')

sum=0.0
sum=cm[0,0]+cm[1,1]

accuracy=sum/2

print('accuracy=', accuracy)


cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, binary classification bonafide vs fake, SVM')
plt.savefig('cm_ufeat_bvs_svm.png')

# 2) logistic ovr
model = LogisticRegression(random_state=0, multi_class='ovr', max_iter=10000)
model.fit(train_features, train_labels)
predictions=model.predict(test_features)
cm=confusion_matrix(test_labels, predictions, normalize='true')

sum=0.0
sum=cm[0,0]+cm[1,1]

accuracy=sum/2

print('accuracy=', accuracy)


cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, binary classification bonafide vs fake,'
          ' Logistic regression')
plt.savefig('cm_ufeat_bvs_logovr.png')

#3) random forest

model=RandomForestClassifier(n_estimators=5000, min_samples_split=2,
                              min_samples_leaf=1, min_weight_fraction_leaf=0.0,

                              verbose=0, n_jobs=-1, max_samples=None,
                              max_depth=None, max_leaf_nodes=None)
model.fit(train_features, train_labels)
predictions=model.predict(test_features)
cm=confusion_matrix(test_labels, predictions, normalize='true')

sum=0.0
sum=cm[0,0]+cm[1,1]

accuracy=sum/2

print('accuracy=', accuracy)


cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, binary classification bonafide vs fake,'
          ' Random forest')
plt.savefig('cm_ufeat_bvs_forest.png')


# binary classification bona vs each one of the fakes

train_features=np.concatenate((features_a19[0:3000],
                               features_bona[0:3000]))

train_labels=np.zeros(len(train_features))
for i in range(3000):
    train_labels[i]=1

test_features=np.concatenate((features_a19[3000:4000],
                              features_bona[3000:4000]))

test_labels=np.zeros(len(test_features))
for i in range(1000):
    test_labels[i]=1

class_names=['bonafide', 'a19']

model=svm.SVC()
model.fit(train_features, train_labels)
predictions=model.predict(test_features)
cm=confusion_matrix(test_labels, predictions, normalize='true')

sum=0.0
sum=cm[0,0]+cm[1,1]

accuracy=sum/2

print('accuracy=', accuracy)


cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, binary classification bonafide vs a19,'
          ' SVM')
plt.savefig('cm_ufeat_a19_svm.png')


# ONE CLASS


test_features=np.concatenate((   features_a07[2000:2070],
                                 features_a08[2000:2070],
                                 features_a09[2000:2070],
                              features_a10[2000:2070],
                                 features_a11[2000:2070],
                                 features_a12[2000:2070],
                                 features_a13[2000:2070],
                                 features_a14[2000:2070],

                              features_a15[2000:2070],
                              features_a16[2000:2070],
                                 features_a17[2000:2070],
                                 features_a18[2000:2070],
                                 features_a19[2000:2070],
                                 features_bona_asv[3000:3910],
                              ))
test_labels=np.ones(len(test_features))

for i in range(910):
    test_labels[i]=-1

class_names=['fake', 'bonafide']

one_class_svm=OneClassSVM(gamma='auto').fit(features_bona[0:3000])
predictions=one_class_svm.predict(test_features)
scores=one_class_svm.score_samples(test_features)

cm=confusion_matrix(test_labels, predictions, normalize='true')

sum=0.0
sum=cm[0,0]+cm[1,1]

accuracy=sum/2

print('accuracy=', accuracy)


cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, One class SVM trained on bonafide features'
          )
plt.savefig('cm_ufeat_oneclass_bona.png')