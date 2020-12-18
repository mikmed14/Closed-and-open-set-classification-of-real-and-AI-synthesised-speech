import scipy as sp
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
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




# normalization

scaler = MinMaxScaler()
scaler.fit(features_bona_asv)
features_bona_asv=scaler.transform(features_bona_asv)
scaler.fit(features_a07)
features_a07=scaler.transform(features_a07)
scaler.fit(features_a08)
features_a08=scaler.transform(features_a08)
scaler.fit(features_a09)
features_a9=scaler.transform(features_a09)
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


# MULTICLASS CLASSIFIERS

train_features=np.concatenate((features_a07[0:3000], features_a08[0:3000],
                                    features_a09[0:3000], features_a10[0:3000],
                                    features_a11[0:3000], features_a12[0:3000],
                                    features_a13[0:3000], features_a14[0:3000],
                                    features_a15[0:3000], features_a16[0:3000],
                                    features_a17[0:3000], features_a18[0:3000],
                                    features_a19[0:3000], features_bona_asv[0:3000] ))

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

class_names=['bonafide', 'a07', 'a08', 'a09', 'a10', 'a11', 'a12',
             'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19']

# 1) svm
model=svm.SVC()
model.fit(train_features, train_labels)
predictions=model.predict(test_features)
cm=confusion_matrix(test_labels, predictions, normalize='true')

sum=0
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
plt.savefig('cm_farid_mutlti_svm.png')

# 2) logistic ovr


model = LogisticRegression(random_state=0, multi_class='ovr', max_iter=10000)
model.fit(train_features, train_labels)
predictions=model.predict(test_features)
#predictions_proba=model.predict_proba(test_features)

cm=confusion_matrix(test_labels, predictions, normalize='true')
#score=model.score(train_features, train_labels)

sum=0
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
plt.savefig('cm_farid_mutlti_logovr.png')


#3) random forest

model=RandomForestClassifier(n_estimators=5000, min_samples_split=2,
                              min_samples_leaf=1, min_weight_fraction_leaf=0.0,

                              verbose=0, n_jobs=-1, max_samples=None,
                              max_depth=None, max_leaf_nodes=None)
model.fit(train_features, train_labels)
predictions=model.predict(test_features)

cm=confusion_matrix(test_labels, predictions, normalize='true')
#score=model.score(train_features, train_labels)

sum=0
for i in range(14):
    sum=sum+ cm[i,i]

accuracy=sum/14

print('accuracy=', accuracy)




cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(15,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, multilabel classification, Random Forest')
plt.savefig('cm_farid_mutlti_forest.png')


# BINARY CLASSIFIERS : bona vs all fake

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

class_names=['bonafide', 'fake']

# 1) svm
model=svm.SVC()
model.fit(train_features, train_labels)
predictions=model.predict(test_features)
cm=confusion_matrix(test_labels, predictions, normalize='true')

sum=cm[0,0]+cm[1,1]


accuracy=sum/2

print('accuracy=' , accuracy)



cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, binary classification bonafide vs fake, SVM')
plt.savefig('cm_farid_bvs_svm.png')

# 2) logistic ovr


model = LogisticRegression(random_state=0, multi_class='ovr', max_iter=10000)
model.fit(train_features, train_labels)
predictions=model.predict(test_features)
#predictions_proba=model.predict_proba(test_features)

cm=confusion_matrix(test_labels, predictions, normalize='true')
#score=model.score(train_features, train_labels)

sum=cm[0,0]+cm[1,1]


accuracy=sum/2

print('accuracy=' , accuracy)



cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, binary classification bonafide vs fake, Logistic Regression')
plt.savefig('cm_farid_bvs_logovr.png')






# 3) random forest

model=RandomForestClassifier(n_estimators=5000, min_samples_split=2,
                              min_samples_leaf=1, min_weight_fraction_leaf=0.0,

                              verbose=1, n_jobs=-1, max_samples=None,
                              max_depth=None, max_leaf_nodes=None)
model.fit(train_features, train_labels)
predictions=model.predict(test_features)

cm=confusion_matrix(test_labels, predictions, normalize='true')
#score=model.score(train_features, train_labels)

sum=cm[0,0]+cm[1,1]


accuracy=sum/2

print('accuracy=' , accuracy)



cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, binary classification bonafide vs fake, Random Forest')
plt.savefig('cm_farid_bvs_forest.png')






# BINARY CLASSIFIERS : bona vs each fake

# a07

train_features=np.concatenate((features_a19[0:3000],
                               features_bona_asv[0:3000]))
train_labels=np.zeros(len(train_features))
for i in range(3000):
    train_labels[i]=1

test_features=np.concatenate((features_a19[3000:4900],
                              features_bona_asv[3000:4900]))
test_labels=np.zeros(len(test_features))
for i in range(1900):
    test_labels[i]=1

class_names=['bonafide', 'a19']



model=svm.SVC()
model.fit(train_features, train_labels)
predictions=model.predict(test_features)

cm=confusion_matrix(test_labels, predictions, normalize='true')
#score=model.score(train_features, train_labels)

sum=cm[0,0]+cm[1,1]


accuracy=sum/2

print('accuracy=' , accuracy)



cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, binary classification bonafide vs a19, SVM')
plt.savefig('cm_farid_a19_svm.png')


# OPEN SET

# a09 a11 a13 a14 a17

train_features_a09=np.concatenate((features_a09[0:2000],
                                   features_a11[0:500],
                                   features_a13[0:500],
                                   features_a14[0:500],
                                   features_a17[0:500],
                                   ))
train_labels_a09=np.zeros(len(train_features_a09))
for i in range(2000):
    train_labels_a09[i]=1

train_features_a11=np.concatenate((features_a11[0:2000],
                                   features_a09[0:500],
                                   features_a13[0:500],
                                   features_a14[0:500],
                                   features_a17[0:500],
                                   ))
train_labels_a11=np.zeros(len(train_features_a11))
for i in range(2000):
    train_labels_a11[i]=1

train_features_a13=np.concatenate((features_a13[0:2000],
                                   features_a09[0:500],
                                   features_a11[0:500],
                                   features_a14[0:500],
                                   features_a17[0:500],
                                   ))
train_labels_a13=np.zeros(len(train_features_a13))
for i in range(2000):
    train_labels_a13[i]=1

train_features_a14=np.concatenate((features_a14[0:2000],
                                   features_a09[0:500],
                                   features_a11[0:500],
                                   features_a13[0:500],
                                   features_a17[0:500],
                                   ))
train_labels_a14=np.zeros(len(train_features_a14))
for i in range(2000):
    train_labels_a14[i]=1

train_features_a17=np.concatenate((features_a17[0:2000],
                                   features_a09[0:500],
                                   features_a11[0:500],
                                   features_a13[0:500],
                                   features_a14[0:500],
                                   ))
train_labels_a17=np.zeros(len(train_features_a17))
for i in range(2000):
    train_labels_a17[i]=1


model09=svm.SVC(probability=True)
model09.fit(train_features_a09, train_labels_a09)

model11=svm.SVC(probability=True)
model11.fit(train_features_a11, train_labels_a11)

model13=svm.SVC(probability=True)
model13.fit(train_features_a13, train_labels_a13)

model14=svm.SVC(probability=True)
model14.fit(train_features_a14, train_labels_a14)

model17=svm.SVC(probability=True)
model17.fit(train_features_a17, train_labels_a17)


test_features=np.concatenate((features_a09[2000:2500],
                              features_a11[2000:2500],
                              features_a13[2000:2500],
                              features_a14[2000:2500],
                              features_a17[2000:2500],
                              features_a15[2000:2125],
                              features_a18[2000:2125],
                              features_a19[2000:2125],
                              features_bona_asv[2000:2125]))


class_names=['unknown', 'a09', 'a11', 'a13', 'a14', 'a17']

predictions_a09=model09.predict(test_features)
predictions_a11=model11.predict(test_features)
predictions_a13=model13.predict(test_features)
predictions_a14=model14.predict(test_features)
predictions_a17=model17.predict(test_features)

predictions_proba_a09=model09.predict_proba(test_features)
predictions_proba_a11=model11.predict_proba(test_features)
predictions_proba_a13=model13.predict_proba(test_features)
predictions_proba_a14=model14.predict_proba(test_features)
predictions_proba_a17=model17.predict_proba(test_features)

predictions_tot=np.zeros(len(test_features))

sum=np.zeros(len(test_features))

for i in range(len(test_features)):

    if (predictions_a09[i]==1 and predictions_a11[i]==0 and
            predictions_a13[i]==0 and predictions_a14[i]==0 and
            predictions_a17[i]==0):

        predictions_tot[i]=9

    if (predictions_a09[i] == 0 and predictions_a11[i] == 1 and
            predictions_a13[i] == 0 and predictions_a14[i] == 0 and
            predictions_a17[i] == 0):
        predictions_tot[i] = 11

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_a13[i] == 1 and predictions_a14[i] == 0 and
            predictions_a17[i] == 0):
        predictions_tot[i] = 13

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_a13[i] == 0 and predictions_a14[i] == 1 and
            predictions_a17[i] == 0):
        predictions_tot[i] = 14

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_a13[i] == 0 and predictions_a14[i] == 0 and
            predictions_a17[i] == 1):
        predictions_tot[i] = 17

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_a13[i] == 0 and predictions_a14[i] == 0 and
            predictions_a17[i] == 0):
        predictions_tot[i] = -1 # unknown class

    sum[i]=sum[i]+predictions_a09[i]+predictions_a11[i]+ \
        predictions_a13[i]+ predictions_a14[i]+ predictions_a17[i]
    if sum[i] > 1 :
        predictions_tot[i]=-2 #more than one class



max_proba=0.0

for i in range(len(predictions_tot)):
    if (predictions_tot[i]==-2):

        max_proba=max(predictions_proba_a09[i][1],
                         predictions_proba_a11[i][1],
                         predictions_proba_a13[i][1],
                         predictions_proba_a14[i][1],
                         predictions_proba_a17[i][1])

        if (max_proba==predictions_proba_a09[i][1]):
            predictions_tot[i]=9
        if (max_proba==predictions_proba_a11[i][1]):
            predictions_tot[i]=11
        if (max_proba==predictions_proba_a13[i][1]):
            predictions_tot[i]=13
        if (max_proba==predictions_proba_a14[i][1]):
            predictions_tot[i]=14
        if (max_proba==predictions_proba_a17[i][1]):
            predictions_tot[i]=17



more=0

for i in range(len(predictions_tot)):
    if predictions_tot[i]==-2:
        more=more+1



test_labels=np.zeros(len(predictions_tot))
for i in range(500):
    test_labels[i]=9
    test_labels[i+500]=11
    test_labels[i+1000]=13
    test_labels[i+1500]=14
    test_labels[i+2000]=17
    test_labels[i+2500]=-1
    #test_labels[i+3000]=-1
    #test_labels[i+3500]=-1
    #test_labels[i+4000]=-1


cm=confusion_matrix(test_labels, predictions_tot, normalize='true')

sum=0
for i in range(6):
    sum=sum+ cm[i,i]

accuracy=sum/6

print('accuracy=', accuracy)



cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, Open set classification with bonafide as unknown')
plt.savefig('cm_farid_open_strain.png')

#roc curve
maxproba=np.zeros(len(test_features))
almost_max_proba=np.zeros(len(test_features))
proba=np.zeros(5)
proba_sorted=np.zeros(5)

for i in range(len(test_features)):
    proba[0]=predictions_proba_a09[i][1]
    proba[1] = predictions_proba_a11[i][1]
    proba[2] = predictions_proba_a13[i][1]
    proba[3] = predictions_proba_a14[i][1]
    proba[4] = predictions_proba_a17[i][1]
    proba_sorted=sorted(proba)
    maxproba[i]=proba_sorted[4]
    almost_max_proba[i]=proba_sorted[3]

ratio=almost_max_proba/maxproba

test_labels_roc=np.zeros(len(test_labels))
for i in range(len(test_labels)):
    if (test_labels[i]>0):
        test_labels_roc[i]=0
    if (test_labels[i]==-1):
        test_labels_roc[i]=1


false_pos_rate07, true_pos_rate07, thresholds07 = \
    roc_curve(test_labels_roc, maxproba, 0)
roc_auc07 = auc(false_pos_rate07, true_pos_rate07,)
plt.figure()
plt.plot(false_pos_rate07, true_pos_rate07, linewidth=5,
         label='AUC = %0.3f'% roc_auc07)
plt.plot([0,1],[0,1], linewidth=5)
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('ROC curve, max score, bonafide as unknown')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('roc_farid_max_strain.png')

false_pos_rate07, true_pos_rate07, thresholds07 = \
    roc_curve(test_labels_roc, ratio)
roc_auc07 = auc(false_pos_rate07, true_pos_rate07,)
plt.figure()
plt.plot(false_pos_rate07, true_pos_rate07, linewidth=5,
         label='AUC = %0.3f'% roc_auc07)
plt.plot([0,1],[0,1], linewidth=5)
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('ROC curve, score ratio, bonafide as unknown')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('roc_farid_ratio_strain.png')









# OPEN SET

# bonafide, a09 a11 a14 a17

train_features_bona=np.concatenate((features_bona_asv[0:2000],
                                   features_a09[0:500],
                                   features_a11[0:500],
                                   features_a14[0:500],
                                   features_a17[0:500],
                                   ))
train_labels_bona=np.zeros(len(train_features_bona))
for i in range(2000):
    train_labels_bona[i]=1


train_features_a09=np.concatenate((features_a09[0:2000],
                                   features_a11[0:500],
                                   features_bona_asv[0:500],
                                   features_a14[0:500],
                                   features_a17[0:500],
                                   ))
train_labels_a09=np.zeros(len(train_features_a09))
for i in range(2000):
    train_labels_a09[i]=1

train_features_a11=np.concatenate((features_a11[0:2000],
                                   features_a09[0:500],
                                   features_bona_asv[0:500],
                                   features_a14[0:500],
                                   features_a17[0:500],
                                   ))
train_labels_a11=np.zeros(len(train_features_a11))
for i in range(2000):
    train_labels_a11[i]=1



train_features_a14=np.concatenate((features_a14[0:2000],
                                   features_a09[0:500],
                                   features_a11[0:500],
                                   features_bona_asv[0:500],
                                   features_a17[0:500],
                                   ))
train_labels_a14=np.zeros(len(train_features_a14))
for i in range(2000):
    train_labels_a14[i]=1

train_features_a17=np.concatenate((features_a17[0:2000],
                                   features_a09[0:500],
                                   features_a11[0:500],
                                   features_bona_asv[0:500],
                                   features_a14[0:500],
                                   ))
train_labels_a17=np.zeros(len(train_features_a17))
for i in range(2000):
    train_labels_a17[i]=1


modelbona=svm.SVC(probability=True)
modelbona.fit(train_features_bona, train_labels_bona)


model09=svm.SVC(probability=True)
model09.fit(train_features_a09, train_labels_a09)

model11=svm.SVC(probability=True)
model11.fit(train_features_a11, train_labels_a11)


model14=svm.SVC(probability=True)
model14.fit(train_features_a14, train_labels_a14)

model17=svm.SVC(probability=True)
model17.fit(train_features_a17, train_labels_a17)


test_features=np.concatenate(( features_bona_asv[2000:4000],  features_a09[2000:4000],#2500
                              features_a11[2000:4000],

                              features_a14[2000:4000],
                              features_a17[2000:4000],
                              features_a15[2000:2500],
                              features_a18[2000:2500],
                              features_a19[2000:2500],
                              features_a13[2000:2500]))

test_labels=np.zeros(len(test_features))
for i in range(2000):
    test_labels[i]=0
    test_labels[i+2000]=9
    test_labels[i+4000]=11
    test_labels[i+6000]=14
    test_labels[i+8000]=17
    test_labels[i+10000]=-1
    #test_labels[i+3000]=-1
    #test_labels[i+3500]=-1
    #test_labels[i+4000]=-1

class_names=['unknown', 'bonafide', 'a09', 'a11', 'a14', 'a17']

'''
one_class_svm=OneClassSVM(gamma='auto').fit(features_bona_asv[0:2000])
predictions_one_class=one_class_svm.predict(test_features)
scores_one_class=one_class_svm.score_samples(test_features)
'''

predictions_bona=modelbona.predict(test_features)
predictions_a09=model09.predict(test_features)
predictions_a11=model11.predict(test_features)
predictions_a14=model14.predict(test_features)
predictions_a17=model17.predict(test_features)


predictions_proba_bona=modelbona.predict_proba(test_features)
predictions_proba_a09=model09.predict_proba(test_features)
predictions_proba_a11=model11.predict_proba(test_features)
predictions_proba_a14=model14.predict_proba(test_features)
predictions_proba_a17=model17.predict_proba(test_features)

predictions_tot=np.zeros(len(test_features))

sum=np.zeros(len(test_features))

for i in range(len(test_features)):

    if (predictions_a09[i]==1 and predictions_a11[i]==0 and
            predictions_bona[i]==0 and predictions_a14[i]==0 and
            predictions_a17[i]==0):

        predictions_tot[i]=9

    if (predictions_a09[i] == 0 and predictions_a11[i] == 1 and
            predictions_bona[i] == 0 and predictions_a14[i] == 0 and
            predictions_a17[i] == 0):
        predictions_tot[i] = 11

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_bona[i] == 1 and predictions_a14[i] == 0 and
            predictions_a17[i] == 0):
        predictions_tot[i] = 0

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_bona[i] == 0 and predictions_a14[i] == 1 and
            predictions_a17[i] == 0):
        predictions_tot[i] = 14

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_bona[i] == 0 and predictions_a14[i] == 0 and
            predictions_a17[i] == 1):
        predictions_tot[i] = 17

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_bona[i] == 0 and predictions_a14[i] == 0 and
            predictions_a17[i] == 0):
        predictions_tot[i] = -1 # unknown class

    sum[i]=sum[i]+predictions_a09[i]+predictions_a11[i]+ \
        predictions_bona[i]+ predictions_a14[i]+ predictions_a17[i]
    if sum[i] > 1 :
        predictions_tot[i]=-2 #more than one class



max_proba=0.0

for i in range(len(predictions_tot)):
    if (predictions_tot[i]==-2):

        max_proba=max(predictions_proba_a09[i][1],
                         predictions_proba_a11[i][1],
                         predictions_proba_bona[i][1],
                         predictions_proba_a14[i][1],
                         predictions_proba_a17[i][1])

        if (max_proba==predictions_proba_a09[i][1]):
            predictions_tot[i]=9
        if (max_proba==predictions_proba_a11[i][1]):
            predictions_tot[i]=11
        if (max_proba==predictions_proba_bona[i][1]):
            predictions_tot[i]=0
        if (max_proba==predictions_proba_a14[i][1]):
            predictions_tot[i]=14
        if (max_proba==predictions_proba_a17[i][1]):
            predictions_tot[i]=17



more=0

for i in range(len(predictions_tot)):
    if predictions_tot[i]==-2:
        more=more+1






cm=confusion_matrix(test_labels, predictions_tot, normalize='true')

sum=0
for i in range(6):
    sum=sum+ cm[i,i]

accuracy=sum/6

print('accuracy=', accuracy)


cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, Open set classification with bonafide as known')
plt.savefig('cm_farid_open_btrain.png')




#roc curve
maxproba=np.zeros(len(test_features))
almost_max_proba=np.zeros(len(test_features))
proba=np.zeros(5)
proba_sorted=np.zeros(5)

for i in range(len(test_features)):
    proba[0]=predictions_proba_a09[i][1]
    proba[1] = predictions_proba_a11[i][1]
    proba[2] = predictions_proba_bona[i][1]
    proba[3] = predictions_proba_a14[i][1]
    proba[4] = predictions_proba_a17[i][1]
    proba_sorted=sorted(proba)
    maxproba[i]=proba_sorted[4]
    almost_max_proba[i]=proba_sorted[3]

ratio=almost_max_proba/maxproba

test_labels_roc=np.zeros(len(test_labels))
for i in range(len(test_labels)):
    if (test_labels[i]>=0):
        test_labels_roc[i]=0
    if (test_labels[i]==-1):
        test_labels_roc[i]=1

'''
ORO_features=np.zeros((len(test_features), 3))
for i in range(len(test_features)):
    ORO_features[i, 0]=predictions_one_class[i]
    ORO_features[i, 1]=predictions_tot[i]
    ORO_features[i, 2]=maxproba[i]

np.save('ORO_features.npy', ORO_features)
np.save( 'ORO_labels.npy', test_labels)
'''

false_pos_rate, true_pos_rate, thresholds = \
    roc_curve(test_labels_roc, maxproba, 0)
roc_auc = auc(false_pos_rate, true_pos_rate,)
plt.figure()
plt.plot(false_pos_rate, true_pos_rate, linewidth=5,
         label='AUC = %0.3f'% roc_auc)
plt.plot([0,1],[0,1], linewidth=5)
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('ROC curve, max score, bonafide as known')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('roc_farid_max_btrain.png')






false_pos_rate, true_pos_rate, thresholds = \
    roc_curve(test_labels_roc, ratio)
roc_auc = auc(false_pos_rate, true_pos_rate,)
plt.figure()
plt.plot(false_pos_rate, true_pos_rate, linewidth=5,
         label='AUC = %0.3f'% roc_auc)
plt.plot([0,1],[0,1], linewidth=5)
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('ROC curve, score ratio, bonafide as known')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('roc_farid_ratio_btrain.png')





############################## ONE CLASS ###############################


test_features=np.concatenate((   features_a09[2000:2250],
                              features_a11[2000:2250],

                              features_a14[2000:2250],
                              features_a17[2000:2250],
                                 features_bona_asv[3000:4000],
                              ))
test_labels=np.ones(len(test_features))

for i in range(1000):
    test_labels[i]=-1

class_names=['fake', 'bonafide']

one_class_svm=OneClassSVM(gamma='auto').fit(features_bona_asv[0:3000])
predictions=one_class_svm.predict(test_features)
scores=one_class_svm.score_samples(test_features)

cm=confusion_matrix(test_labels, predictions, normalize='true')

accuracy=(cm[0,0]+cm[1,1])/2

print('accuracy=', accuracy)


cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, One class SVM trained on bonafide features')
plt.savefig('cm_farid_oneclass_bona.png')





#####################################################################
# 2d plot

features=np.concatenate((features_bona_asv[0:4000],
                         features_a09[0:4000],
                         features_a11[0:4000],
                         features_a14[0:4000],
                         features_a17[0:4000]))

pca=PCA(n_components=2)
pca.fit(features)
reduced=pca.transform(features)

reduced_bona=reduced[0:4000]
reduced_a09=reduced[4000:8000]
reduced_a11=reduced[8000:12000]
reduced_a14=reduced[12000:16000]
reduced_a17=reduced[16000:20000]

############################### ICA ###################################

features=np.concatenate((features_bona_asv[0:4000],
                         features_a09[0:4000],
                         features_a11[0:4000],
                         features_a14[0:4000],
                         features_a17[0:4000]))

ica=FastICA(n_components=2, max_iter=10000)
ica.fit(features)
reduced=ica.transform(features)

reduced_bona=reduced[0:4000]
reduced_a09=reduced[4000:8000]
reduced_a11=reduced[8000:12000]
reduced_a14=reduced[12000:16000]
reduced_a17=reduced[16000:20000]


####################### TSNE #####################################

features=np.concatenate((features_bona_asv[0:4000],
                         features_a09[0:4000],
                         features_a11[0:4000],
                         features_a14[0:4000],
                         features_a17[0:4000]))

reduced = TSNE(n_components=2).fit_transform(features)


reduced_bona=reduced[0:4000]
reduced_a09=reduced[4000:8000]
reduced_a11=reduced[8000:12000]
reduced_a14=reduced[12000:16000]
reduced_a17=reduced[16000:20000]

############ 2d plot

plt.figure()
scatterbona=plt.scatter(reduced_bona[:, 0],reduced_bona[:, 1], c='tab:blue', alpha=0.5 )
#scatter09=plt.scatter(reduced_a09[:, 0],reduced_a09[:, 1], c='#000000', alpha=0.5 )
#scatter11=plt.scatter(reduced_a11[:, 0],reduced_a11[:, 1], c='#00ffff', alpha=0.5 )
scatter14=plt.scatter(reduced_a14[:, 0],reduced_a14[:, 1], c='tab:green', alpha=0.5 )
scatter17=plt.scatter(reduced_a17[:, 0],reduced_a17[:, 1], c='tab:orange', alpha=0.5 )
#plt.legend((scatterbona, scatter09, scatter11, scatter14,
#            scatter17), ('bonafide', 'a09', 'a11', 'a14', 'a17'))

plt.legend(( scatterbona, scatter14,
            scatter17), ( 'bonafide', 'a14', 'a17'))


plt.title('2D Scatter plot, features reduced with TSNE')
plt.savefig('scatter_tsne_2.png')

# 3d scatter plot


features=np.concatenate((features_bona_asv[0:4000],
                         features_a09[0:4000],
                         features_a11[0:4000],
                         features_a14[0:4000],
                         features_a17[0:4000]))

pca=PCA(n_components=3)
pca.fit(features)
reduced=pca.transform(features)

reduced_bona=reduced[0:4000]
reduced_a09=reduced[4000:8000]
reduced_a11=reduced[8000:12000]
reduced_a14=reduced[12000:16000]
reduced_a17=reduced[16000:20000]

############################### ICA ###################################

features=np.concatenate((features_bona_asv[0:4000],
                         features_a09[0:4000],
                         features_a11[0:4000],
                         features_a14[0:4000],
                         features_a17[0:4000]))

ica=FastICA(n_components=3, max_iter=10000)
ica.fit(features)
reduced=ica.transform(features)

reduced_bona=reduced[0:4000]
reduced_a09=reduced[4000:8000]
reduced_a11=reduced[8000:12000]
reduced_a14=reduced[12000:16000]
reduced_a17=reduced[16000:20000]


####################### TSNE #####################################

features=np.concatenate((features_bona_asv[0:4000],
                         features_a09[0:4000],
                         features_a11[0:4000],
                         features_a14[0:4000],
                         features_a17[0:4000]))

reduced = TSNE(n_components=3).fit_transform(features)


reduced_bona=reduced[0:4000]
reduced_a09=reduced[4000:8000]
reduced_a11=reduced[8000:12000]
reduced_a14=reduced[12000:16000]
reduced_a17=reduced[16000:20000]


fig = plt.figure()
ax = Axes3D(fig)
scatterbona=ax.scatter(reduced_bona[:,0], reduced_bona[:,1], reduced_bona[:,2], c='tab:blue')
scatter09=ax.scatter(reduced_a09[:,0], reduced_a09[:,1], reduced_a09[:,2], c='tab:green')
scatter11=ax.scatter(reduced_a11[:,0], reduced_a11[:,1], reduced_a11[:,2], c='tab:orange')
scatter14=ax.scatter(reduced_a14[:,0], reduced_a14[:,1], reduced_a14[:,2], c='#000000')
scatter17=ax.scatter(reduced_a17[:,0], reduced_a17[:,1], reduced_a17[:,2], c='#ff0000')

plt.legend((scatterbona, scatter09, scatter11, scatter14,
            scatter17), ('bonafide', 'a09', 'a11', 'a14', 'a17'))

#plt.legend(( scatterbona, scatter14,
#            scatter17), ( 'bonafide', 'a14', 'a17'))
#plt.title('3D Scatter plot, features reduced with TSNE')
plt.savefig('scatter_tsne_3d_3.png')


################################ pairplot ###########################

indexes_bona=[0]*len(features_bona_asv)
indexes_09=[9]*len(features_a09)
indexes_11=[11]*len(features_a11)
indexes_14=[14]*len(features_a14)
indexes_17=[17]*len(features_a17)
indexes=indexes_bona+indexes_09+indexes_11+indexes_14+indexes_17

indexes_array=np.array(indexes)
indexes_array=np.reshape(indexes_array, (len(indexes_array), 1))

features=np.concatenate((features_bona_asv, features_a09,
                         features_a11, features_a14, features_a17))

features_with_labels=np.concatenate((features, indexes_array), axis=1)

columns=['module_mean', 'module_variance', 'module_skewness',
         'module_kurtosis', 'phase_mean', 'phase_variance',
         'phase_skewness', 'phase_kurtosis', 'label']

dataframe=pd.DataFrame(data=features_with_labels, index=indexes,
                       columns=columns)

sns.set(style="ticks", color_codes=True)

g = sns.pairplot(dataframe, hue='label')

g.savefig('pairplot')

#####################################################################
# ORO FEATURES DECISION TREE AND CLUSTERING
#####################################################################

ORO_features=np.load('ORO_features.npy')
ORO_labels=np.load('ORO_labels.npy')

features_bona=ORO_features[0:2000]
features_a09=ORO_features[2000:4000]
features_a11=ORO_features[4000:6000]
features_a14=ORO_features[6000:8000]
features_a17=ORO_features[8000:10000]
features_unknown_spoof=ORO_features[10000:12000]

labels_bona=ORO_labels[0:2000]
labels_a09=ORO_labels[2000:4000]
labels_a11=ORO_labels[4000:6000]
labels_a14=ORO_labels[6000:8000]
labels_a17=ORO_labels[8000:10000]
labels_unknown_spoof=ORO_labels[10000:12000]

train_features=np.concatenate((features_bona[0:1500],
                               features_a09[0:1500],
                               features_a11[0:1500],
                               features_a14[0:1500],
                               features_a17[0:1500],
                               features_unknown_spoof[0:1500]))

test_features=np.concatenate((features_bona[1500:2000],
                               features_a09[1500:2000],
                               features_a11[1500:2000],
                               features_a14[1500:2000],
                               features_a17[1500:2000],
                               features_unknown_spoof[1500:2000]))

train_labels=np.concatenate((labels_bona[0:1500],
                             labels_a09[0:1500],
                             labels_a11[0:1500],
                             labels_a14[0:1500],
                             labels_a17[0:1500],
                             labels_unknown_spoof[0:1500]))


test_labels=np.concatenate((labels_bona[1500:2000],
                             labels_a09[1500:2000],
                             labels_a11[1500:2000],
                             labels_a14[1500:2000],
                             labels_a17[1500:2000],
                             labels_unknown_spoof[1500:2000]))




train_labels_bs=np.zeros(len(train_labels))
train_labels_ku=np.zeros(len(train_labels))
test_labels_bs=np.zeros(len(test_labels))
test_labels_ku=np.zeros(len(test_labels))

# 0 = bonafide, 1 = spoof
for i in range(len(train_labels)):
    if train_labels[i]==0:
        train_labels_bs[i]=0
    if train_labels[i]!=0:
        train_labels_bs[i]=1

# 0 = known,  1 = unknown

for i in range(len(train_labels)):
    if train_labels[i]==-1:
        train_labels_ku[i]= 1
    if train_labels[i]!=-1:
        train_labels_ku[i]=0

# 0 = bonafide, 1 = spoof
for i in range(len(test_labels)):
    if test_labels[i] == 0:
        test_labels_bs[i] = 0
    if test_labels[i] != 0:
        test_labels_bs[i] = 1

# 0 = known,  1 = unknown

for i in range(len(test_labels)):
    if test_labels[i] == -1:
        test_labels_ku[i] = 1
    if test_labels[i] != -1:
        test_labels_ku[i] = 0

# decision tree

tree_bs = tree.DecisionTreeClassifier()
tree_bs.fit(train_features, train_labels_bs)

tree_ku = tree.DecisionTreeClassifier()
tree_ku.fit(train_features, train_labels_ku)

predictions_bs=tree_bs.predict(test_features)
predictions_ku=tree_ku.predict(test_features)

cm_bs=confusion_matrix(test_labels_bs, predictions_bs, normalize='true')
cm_ku=confusion_matrix(test_labels_ku, predictions_ku, normalize='true')

accuracy_bs=(cm_bs[0,0]+cm_bs[1,1])/2
accuracy_ku=(cm_ku[0,0]+cm_ku[1,1])/2

print('accuracy bonafide vs spoof =', accuracy_bs )
print('accuracy known vs unknown = ', accuracy_ku)
class_names_bs=['bonafide', 'spoof']
class_names_ku=['known', 'unknown']

cm_bs_df=pd.DataFrame(cm_bs, index=class_names_bs, columns=class_names_bs )
cm_ku_df=pd.DataFrame(cm_ku, index=class_names_ku, columns=class_names_ku)

plt.figure(figsize=(10,7))
sns.heatmap(cm_bs_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix, Bonafide vs fake, Decision tree')
plt.savefig('cm_farid_ORO_bs.png')

plt.figure(figsize=(10,7))
sns.heatmap(cm_ku_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix, Known vs unknown, Decision tree')
plt.savefig('cm_farid_ORO_ku.png')



# CLUSTERING

ORO_features=np.load('ORO_features.npy')
ORO_labels=np.load('ORO_labels.npy')

features_bona=ORO_features[0:2000]
features_a09=ORO_features[2000:4000]
features_a11=ORO_features[4000:6000]
features_a14=ORO_features[6000:8000]
features_a17=ORO_features[8000:10000]
features_unknown_spoof=ORO_features[10000:12000]

labels_bona=ORO_labels[0:2000]
labels_a09=ORO_labels[2000:4000]
labels_a11=ORO_labels[4000:6000]
labels_a14=ORO_labels[6000:8000]
labels_a17=ORO_labels[8000:10000]
labels_unknown_spoof=ORO_labels[10000:12000]

train_features=np.concatenate((features_bona[0:1500],
                               features_a09[0:1500],
                               features_a11[0:1500],
                               features_a14[0:1500],
                               features_a17[0:1500],
                               features_unknown_spoof[0:1500]))

test_features=np.concatenate((features_bona[1500:2000],
                               features_a09[1500:2000],
                               features_a11[1500:2000],
                               features_a14[1500:2000],
                               features_a17[1500:2000],
                               features_unknown_spoof[1500:2000]))

test_labels=np.concatenate((labels_bona[1500:2000],
                             labels_a09[1500:2000],
                             labels_a11[1500:2000],
                             labels_a14[1500:2000],
                             labels_a17[1500:2000],
                             labels_unknown_spoof[1500:2000]))


test_labels_bs=np.zeros(len(test_labels))
test_labels_ku=np.zeros(len(test_labels))


# 0 = bonafide, 1 = spoof
for i in range(len(test_labels)):
    if test_labels[i] == 0:
        test_labels_bs[i] = 0
    if test_labels[i] != 0:
        test_labels_bs[i] = 1

# 0 = known,  1 = unknown

for i in range(len(test_labels)):
    if test_labels[i] == -1:
        test_labels_ku[i] = 1
    if test_labels[i] != -1:
        test_labels_ku[i] = 0

kmeans2 = KMeans(n_clusters=2, random_state=0).fit(train_features)

predictions=kmeans2.predict(test_features)

test_features0_bona=[]
test_features0_spoof=[]
test_features1_bona=[]
test_features1_spoof=[]

for i in range(len(test_features)):
    if (predictions[i]==0) and (test_labels_bs[i]==0):
        test_features0_bona.append(test_features[i])
    if (predictions[i]==0) and (test_labels_bs[i]==1):
        test_features0_spoof.append(test_features[i])
    if (predictions[i]==1) and (test_labels_bs[i]==0):
        test_features1_bona.append(test_features[i])
    if (predictions[i]==1) and (test_labels_bs[i]==1):
        test_features1_spoof.append(test_features[i])

test_features0_bona_array=np.array(test_features0_bona)
test_features0_spoof_array=np.array(test_features0_spoof)
test_features1_bona_array=np.array(test_features1_bona)
test_features1_spoof_array=np.array(test_features1_spoof)



fig = plt.figure()
ax = Axes3D(fig)
scatter_bona_0=ax.scatter(test_features0_bona_array[:,0], test_features0_bona_array[:,1], test_features0_bona_array[:,2], c='tab:blue')
scatter_spoof_0=ax.scatter(test_features0_spoof_array[:,0], test_features0_spoof_array[:,1], test_features0_spoof_array[:,2], c='#ffff00') #yellow
scatter_bona_1=ax.scatter(test_features1_bona_array[:,0], test_features1_bona_array[:,1], test_features1_bona_array[:,2], c='tab:green')
scatter_spoof_1=ax.scatter(test_features1_spoof_array[:,0], test_features1_spoof_array[:,1], test_features1_spoof_array[:,2], c='#ff0000')#red



plt.legend((scatter_bona_0, scatter_spoof_0, scatter_bona_1, scatter_spoof_1
            ), ('bonafide label 0', 'spoof label 0', 'bonafide label 1', 'spoof label 1'))
plt.title('3D Scatter plot, K-Means Clustering')
plt.savefig('scatter_oro.png')


#####################################################################

# OPEN SET BIRAIN with known-unknown features in the training

# bonafide a09 a11 a14 a17

train_features_bona=np.concatenate((features_bona_asv[0:2999],
                                   features_a09[0:500],
                                   features_a11[0:500],
                                   features_a14[0:500],
                                   features_a17[0:500],
                                   features_a07[0:333],
                                   features_a08[0:333],
                                   features_a10[0:333]

                                   ))
train_labels_bona=np.zeros(len(train_features_bona))
for i in range(2999):
    train_labels_bona[i]=1


train_features_a09=np.concatenate((features_a09[0:2999],
                                   features_a11[0:500],
                                   features_bona_asv[0:500],
                                   features_a14[0:500],
                                   features_a17[0:500],
                                   features_a07[0:333],
                                   features_a08[0:333],
                                   features_a10[0:333]
                                   ))
train_labels_a09=np.zeros(len(train_features_a09))
for i in range(2999):
    train_labels_a09[i]=1

train_features_a11=np.concatenate((features_a11[0:2999],
                                   features_a09[0:500],
                                   features_bona_asv[0:500],
                                   features_a14[0:500],
                                   features_a17[0:500],
                                   features_a07[0:333],
                                   features_a08[0:333],
                                   features_a10[0:333]
                                   ))
train_labels_a11=np.zeros(len(train_features_a11))
for i in range(2999):
    train_labels_a11[i]=1



train_features_a14=np.concatenate((features_a14[0:2999],
                                   features_a09[0:500],
                                   features_a11[0:500],
                                   features_bona_asv[0:500],
                                   features_a17[0:500],
                                   features_a07[0:333],
                                   features_a08[0:333],
                                   features_a10[0:333]
                                   ))
train_labels_a14=np.zeros(len(train_features_a14))
for i in range(2999):
    train_labels_a14[i]=1

train_features_a17=np.concatenate((features_a17[0:2999],
                                   features_a09[0:500],
                                   features_a11[0:500],
                                   features_bona_asv[0:500],
                                   features_a14[0:500],
                                   features_a07[0:333],
                                   features_a08[0:333],
                                   features_a10[0:333]
                                   ))
train_labels_a17=np.zeros(len(train_features_a17))
for i in range(2999):
    train_labels_a17[i]=1


train_features_ku=np.concatenate((features_a07[0:1000],
                                  features_a08[0:1000],
                                  features_a10[0:1000],
                                  features_a09[0:600],
                                  features_a11[0:600],
                                  features_bona_asv[0:600],
                                  features_a14[0:600],
                                  features_a17[0:600]))

train_labels_ku=np.zeros(len(train_features_ku))
for i in range(3000):
    train_labels_ku[i]=1

modelbona=svm.SVC(probability=True)
modelbona.fit(train_features_bona, train_labels_bona)


model09=svm.SVC(probability=True)
model09.fit(train_features_a09, train_labels_a09)

model11=svm.SVC(probability=True)
model11.fit(train_features_a11, train_labels_a11)


model14=svm.SVC(probability=True)
model14.fit(train_features_a14, train_labels_a14)

model17=svm.SVC(probability=True)
model17.fit(train_features_a17, train_labels_a17)

modelku=svm.SVC(probability=True)
modelku.fit(train_features_ku, train_labels_ku)


test_features=np.concatenate((features_a09[3000:3500],
                              features_a11[3000:3500],
                              features_bona_asv[3000:3500],
                              features_a14[3000:3500],
                              features_a17[3000:3500],

                              features_a15[3000:3125],
                              features_a18[3000:3125],
                              features_a19[3000:3125],
                              features_a13[3000:3125],
                              features_a07[3000:3160],
                              features_a08[3000:3160],
                              features_a10[3000:3160]))

test_labels=np.zeros(len(test_features))
for i in range(500):
    test_labels[i]=9
    test_labels[i+500]=11
    test_labels[i+1000]=0
    test_labels[i+1500]=14
    test_labels[i+2000]=17
    test_labels[i+2500]=-1

for i in range(3000,3480):
    test_labels[i]=-2

class_names=['known/unknown', 'unknown', 'bonafide', 'a09', 'a11',  'a14', 'a17']

predictions_a09=model09.predict(test_features)
predictions_a11=model11.predict(test_features)
predictions_bona=modelbona.predict(test_features)
predictions_a14=model14.predict(test_features)
predictions_a17=model17.predict(test_features)
predictions_ku=modelku.predict(test_features)

predictions_proba_a09=model09.predict_proba(test_features)
predictions_proba_a11=model11.predict_proba(test_features)
predictions_proba_bona=modelbona.predict_proba(test_features)
predictions_proba_a14=model14.predict_proba(test_features)
predictions_proba_a17=model17.predict_proba(test_features)
predictions_proba_ku=modelku.predict_proba(test_features)

predictions_tot=np.zeros(len(test_features))

sum=np.zeros(len(test_features))

for i in range(len(test_features)):

    if (predictions_a09[i]==1 and predictions_a11[i]==0 and
            predictions_bona[i]==0 and predictions_a14[i]==0 and
            predictions_a17[i]==0 and predictions_ku[i]==0):

        predictions_tot[i]=9

    if (predictions_a09[i] == 0 and predictions_a11[i] == 1 and
            predictions_bona[i] == 0 and predictions_a14[i] == 0 and
            predictions_a17[i] == 0 and predictions_ku[i]==0):
        predictions_tot[i] = 11

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_bona[i] == 1 and predictions_a14[i] == 0 and
            predictions_a17[i] == 0 and predictions_ku[i]==0):
        predictions_tot[i] = 0

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_bona[i] == 0 and predictions_a14[i] == 1 and
            predictions_a17[i] == 0 and predictions_ku[i]==0):
        predictions_tot[i] = 14

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_bona[i] == 0 and predictions_a14[i] == 0 and
            predictions_a17[i] == 1 and predictions_ku[i]==0):
        predictions_tot[i] = 17

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_bona[i] == 0 and predictions_a14[i] == 0 and
            predictions_a17[i] == 0 and predictions_ku[i]==1):
        predictions_tot[i] = -2 #known unknown class

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_bona[i] == 0 and predictions_a14[i] == 0 and
            predictions_a17[i] == 0 and predictions_ku[i]==0):
        predictions_tot[i] = -1 # unknown class

    sum[i]=sum[i]+predictions_a09[i]+predictions_a11[i]+ \
        predictions_bona[i]+ predictions_a14[i]+ predictions_a17[i]+\
           predictions_ku[i]
    if sum[i] > 1 :
        predictions_tot[i]=-3 #more than one class



max_proba=0.0

for i in range(len(predictions_tot)):
    if (predictions_tot[i]==-3):

        max_proba=max(predictions_proba_a09[i][1],
                         predictions_proba_a11[i][1],
                         predictions_proba_bona[i][1],
                         predictions_proba_a14[i][1],
                         predictions_proba_a17[i][1],
                         predictions_proba_ku[i][1])

        if (max_proba==predictions_proba_a09[i][1]):
            predictions_tot[i]=9
        if (max_proba==predictions_proba_a11[i][1]):
            predictions_tot[i]=11
        if (max_proba==predictions_proba_bona[i][1]):
            predictions_tot[i]=0
        if (max_proba==predictions_proba_a14[i][1]):
            predictions_tot[i]=14
        if (max_proba==predictions_proba_a17[i][1]):
            predictions_tot[i]=17
        if (max_proba==predictions_proba_ku[i][1]):
            predictions_tot[i]=-2



more=0

for i in range(len(predictions_tot)):
    if predictions_tot[i]==-3:
        more=more+1


'''
test_labels=np.zeros(len(test_features))
for i in range(500):
    test_labels[i]=9
    test_labels[i+500]=11
    test_labels[i+1000]=0
    test_labels[i+1500]=14
    test_labels[i+2000]=17
    test_labels[i+2500]=-1

for i in range(3000,3480):
    test_labels[i]=-1
    
    
for i in range(len(predictions_tot)):
    if (predictions_tot[i]==-2 ) :
        predictions_tot[i]=-1
        
class_names=['unknown', 'bonafide',  'a09', 'a11', 'a14', 'a17']

cm=confusion_matrix(test_labels, predictions_tot, normalize='true')

sum=0
for i in range(6):
    sum=sum+ cm[i,i]

accuracy=sum/6

print('accuracy=', accuracy)



cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, Open set classification with known/unknown features and bonafide as known')
plt.savefig('cm_farid_open_btrain_ku2.png')
            
'''
test_labels_c = np.zeros(len(test_features))
for i in range(500):
    test_labels_c[i] = 1
    test_labels_c[i + 500] = 1
    test_labels_c[i + 1000] = 1
    test_labels_c[i + 1500] = 1
    test_labels_c[i + 2000] = 1
    test_labels_c[i + 2500] = -1

for i in range(3000, 3480):
    test_labels_c[i] = -1

predictions_tot_c = np.zeros(len(predictions_tot))

for i in range(len(predictions_tot)):
    if ((predictions_tot[i] == -2) or (predictions_tot[i] == -1)):
        predictions_tot_c[i] = -1
    if ((predictions_tot[i] == 9) or (predictions_tot[i] == 11) or
            (predictions_tot[i] == 0) or (predictions_tot[i] == 14)
            or (predictions_tot[i] == 17)):
        predictions_tot_c[i] = 1

cm_c = confusion_matrix(test_labels_c, predictions_tot_c, normalize='true')

c1 = cm_c[0, 0]
c2 = cm_c[1, 0]


cm=confusion_matrix(test_labels, predictions_tot, normalize='true')

sum=0
for i in range(7):
    sum=sum+ cm[i,i]

accuracy=sum/7

print('accuracy=', accuracy)



cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, Open set classification with known/unknown features and bonafide as known')
plt.savefig('cm_farid_open_btrain_ku1.png')

#roc curve
maxproba=np.zeros(len(test_features))
almost_max_proba=np.zeros(len(test_features))
proba=np.zeros(5)
proba_sorted=np.zeros(5)

for i in range(len(test_features)):
    proba[0]=predictions_proba_a09[i][1]
    proba[1] = predictions_proba_a11[i][1]
    proba[2] = predictions_proba_bona[i][1]
    proba[3] = predictions_proba_a14[i][1]
    proba[4] = predictions_proba_a17[i][1]
    proba_sorted=sorted(proba)
    maxproba[i]=proba_sorted[4]
    almost_max_proba[i]=proba_sorted[3]

ratio=almost_max_proba/maxproba

test_labels_roc=np.zeros(len(test_labels))
for i in range(len(test_labels)):
    if (test_labels[i]>=0):
        test_labels_roc[i]=0
    if (test_labels[i]==-1):
        test_labels_roc[i]=1



false_pos_rate07_max, true_pos_rate07_max, thresholds07_max = \
    roc_curve(test_labels_roc, maxproba, 0)
roc_auc07_max = auc(false_pos_rate07_max, true_pos_rate07_max, )
plt.figure()
plt.plot(false_pos_rate07_max, true_pos_rate07_max, linewidth=5,
         label='Max score, AUC = %0.3f' % roc_auc07_max)

false_pos_rate07_ratio, true_pos_rate07_ratio, thresholds07_ratio = \
    roc_curve(test_labels_roc, ratio)
roc_auc07_ratio = auc(false_pos_rate07_ratio, true_pos_rate07_ratio, )
#plt.figure()
plt.plot(false_pos_rate07_ratio, true_pos_rate07_ratio, linewidth=5,
         label='Score ratio, AUC = %0.3f' % roc_auc07_ratio)

plt.plot([0, 1], [0, 1], linewidth=5)
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.scatter(c2,c1,c='#ff5733' )
#plt.title('ROC curves, maximum score and score ratio, bonafide as unknown')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('roc_farid_btrain_confronto_final_2.png')




false_pos_rate07, true_pos_rate07, thresholds07 = \
    roc_curve(test_labels_roc, maxproba, 0)
roc_auc07 = auc(false_pos_rate07, true_pos_rate07,)
plt.figure()
plt.plot(false_pos_rate07, true_pos_rate07, linewidth=5,
         label='AUC = %0.3f'% roc_auc07)
plt.plot([0,1],[0,1], linewidth=5)
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('ROC curve, max score, bonafide as unknown')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('roc_farid_max_strain.png')

false_pos_rate07, true_pos_rate07, thresholds07 = \
    roc_curve(test_labels_roc, ratio)
roc_auc07 = auc(false_pos_rate07, true_pos_rate07,)
plt.figure()
plt.plot(false_pos_rate07, true_pos_rate07, linewidth=5,
         label='AUC = %0.3f'% roc_auc07)
plt.plot([0,1],[0,1], linewidth=5)
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('ROC curve, score ratio, bonafide as unknown')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('roc_farid_ratio_strain.png')

# OPEN SET STRAIN with known-unknown features in the training

# a09 a11 a13 a14 a17

train_features_a09 = np.concatenate((features_a09[0:2999],
                                     features_a11[0:500],
                                     features_a13[0:500],
                                     features_a14[0:500],
                                     features_a17[0:500],
                                     features_a07[0:333],
                                     features_a08[0:333],
                                     features_a10[0:333]
                                     ))
train_labels_a09 = np.zeros(len(train_features_a09))
for i in range(2999):
    train_labels_a09[i] = 1

train_features_a11 = np.concatenate((features_a11[0:2999],
                                     features_a09[0:500],
                                     features_a13[0:500],
                                     features_a14[0:500],
                                     features_a17[0:500],
                                     features_a07[0:333],
                                     features_a08[0:333],
                                     features_a10[0:333]
                                     ))
train_labels_a11 = np.zeros(len(train_features_a11))
for i in range(2999):
    train_labels_a11[i] = 1

train_features_a13 = np.concatenate((features_a13[0:2999],
                                     features_a09[0:500],
                                     features_a11[0:500],
                                     features_a14[0:500],
                                     features_a17[0:500],
                                     features_a07[0:333],
                                     features_a08[0:333],
                                     features_a10[0:333]

                                     ))
train_labels_a13 = np.zeros(len(train_features_a13))
for i in range(2999):
    train_labels_a13[i] = 1

train_features_a14 = np.concatenate((features_a14[0:2999],
                                     features_a09[0:500],
                                     features_a11[0:500],
                                     features_a13[0:500],
                                     features_a17[0:500],
                                     features_a07[0:333],
                                     features_a08[0:333],
                                     features_a10[0:333]
                                     ))
train_labels_a14 = np.zeros(len(train_features_a14))
for i in range(2999):
    train_labels_a14[i] = 1

train_features_a17 = np.concatenate((features_a17[0:2999],
                                     features_a09[0:500],
                                     features_a11[0:500],
                                     features_a13[0:500],
                                     features_a14[0:500],
                                     features_a07[0:333],
                                     features_a08[0:333],
                                     features_a10[0:333]
                                     ))
train_labels_a17 = np.zeros(len(train_features_a17))
for i in range(2999):
    train_labels_a17[i] = 1

train_features_ku = np.concatenate((features_a07[0:1000],
                                    features_a08[0:1000],
                                    features_a10[0:1000],
                                    features_a09[0:600],
                                    features_a11[0:600],
                                    features_a13[0:600],
                                    features_a14[0:600],
                                    features_a17[0:600]))

train_labels_ku = np.zeros(len(train_features_ku))
for i in range(3000):
    train_labels_ku[i] = 1

model09 = svm.SVC(probability=True)
model09.fit(train_features_a09, train_labels_a09)

model11 = svm.SVC(probability=True)
model11.fit(train_features_a11, train_labels_a11)

model13 = svm.SVC(probability=True)
model13.fit(train_features_a13, train_labels_a13)

model14 = svm.SVC(probability=True)
model14.fit(train_features_a14, train_labels_a14)

model17 = svm.SVC(probability=True)
model17.fit(train_features_a17, train_labels_a17)

modelku = svm.SVC(probability=True)
modelku.fit(train_features_ku, train_labels_ku)

test_features = np.concatenate((features_a09[3000:3500],
                                features_a11[3000:3500],
                                features_a13[3000:3500],
                                features_a14[3000:3500],
                                features_a17[3000:3500],

                                features_a15[3000:3125],
                                features_a18[3000:3125],
                                features_a19[3000:3125],
                                features_bona_asv[3000:3125],
                                features_a07[3000:3160],
                                features_a08[3000:3160],
                                features_a10[3000:3160]))

test_labels = np.zeros(len(test_features))
for i in range(500):
    test_labels[i] = 9
    test_labels[i + 500] = 11
    test_labels[i + 1000] = 13
    test_labels[i + 1500] = 14
    test_labels[i + 2000] = 17
    test_labels[i + 2500] = -1

for i in range(3000, 3480):
    test_labels[i] = -2

class_names = ['known/unknown', 'unknown', 'a09', 'a11', 'a13', 'a14', 'a17']

predictions_a09 = model09.predict(test_features)
predictions_a11 = model11.predict(test_features)
predictions_a13 = model13.predict(test_features)
predictions_a14 = model14.predict(test_features)
predictions_a17 = model17.predict(test_features)
predictions_ku = modelku.predict(test_features)

predictions_proba_a09 = model09.predict_proba(test_features)
predictions_proba_a11 = model11.predict_proba(test_features)
predictions_proba_a13 = model13.predict_proba(test_features)
predictions_proba_a14 = model14.predict_proba(test_features)
predictions_proba_a17 = model17.predict_proba(test_features)
predictions_proba_ku = modelku.predict_proba(test_features)

predictions_tot = np.zeros(len(test_features))

sum = np.zeros(len(test_features))

for i in range(len(test_features)):

    if (predictions_a09[i] == 1 and predictions_a11[i] == 0 and
            predictions_a13[i] == 0 and predictions_a14[i] == 0 and
            predictions_a17[i] == 0 and predictions_ku[i] == 0):
        predictions_tot[i] = 9

    if (predictions_a09[i] == 0 and predictions_a11[i] == 1 and
            predictions_a13[i] == 0 and predictions_a14[i] == 0 and
            predictions_a17[i] == 0 and predictions_ku[i] == 0):
        predictions_tot[i] = 11

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_a13[i] == 1 and predictions_a14[i] == 0 and
            predictions_a17[i] == 0 and predictions_ku[i] == 0):
        predictions_tot[i] = 13

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_a13[i] == 0 and predictions_a14[i] == 1 and
            predictions_a17[i] == 0 and predictions_ku[i] == 0):
        predictions_tot[i] = 14

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_a13[i] == 0 and predictions_a14[i] == 0 and
            predictions_a17[i] == 1 and predictions_ku[i] == 0):
        predictions_tot[i] = 17

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_a13[i] == 0 and predictions_a14[i] == 0 and
            predictions_a17[i] == 0 and predictions_ku[i] == 1):
        predictions_tot[i] = -2  # known unknown class

    if (predictions_a09[i] == 0 and predictions_a11[i] == 0 and
            predictions_a13[i] == 0 and predictions_a14[i] == 0 and
            predictions_a17[i] == 0 and predictions_ku[i] == 0):
        predictions_tot[i] = -1  # unknown class

    sum[i] = sum[i] + predictions_a09[i] + predictions_a11[i] + \
             predictions_a13[i] + predictions_a14[i] + predictions_a17[i] + \
             predictions_ku[i]
    if sum[i] > 1:
        predictions_tot[i] = -3  # more than one class

max_proba = 0.0

for i in range(len(predictions_tot)):
    if (predictions_tot[i] == -3):

        max_proba = max(predictions_proba_a09[i][1],
                        predictions_proba_a11[i][1],
                        predictions_proba_a13[i][1],
                        predictions_proba_a14[i][1],
                        predictions_proba_a17[i][1],
                        predictions_proba_ku[i][1])

        if (max_proba == predictions_proba_a09[i][1]):
            predictions_tot[i] = 9
        if (max_proba == predictions_proba_a11[i][1]):
            predictions_tot[i] = 11
        if (max_proba == predictions_proba_a13[i][1]):
            predictions_tot[i] = 13
        if (max_proba == predictions_proba_a14[i][1]):
            predictions_tot[i] = 14
        if (max_proba == predictions_proba_a17[i][1]):
            predictions_tot[i] = 17
        if (max_proba == predictions_proba_ku[i][1]):
            predictions_tot[i] = -2

more = 0

for i in range(len(predictions_tot)):
    if predictions_tot[i] == -3:
        more = more + 1

'''
test_labels=np.zeros(len(test_features))
for i in range(500):
    test_labels[i]=9
    test_labels[i+500]=11
    test_labels[i+1000]=13
    test_labels[i+1500]=14
    test_labels[i+2000]=17
    test_labels[i+2500]=-1

for i in range(3000,3480):
    test_labels[i]=-1


for i in range(len(predictions_tot)):
    if (predictions_tot[i]==-2 ) :
        predictions_tot[i]=-1

class_names=['unknown', 'a09', 'a11', 'a13', 'a14', 'a17']

cm=confusion_matrix(test_labels, predictions_tot, normalize='true')

sum=0
for i in range(6):
    sum=sum+ cm[i,i]

accuracy=sum/6

print('accuracy=', accuracy)



cm_df=pd.DataFrame(cm, index=class_names, columns=class_names )
plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, Open set classification with known/unknown features and bonafide as unknown')
plt.savefig('cm_farid_open_strain_ku2.png')

'''
# confronto

test_labels_c = np.zeros(len(test_features))
for i in range(500):
    test_labels_c[i] = 1
    test_labels_c[i + 500] = 1
    test_labels_c[i + 1000] = 1
    test_labels_c[i + 1500] = 1
    test_labels_c[i + 2000] = 1
    test_labels_c[i + 2500] = -1

for i in range(3000, 3480):
    test_labels_c[i] = -1

predictions_tot_c = np.zeros(len(predictions_tot))

for i in range(len(predictions_tot)):
    if ((predictions_tot[i] == -2) or (predictions_tot[i] == -1)):
        predictions_tot_c[i] = -1
    if ((predictions_tot[i] == 9) or (predictions_tot[i] == 11) or
            (predictions_tot[i] == 13) or (predictions_tot[i] == 14)
            or (predictions_tot[i] == 17)):
        predictions_tot_c[i] = 1

cm_c = confusion_matrix(test_labels_c, predictions_tot_c, normalize='true')

c1 = cm_c[0, 0]
c2 = cm_c[1, 0]




cm = confusion_matrix(test_labels, predictions_tot, normalize='true')

sum = 0
for i in range(7):
    sum = sum + cm[i, i]

accuracy = sum / 7

print('accuracy=', accuracy)

cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix, Open set classification with known/unknown features and bonafide as unknown')
plt.savefig('cm_farid_open_strain_ku1.png')

# roc curve
maxproba = np.zeros(len(test_features))
almost_max_proba = np.zeros(len(test_features))
proba = np.zeros(5)
proba_sorted = np.zeros(5)

for i in range(len(test_features)):
    proba[0] = predictions_proba_a09[i][1]
    proba[1] = predictions_proba_a11[i][1]
    proba[2] = predictions_proba_a13[i][1]
    proba[3] = predictions_proba_a14[i][1]
    proba[4] = predictions_proba_a17[i][1]
    proba_sorted = sorted(proba)
    maxproba[i] = proba_sorted[4]
    almost_max_proba[i] = proba_sorted[3]

ratio = almost_max_proba / maxproba

test_labels_roc = np.zeros(len(test_labels))
for i in range(len(test_labels)):
    if (test_labels[i] > 0):
        test_labels_roc[i] = 0
    if (test_labels[i] == -1):
        test_labels_roc[i] = 1



false_pos_rate07_max, true_pos_rate07_max, thresholds07_max = \
    roc_curve(test_labels_roc, maxproba, 0)
roc_auc07_max = auc(false_pos_rate07_max, true_pos_rate07_max, )
plt.figure()
plt.plot(false_pos_rate07_max, true_pos_rate07_max, linewidth=5,
         label='Max score, AUC = %0.3f' % roc_auc07_max)

false_pos_rate07_ratio, true_pos_rate07_ratio, thresholds07_ratio = \
    roc_curve(test_labels_roc, ratio)
roc_auc07_ratio = auc(false_pos_rate07_ratio, true_pos_rate07_ratio, )
#plt.figure()
plt.plot(false_pos_rate07_ratio, true_pos_rate07_ratio, linewidth=5,
         label='Score ratio, AUC = %0.3f' % roc_auc07_ratio)


plt.plot([0, 1], [0, 1], linewidth=5)
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.scatter(c2,c1,c='#ff5733' )
#plt.title('ROC curves, max score and score ratio, bonafide as unknown')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('roc_farid_strain_confronto_final_2.png')



false_pos_rate07, true_pos_rate07, thresholds07 = \
    roc_curve(test_labels_roc, maxproba, 0)
roc_auc07 = auc(false_pos_rate07, true_pos_rate07, )
plt.figure()
plt.plot(false_pos_rate07, true_pos_rate07, linewidth=5,
         label='AUC = %0.3f' % roc_auc07)
plt.plot([0, 1], [0, 1], linewidth=5)
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('ROC curve, max score, bonafide as unknown')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('roc_farid_max_strain.png')

false_pos_rate07, true_pos_rate07, thresholds07 = \
    roc_curve(test_labels_roc, ratio)
roc_auc07 = auc(false_pos_rate07, true_pos_rate07, )
plt.figure()
plt.plot(false_pos_rate07, true_pos_rate07, linewidth=5,
         label='AUC = %0.3f' % roc_auc07)
plt.plot([0, 1], [0, 1], linewidth=5)
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('ROC curve, score ratio, bonafide as unknown')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('roc_farid_ratio_strain.png')









a09_bic = np.load('npy_arrays/asv_unet_128/a09_bic_128.npy')
bona_asv=np.load('bona_bic_eval_128.npy')

a09_modules=np.abs(a09_bic)
bona_modules=np.abs(bona_asv)


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(a09_modules[i].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(bona_modules[i].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
#plt.show()
plt.savefig('bicoherences_bona_vs_a09.png')