import soundfile as sf
#from PIL import Image
from scipy import signal
import scipy as sp
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras

win_len=256
overlap=128


def compute_bicoherence(audio, rate, nperseg=256, noverlap=128):
    """ Compute the bicoherence between two signals of the same lengths s1 and s2
    using the function scipy.signal.spectrogram
    """

    # compute the stft
    f_axis, t_axis, spec = signal.spectrogram(audio, fs=rate, nperseg=nperseg, noverlap=noverlap, mode='complex',
                                              return_onesided=False)

    # transpose (f, t) -> (t, f)
    spec = np.transpose(spec, [1, 0])

    # compute the bicoherence
    arg = np.arange(-f_axis.size / 4, f_axis.size / 4, dtype='int')
    sumarg = arg[:, None] + arg[None, :]

    num = np.mean(spec[:, arg, None] * spec[:, None, arg] *
                  np.conjugate(spec[:, sumarg]),
                  axis=0)

    denum = np.sqrt(np.mean(
        np.abs(spec[:, arg, None] * spec[:, None, arg]) ** 2, axis=0) * np.mean(
        np.abs(np.conjugate(spec[:, sumarg])) ** 2,
        axis=0))
    bicoh = num / denum
    # bicoh = sp.fftpack.fftshift(bicoh)
    # f1 = sp.fftpack.fftshift(f1)

    #return f_axis[arg], f_axis[sumarg], bicoh
    return bicoh






file_test=open("eval.txt", "r")
lines_test=file_test.readlines()
filecodes_test=["" for i in range(len(lines_test))]
filelabels_test=np.zeros(len(lines_test))


for x in range(len(lines_test)):
    line_test=lines_test[x]
    code_test=line_test[13:20]
    filecodes_test[x]=code_test
    label_test=line_test[23:28]
    if label_test=="- bon":
        filelabels_test[x] = 0
        #bon=bon+1
    if label_test=="A07 s":
       filelabels_test[x]=7
    if label_test=="A08 s":
       filelabels_test[x]=8
    if label_test=="A09 s":
       filelabels_test[x]=9
    if label_test=="A10 s":
       filelabels_test[x]=10
    if label_test=="A11 s":
       filelabels_test[x]=11
    if label_test=="A12 s":
       filelabels_test[x]=12
    if label_test=="A13 s":
       filelabels_test[x]=13
    if label_test=="A14 s":
       filelabels_test[x]=14
    if label_test=="A15 s":
       filelabels_test[x]=15
    if label_test=="A16 s":
       filelabels_test[x]=16
    if label_test=="A17 s":
       filelabels_test[x]=17
    if label_test=="A18 s":
       filelabels_test[x]=18
    if label_test=="A19 s":
       filelabels_test[x]=19

bonafide_codes_test=[]
bonafide_labels_test=[]

a07_codes=[]
a07_labels=[]

a08_codes=[]
a08_labels=[]

a09_codes=[]
a09_labels=[]

a10_codes=[]
a10_labels=[]

a11_codes=[]
a11_labels=[]

a12_codes=[]
a12_labels=[]

a13_codes=[]
a13_labels=[]

a14_codes=[]
a14_labels=[]

a15_codes=[]
a15_labels=[]

a16_codes=[]
a16_labels=[]

a17_codes=[]
a17_labels=[]

a18_codes=[]
a18_labels=[]

a19_codes=[]
a19_labels=[]



for i in range(len(filecodes_test)):
    if filelabels_test[i]==0:
        bonafide_codes_test.append(filecodes_test[i])
        bonafide_labels_test.append(0)
    if filelabels_test[i]==7:
        a07_codes.append(filecodes_test[i])
        a07_labels.append(7)
    if filelabels_test[i]==8:
        a08_codes.append(filecodes_test[i])
        a08_labels.append(8)
    if filelabels_test[i]==9:
        a09_codes.append(filecodes_test[i])
        a09_labels.append(9)
    if filelabels_test[i]==10:
        a10_codes.append(filecodes_test[i])
        a10_labels.append(10)
    if filelabels_test[i]==11:
        a11_codes.append(filecodes_test[i])
        a11_labels.append(11)
    if filelabels_test[i]==12:
        a12_codes.append(filecodes_test[i])
        a12_labels.append(12)
    if filelabels_test[i]==13:
        a13_codes.append(filecodes_test[i])
        a13_labels.append(13)
    if filelabels_test[i]==14:
        a14_codes.append(filecodes_test[i])
        a14_labels.append(14)
    if filelabels_test[i]==15:
        a15_codes.append(filecodes_test[i])
        a15_labels.append(15)
    if filelabels_test[i]==16:
        a16_codes.append(filecodes_test[i])
        a16_labels.append(16)
    if filelabels_test[i]==17:
        a17_codes.append(filecodes_test[i])
        a17_labels.append(17)
    if filelabels_test[i]==18:
        a18_codes.append(filecodes_test[i])
        a18_labels.append(18)
    if filelabels_test[i]==19:
        a19_codes.append(filecodes_test[i])
        a19_labels.append(19)




bona_bic=np.zeros((len(bonafide_codes_test), int(win_len/2), int(win_len/2)), np.cfloat)
a07_bic=np.zeros((len(a07_codes), int(win_len/2), int(win_len/2)), np.cfloat)
a08_bic=np.zeros((len(a08_codes), int(win_len/2), int(win_len/2)), np.cfloat)
a09_bic=np.zeros((len(a09_codes), int(win_len/2), int(win_len/2)), np.cfloat)
a10_bic=np.zeros((len(a10_codes), int(win_len/2), int(win_len/2)), np.cfloat)
a11_bic=np.zeros((len(a11_codes), int(win_len/2), int(win_len/2)), np.cfloat)
a12_bic=np.zeros((len(a12_codes), int(win_len/2), int(win_len/2)), np.cfloat)
a13_bic=np.zeros((len(a13_codes), int(win_len/2), int(win_len/2)), np.cfloat)
a14_bic=np.zeros((len(a14_codes), int(win_len/2), int(win_len/2)), np.cfloat)
a15_bic=np.zeros((len(a15_codes), int(win_len/2), int(win_len/2)), np.cfloat)
a16_bic=np.zeros((len(a16_codes), int(win_len/2), int(win_len/2)), np.cfloat)
a17_bic=np.zeros((len(a17_codes), int(win_len/2), int(win_len/2)), np.cfloat)
a18_bic=np.zeros((len(a18_codes), int(win_len/2), int(win_len/2)), np.cfloat)
a19_bic=np.zeros((len(a19_codes), int(win_len/2), int(win_len/2)), np.cfloat)


for i in range(0, len(bonafide_codes_test)):
    audio, fs = sf.read('eval/flac/LA_E_' + bonafide_codes_test[i] + '.flac')
    bona_bic[i, :, :] = compute_bicoherence(audio, fs)
print('bona ok')

for i in range(0, len(a07_codes)):
    audio, fs = sf.read('eval/flac/LA_E_' + a07_codes[i] + '.flac')
    a07_bic[i, :, :] = compute_bicoherence(audio, fs)
print('a07 ok')

for i in range(0, len(a08_codes)):
    audio, fs = sf.read('eval/flac/LA_E_' + a08_codes[i] + '.flac')
    a08_bic[i, :, :] = compute_bicoherence(audio, fs)
print('a08 ok')
for i in range(0, len(a09_codes)):
    audio, fs = sf.read('eval/flac/LA_E_' + a09_codes[i] + '.flac')
    a09_bic[i, :, :] = compute_bicoherence(audio, fs)
print('a09 ok')
for i in range(0, len(a10_codes)):
    audio, fs = sf.read('eval/flac/LA_E_' + a10_codes[i] + '.flac')
    a10_bic[i, :, :] = compute_bicoherence(audio, fs)
print('a10 ok')
for i in range(0, len(a11_codes)):
    audio, fs = sf.read('eval/flac/LA_E_' + a11_codes[i] + '.flac')
    a11_bic[i, :, :] = compute_bicoherence(audio, fs)
print('a11 ok')
for i in range(0, len(a12_codes)):
    audio, fs = sf.read('eval/flac/LA_E_' + a12_codes[i] + '.flac')
    a12_bic[i, :, :] = compute_bicoherence(audio, fs)
print('a12 ok')
for i in range(0, len(a13_codes)):
    audio, fs = sf.read('eval/flac/LA_E_' + a13_codes[i] + '.flac')
    a13_bic[i, :, :] = compute_bicoherence(audio, fs)
print('a13 ok')
for i in range(0, len(a14_codes)):
    audio, fs = sf.read('eval/flac/LA_E_' + a14_codes[i] + '.flac')
    a14_bic[i, :, :] = compute_bicoherence(audio, fs)
print('a14 ok')
for i in range(0, len(a15_codes)):
    audio, fs = sf.read('eval/flac/LA_E_' + a15_codes[i] + '.flac')
    a15_bic[i, :, :] = compute_bicoherence(audio, fs)
print('a15 ok')
for i in range(0, len(a16_codes)):
    audio, fs = sf.read('eval/flac/LA_E_' + a16_codes[i] + '.flac')
    a16_bic[i, :, :] = compute_bicoherence(audio, fs)
print('a16 ok')
for i in range(0, len(a17_codes)):
    audio, fs = sf.read('eval/flac/LA_E_' + a17_codes[i] + '.flac')
    a17_bic[i, :, :] = compute_bicoherence(audio, fs)
print('a17 ok')
for i in range(0, len(a18_codes)):
    audio, fs = sf.read('eval/flac/LA_E_' + a18_codes[i] + '.flac')
    a18_bic[i, :, :] = compute_bicoherence(audio, fs)
print('a18 ok')

for i in range(0, len(a19_codes)):
    audio, fs = sf.read('eval/flac/LA_E_' + a19_codes[i] + '.flac')
    a19_bic[i, :, :] = compute_bicoherence(audio, fs)
print('a19 ok')



np.save('bona_bic_eval_128', bona_bic)
np.save('a07_bic_128', a07_bic)
np.save('a08_bic_128', a08_bic)
np.save('a09_bic_128', a09_bic)
np.save('a10_bic_128', a10_bic)
np.save('a11_bic_128', a11_bic)
np.save('a12_bic_128', a12_bic)
np.save('a13_bic_128', a13_bic)
np.save('a14_bic_128', a14_bic)
np.save('a15_bic_128', a15_bic)
np.save('a16_bic_128', a16_bic)
np.save('a17_bic_128', a17_bic)
np.save('a18_bic_128', a18_bic)
np.save('a19_bic_128', a19_bic)