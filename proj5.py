import scipy.io
import matplotlib.pyplot    as    plt
import matplotlib
import numpy as np
from numpy import *
from scipy import signal
from sklearn import svm
from sklearn.svm import SVC
import cv2
from sklearn.neighbors import KNeighborsClassifier 

            
def get_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def sliding_window(a, window):
    if not hasattr(window, '__iter__'):
        return get_window(a, window)
    for i, win in enumerate(window):
        if win > 1:
            a = a.swapaxes(i, -1)
            a = get_window(a, win)
            a = a.swapaxes(-2, i)
    return a

def plotHist(bin_edges,histogram):
    plt.figure()
    plt.bar(bin_edges[:-1], histogram, width = 0.5, color='#0504aa',alpha=0.7)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value',fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylabel('Frequency',fontsize=15)
    plt.title('Histogram',fontsize=15)
    plt.show()

def post_calculation(pred,actual):

    hits = 0
    FP = 0
    for i in range(len(pred)):
        if np.equal(pred[i], actual[i]):
            hits += 1

        if pred[i] != pred[i]:
            FP += 1
    return (hits/len(pred))*100,(FP/len(pred))*100



def calculation(m,med):
    temp = []
    for i in m:
        if i > med:
            temp.append(1)
        else:
            temp.append(0)
    j = 0
    m = temp
    ans = 0
    for i in m:
        ans = ans + (pow(2, j) * i)
        j = j + 1
    return ans


def getDecimal(window):
    m= np.zeros(8)
    m[0] = window[1][2]
    m[1] = window[2][2]
    m[2] = window[2][1]
    m[3] = window[2][0]
    m[4] = window[1][0]
    m[5] = window[0][0]
    m[6] = window[0][1]
    m[7] = window[0][2]
    temp = calculation(m,window[1][1])
    return temp


#starting the training and testing of the given data

mat_load = scipy.io.loadmat('AR_database.mat')
Tt_dataMatrix = mat_load.get('Tt_dataMatrix')
Tr_dataMatrix = mat_load.get('Tr_dataMatrix')
Train_lbl = mat_load.get('Tr_sampleLabels').reshape(700, 1)
test_lbl = mat_load.get('Tt_sampleLabels').reshape(700, 1)
Train_lbl = np.ravel(Train_lbl)
test_lbl = np.ravel(test_lbl)
Train_lbl = Train_lbl[:50]
test_lbl = test_lbl[:10]
training_h = []

def getLBP(img):
    filtsize=(3,3)
    b = sliding_window(img, filtsize)
    newimg = img
    for x in range(163):
        for y in range(118):
            newimg[x][y] = getDecimal(b[x][y])
    return newimg

print(" Generating LBP for Training  , please wait \n")
for i in range(50):
    image = Tr_dataMatrix[:, i].reshape(120, 165)
    image = np.rot90(image, -1)
    img=getLBP(image)
    histogram, bin_edges = np.histogram(img, bins=256, range=(0, 256))
    #plotHist(bin_edges,histogram)
    training_h.append(np.array(histogram))

test_h = []

print("\n Gathering LBP for Test data , please wait \n")
for i in range(10):
    image = Tt_dataMatrix[:, i].reshape(120, 165)
    image = np.rot90(image, -1)
    img=getLBP(image)
    histogram, bin_edges = np.histogram(img, bins=256, range=(0, 256))
    #plotHist(bin_edges,histogram)
    test_h.append(np.array(histogram))


training_h = np.array(training_h)
test_h = np.array(test_h)

inputs = input("Enter (1) for SVC - rbf , (2) for SVC - linear , (3) for SVC - poly and (4) for KNN  : ")

if inputs == '1':
    print('Classifier: SVM, Kernel selected : rbf')
    d = SVC(kernel='rbf', gamma='auto')
    d.fit(training_h, Train_lbl)
    p = d.predict(test_h)
    p = np.ravel(p)
elif inputs == '2':
    print('Classifier: SVM, Kernel selected : linear')
    d = SVC(kernel='linear', gamma='auto')
    d.fit(training_h, Train_lbl)
    p = d.predict(test_h)
    p = np.ravel(p)
elif inputs == '3':
    print('Classifier: SVM, Kernel selected : poly')
    d = SVC(kernel='poly', gamma='auto')
    d.fit(training_h, Train_lbl)
    p = d.predict(test_h)
    p = np.ravel(p)
elif inputs == '4':
    print('Classifier: KNN, Neighbors = 2 ')
    d = KNeighborsClassifier(n_neighbors=2)
    d.fit(training_h, Train_lbl)
    p = d.predict(test_h)
    p = np.ravel(p)

else:
    print('Sorry wrong choice2 , Please re run the program')
acc,false = post_calculation(p,test_lbl)
print("the predction percentage is = ", acc)
print("false Positive Rate ", false)