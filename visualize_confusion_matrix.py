import numpy as np
import scipy
#import cv2
from random import shuffle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import itertools
import pickle
from sklearn import *
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn import *
from sklearn.metrics import confusion_matrix
import h5py


# reading true labels for the test set
h5f = h5py.File('dataset/data.h5','r')
true_labels = h5f['y_test'][:].astype(np.int64)	


# reading predicted labels
y_pred = np.zeros_like(true_labels)
c = 0
with open('/Users/azarf/Documents/Courses/Winter2016/CS231N/project/CS231n/deep_y_pred.txt', 'r') as f:
	for line in f:
		y_pred[c] = int(line.rstrip('\n')) - 1
		c += 1


pred_labels = list(y_pred)
true_labels = list(true_labels)

# build confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
    
# plot confusion matrix
fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(cm), cmap=plt.cm.jet, 
                interpolation='nearest')

width = len(cm)
height = len(cm[0])

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(cm[x][y]), xy=(y, x), 
                    horizontalalignment='center',verticalalignment='center')
cb = fig.colorbar(res)
marked=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
tick_marks = np.arange(len(marked))
plt.xticks(tick_marks, marked, rotation=45,fontsize=10)
plt.yticks(tick_marks, marked,fontsize=10)
plt.xlabel('Predicted labels', fontsize=12)
ax.xaxis.set_label_position('top') 
plt.ylabel('True labels',fontsize=12)
plt.savefig('confusion_matrix_deep_model.png', format='png')
