#! /usr/bin/env python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
#import lmdb
from sklearn.metrics import confusion_matrix

#sys.path.insert(0, "/home/ekoryagin/digits-2.0/caffe/python")
import caffe

pref = '/home/koryagin/'
caffe_root = pref + 'digits-2.0/caffe/'

map = '''ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789~!@#$%^&*()-=+/{}[]\?<>'''

MODEL_FILE = pref+'ASPOSE/asposetester/aspose_lenet32_30iter/deploy.prototxt'
PRETRAINED = pref+'ASPOSE/asposetester/aspose_lenet32_30iter/snapshot_iter_322590.caffemodel'

net = caffe.Classifier(MODEL_FILE, PRETRAINED,image_dims=(32, 32))
caffe.set_mode_cpu()

y_test = []
y_pred = []                                                                                                                
count=0
correct=0  

TEST_FILE=open(pref+'/ASPOSE/data/allmixed_test.txt','r') 
LABELS_FILE=open(pref+'/ASPOSE/data/allmixed_labels.txt','r')
labels=LABELS_FILE.readlines()
TEST_IMAGES=TEST_FILE.readlines()
for i in range(len(TEST_IMAGES)):
    filename=TEST_IMAGES[i].split('.bmp')[0]  
    print 'recognizing file ', TEST_IMAGES[i]  
    parts = filename.split('-')            
    expected = int(parts[-1])
    y_test.append(expected)
    count+=1
            
    # do recognition    
    img = caffe.io.load_image(filename+'.bmp', color=False)

    grayimg = 255*img[:,:,0]
    gi = np.reshape(grayimg, (32,32,1))

    prediction = net.predict([gi])  # predict takes any number of images, and formats them for the Caffe net automatically
    result = int(labels[prediction[0].argmax()])
    proba = prediction[0][prediction[0].argmax()]
    ind = prediction[0].argsort()[-5:][::-1] # top-5 predictions
            
    y_pred.append(result)
            
    if (expected==result):
        correct+=1
        print 'SUCCESS! predicted class: {0}; prob: {1:.2f}'.format(result, proba)                
    else: 
        print 'FAIL! expected: {0}, predicted: {1}; prob: {2:.2f}'.format(expected, result, proba)
    print 'Current score: {0} of {1} ({2:.2f}%)'.format(correct, count, 100.0*correct/count)
            #print 'predicted class: {0}; prob: {1}'.format(prediction[0].argmax(), proba)
            #print 'top 5 predictions: ', ind                                                                

print 'Quality is {0:.2f}%'.format(100.0*correct/count)
print 'Correctly recognized {0} of total {1} images'.format(correct, count)


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(map))
    plt.xticks(tick_marks, map)
    plt.yticks(tick_marks, map)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)    
np.set_printoptions(precision=2, threshold=np.nan)
#print('Confusion matrix, without normalization')
#print(cm)
plt.figure()
plot_confusion_matrix(cm)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#print('Normalized confusion matrix')
#print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

np.savetxt('cm_mixedtest.txt', cm, delimiter=' ')
np.savetxt('cm_mixedtest.txt', cm_normalized, fmt='%.2f',delimiter=' ')

#argmaxes = cm_normalized.argmax(axis=1)
#maxes = cm_normalized.max(axis=1)
tops = np.argsort(cm_normalized, axis=1)
n=len(map)
for i in range(len(map)):
    print "{0}: {1} ({2:.2f}), {3} ({4:.2f}), {5} ({6:.2f})".format(map[i],map[tops[i][-1]],cm_normalized[i][tops[i][-1]], map[tops[i][-2]],cm_normalized[i][tops[i][-2]], map[tops[i][-3]],cm_normalized[i][tops[i][-3]])

plt.show()