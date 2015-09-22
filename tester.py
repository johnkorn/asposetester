#! /usr/bin/env python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
#import lmdb
from sklearn.metrics import confusion_matrix
import scipy.misc
from google.protobuf import text_format
import Image

#sys.path.insert(0, "/home/ekoryagin/digits-2.0/caffe/python")
import caffe
from caffe.proto import caffe_pb2

pref = '/home/koryagin/'
caffe_root = pref + 'digits-2.0/caffe/'

map = '''ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789~!@#$%^&*()-=+/{}[]\?<>'''

MODEL_FILE = pref+'ASPOSE/asposetester/aspose_lenet32_30iter/deploy.prototxt'
PRETRAINED = pref+'ASPOSE/asposetester/aspose_lenet32_30iter/snapshot_iter_322590.caffemodel'

MEAN_FILE = pref+'ASPOSE/asposetester/aspose_lenet32_mean/mean.binaryproto'
#net = caffe.Classifier(MODEL_FILE, PRETRAINED,image_dims=(32, 32))
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
caffe.set_mode_cpu()

y_test = []
y_pred = []                                                                                                                
count=0
correct=0  

def get_transformer(deploy_file, mean_file=None):
    """
    Returns an instance of caffe.io.Transformer
    Arguments:
    deploy_file -- path to a .prototxt file
    Keyword arguments:
    mean_file -- path to a .binaryproto file (optional)
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(
            inputs = {'data': dims}
            )
    t.set_transpose('data', (2,0,1)) # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2,1,0))

    if mean_file:
        # set mean pixel
        with open(mean_file,'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t

def load_image(path, height, width, mode='L'):
    """
    Load an image from disk
    Returns an np.ndarray (channels x width x height)
    Arguments:
    path -- path to an image on disk
    width -- resize dimension
    height -- resize dimension
    Keyword arguments:
    mode -- the PIL mode that the image should be converted to
        (RGB for color or L for grayscale)
    """
    image = Image.open(path)
    image = image.convert(mode)
    image = np.array(image)
    # squash
    image = scipy.misc.imresize(image, (height, width), 'bilinear')
    return image

transformer = get_transformer(MODEL_FILE)

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
      
    image = load_image(filename+'.bmp',32,32)
    
    if image.ndim == 2:
        image = image[:,:,np.newaxis]
    
    dims = transformer.inputs['data'][1:]
    
    image_data = transformer.preprocess('data', image)
    net.blobs['data'].data[0] = image_data
    prediction = net.forward()[net.outputs[-1]]
              
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