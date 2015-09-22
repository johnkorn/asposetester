#! /usr/bin/env python
import sys
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import Image
#import lmdb
from sklearn.metrics import confusion_matrix
import scipy.misc
from google.protobuf import text_format

#sys.path.insert(0, "/home/ekoryagin/digits-2.0/caffe/python")
import caffe
from caffe.proto import caffe_pb2

pref = '/home/koryagin/'
caffe_root = pref + 'digits-2.0/caffe/'

map = '''ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789~!@#$%^&*()-=+/{}[]\?<>'''

s1 = "Cc";
s2 = "Il";
s3 = "0Oo";
s4 = "Pp";
s5 = "Ss";
s6 = "Vv";
s7 = "Ww";
s8 = "Xx";
s9 = "Zz";

MODEL_FILE = pref+'ASPOSE/asposetester/aspose_lenet32_30iter/deploy.prototxt'
PRETRAINED = pref+'ASPOSE/asposetester/aspose_lenet32_30iter/snapshot_iter_322590.caffemodel'

MEAN_FILE = pref+'ASPOSE/asposetester/aspose_lenet32_mean/mean.binaryproto'
#blob = caffe.proto.caffe_pb2.BlobProto()
#data = open(, 'rb' ).read()
#blob.ParseFromString(data)
#arr = np.array( caffe.io.blobproto_to_array(blob) )

#net = caffe.Classifier(MODEL_FILE, PRETRAINED)
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
caffe.set_mode_cpu()
  
y_test = []
y_pred = []                                                                                                              
count=0
correct=0  

path=pref+'ASPOSE/data/PublicAPIChars32/'
LABELS_FILE=open(pref+'/ASPOSE/data/allmixed_labels.txt','r')
labels=LABELS_FILE.readlines()

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
# print map
for i in range(len(map)):
    print str(i)+': '+map[i]

#ferr = open("PublicApiErrorsNew.txt","w") 
for file in os.listdir(path):    
    print 'recognizing file ', file  
    expected = map.index(file.split('.')[0][0])
    count+=1
            
    # do recognition    
    #img = caffe.io.load_image(path+file, color=False)
    #grayimg = 255*img[:,:,0]    
    #gi = np.reshape(grayimg, (32,32,1))
    image = load_image(path+file,32,32)
    
    if image.ndim == 2:
        image = image[:,:,np.newaxis]
    
    dims = transformer.inputs['data'][1:]
    
    image_data = transformer.preprocess('data', image)
    net.blobs['data'].data[0] = image_data
    prediction = net.forward()[net.outputs[-1]]
        
    #prediction = net.predict([gi])  # predict takes any number of images, and formats them for the Caffe net automatically
    result = int(labels[prediction[0].argmax()])
    proba = prediction[0][prediction[0].argmax()]
    ind = prediction[0].argsort()[-5:][::-1] # top-5 predictions
         
    expchar = map[expected]
    reschar = map[result]   
    if ((expchar==reschar) or 
        (expchar in s1 and reschar in s1) or
        (expchar in s2 and reschar in s2) or
        (expchar in s3 and reschar in s3) or
        (expchar in s4 and reschar in s4) or
        (expchar in s5 and reschar in s5) or
        (expchar in s6 and reschar in s6) or
        (expchar in s7 and reschar in s7) or
        (expchar in s8 and reschar in s8) or
        (expchar in s9 and reschar in s9)):
        correct+=1
        print 'SUCCESS! predicted class: \'{0}\'; prob: {1:.2f}'.format(map[result], proba)                
    else: 
        print 'FAIL! {3} - expected: \'{0}\', predicted: \'{1}\'; prob: {2:.2f}'.format(map[expected], map[result], proba, file)
        #ferr.write(path+file+'\n')
        #shutil.copy(path+file, pref+'/ASPOSE/data/PublicApiErrorsNew/')
        #im = Image.fromarray(grayimg.astype(np.uint8))
        #im.save(pref+'/ASPOSE/data/PublicApiErrorsNew/'+file)
    print 'Current score: {0} of {1} ({2:.2f}%)'.format(correct, count, 100.0*correct/count)

#ferr.close()
print 'Quality is {0:.2f}%'.format(100.0*correct/count)
print 'Correctly recognized {0} of total {1} images'.format(correct, count)


