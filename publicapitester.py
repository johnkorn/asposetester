#! /usr/bin/env python
import sys
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import Image
#import lmdb
from sklearn.metrics import confusion_matrix

#sys.path.insert(0, "/home/ekoryagin/digits-2.0/caffe/python")
import caffe

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

MODEL_FILE = pref+'ASPOSE/asposetester/aspose_lenet32-conv50-100/deploy.prototxt'
PRETRAINED = pref+'ASPOSE/asposetester/aspose_lenet32-conv50-100/snapshot_iter_322590.caffemodel'

net = caffe.Classifier(MODEL_FILE, PRETRAINED)
caffe.set_mode_cpu()
  
y_test = []
y_pred = []                                                                                                              
count=0
correct=0  

path=pref+'/ASPOSE/data/PublicAPIChars32/'
LABELS_FILE=open(pref+'/ASPOSE/data/allmixed_labels.txt','r')
labels=LABELS_FILE.readlines()

# print map
for i in range(len(map)):
    print str(i)+': '+map[i]

ferr = open("PublicApiErrorsNew.txt","w") 
for file in os.listdir(path):    
    #print 'recognizing file ', file  
    expected = map.index(file[0])
    count+=1
            
    # do recognition    
    img = caffe.io.load_image(path+file, color=False)

    grayimg = 255*img[:,:,0]
    
    gi = np.reshape(grayimg, (32,32,1))
    
    prediction = net.predict([gi])  # predict takes any number of images, and formats them for the Caffe net automatically
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
        #print 'SUCCESS! predicted class: \'{0}\'; prob: {1:.2f}'.format(result, proba)                
    else: 
        print 'FAIL! {3} - expected: \'{0}\', predicted: \'{1}\'; prob: {2:.2f}'.format(map[expected], map[result], proba, file)
        ferr.write(path+file+'\n')
        #shutil.copy(path+file, pref+'/ASPOSE/data/PublicApiErrorsNew/')
        im = Image.fromarray(grayimg.astype(np.uint8))
        im.save(pref+'/ASPOSE/data/PublicApiErrorsNew/'+file)
    #print 'Current score: {0} of {1} ({2:.2f}%)'.format(correct, count, 100.0*correct/count)

ferr.close()
print 'Quality is {0:.2f}%'.format(100.0*correct/count)
print 'Correctly recognized {0} of total {1} images'.format(correct, count)


