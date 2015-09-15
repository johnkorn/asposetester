#! /usr/bin/env python
import sys
import os
import shutil

pref = '/home/koryagin/ASPOSE/data'
dir = pref + '/scanned/32x32'
savepath = pref + '/ALL_MIXED'
r = [] 
labels = []                                                                                                           
subdirs = [x[0] for x in os.walk(dir)]     
count=0                               
for subdir in subdirs:
    subsubdirs = [x[0] for x in os.walk(subdir)]
    for subsubdir in subsubdirs:
        files = os.walk(subsubdir).next()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:
                fpath = subsubdir + "/" + file                                                                                                        
                filename, file_extension = os.path.splitext(file)           
                parts = filename.split('-')
                expected = parts[-1]
                newfname = savepath + "/" + expected + "/" + file
                shutil.copyfile(fpath, newfname)
                count+=1 
                print 'Progress: ', count

print 'Done!'