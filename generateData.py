import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from scipy import signal

def filter(input_signal,N=8,wn=(3,50),fs=500):
    sos = signal.butter(N, wn, 'bandpass',fs=fs,output='sos')
    output_signal = []
    for i in range(input_signal.shape[0]):
        output_signal.append( signal.sosfilt(sos, input_signal[i,:]))
    return np.asarray(output_signal)
def generate(input,originalPath,targetPath,size=(250,250)):
    count = 0
    for i in range(0,input.shape[0]-size[0],size[0]):#inline dim
        for j in range(0,input.shape[1],size[0]):#time dim
            #print(range(0,input.shape[1],size[0]))
            slice = input[i:i+size[0],j:j+size[1]]
            noise = np.random.normal(0, np.random.randint(0,2), size=slice.shape)
            np.save(targetPath+'_'+str(count),input[i:i+size[0],j:j+size[1]])
            slice = slice + noise
            np.save(originalPath+'_'+str(count),filter(slice/np.max(slice)))
            count  = count + 1
            #print(count)

if __name__ == "__main__":
    fileList = []
    for name in glob.glob('dataset/*.npy'):
        fileList.append(name)
    test = np.load(fileList[100])
    test = [i for i in test]
    test = np.asarray(test)
    test = test/np.max(test)
    noise = np.random.normal(0, 1, size=test.shape)*2
    noise = filter(noise) + test
    noise = noise/np.max(noise)
    #print(test.shape)
    plt.figure(figsize=(20,12))
    plt.subplot(1, 2, 1)
    p1 = plt.imshow(np.transpose(test),interpolation="bilinear",cmap=plt.cm.gray,aspect=3)
    plt.colorbar(shrink=0.4)
    p1 = plt.subplot(1, 2, 2)
    plt.imshow(np.transpose(noise+test),interpolation="bilinear",cmap=plt.cm.gray,aspect=3)
    plt.colorbar(shrink=0.4)
    #plt.show()

    origpath = "dataset/orig/"
    targetPath = "dataset/target/"

    for i in fileList:
        #print(i.split('\\'))
        name = i.split('\\')[1]
        data = np.load(i)
        #data = np.transpose(data)
        data = data/np.max(data)
        print(data.shape)
        generate(data,origpath+name.split('.')[0],targetPath+name.split('.')[0])
        #print(i)


