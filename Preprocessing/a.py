import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

def rgb2gray(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = np.rint(gray)
    return gray
    

img = cv2.imread('original_img.jpg')

#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = rgb2gray(img)

h,w = np.shape(img)

# histogram
his = np.zeros(256)
for a in range(h):
    for b in range(w):
        for n in range(256):
            if img[a,b] == n:
                his[n] += 1
                
        
plt.plot(range(256),his)

# binary
#for n in range(h):
    #for m in range(w):
        #if img[n,m] <= 170:
            #img[n,m] = 0
        #else:
            #img[n,m] = 255
            
# denoising
#for n in range(h-2):
    #for m in range(w-2):
        #img[n+1,m+1] = np.median(sorted([img[n,m],img[n+1,m],img[n+2,m],img[n,m+1],img[n+1,m+1],img[n+2,m+1],img[n,m+2],img[n+1,m+2],img[n+2,m+2]]))



#plt.imshow(img,cmap=plt.get_cmap('gray'))
plt.show()