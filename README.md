# OCR
Implementation of Optical Character Recognition

# 1. Introduction
Optical character recognition or optical character reader, often known as OCR, refers to the process of analyzing and identifying image files of text data and obtaining text and layout information. The text in the image is also recognized and converted to text from image. 
 
According to the identification scenario, the OCR can be roughly divided into a dedicated OCR that identifies a specific scene and a generalized OCR that identifies multiple scenarios. For example, today's ingenious document identification and license plate recognition are typical examples of dedicated OCR. Universal OCR can be used in more complex scenarios and has greater application potential. However, since the scene of the general picture is not fixed and the layout of the text is various, it is more difficult. According to the content of the recognized picture, the scene can be divided into clear and fixed scenes and simpler scenes.
 
The typical OCR is shown as follows:
 
Input --> Image Preprocessing --> Text Detection --> Text Recognition --> Output
 
Usually, in traditional OCR technology, image preprocessing is used to correct optical imaging problems of images. Common pre-processing processes include geometric transformation (perspective, distortion, rotation, etc.), distortion correction, blur removal, image enhancement, and flat-field correction.
 
Text detection detects the location and extent of the text and its layout. It also usually includes layout analysis and text line detection. The main problems solved by text detection are where there is text and how much the text is.
 
Text recognition is based on text detection, which recognizes text content and converts text information in the image into text information. The main problem solved by text recognition is what each text is. The recognized text usually needs to be checked again to ensure its correctness.

# 2. Background:
This section provides technical background information related to suggested topics and definition of important terminologies used throughout this report:

## 2.1 Preprocessing
 In this part, the most important things are RGB image and connected components.
 ### 2.1.1 RGB image
 The original image is an RGB image. The RGB image can also be called as a true color image. A true color image is an image in which each pixel is specified by three values one each for the red, blue, and green components of the pixel scalar. M-by-n-by-3 array of class unit8, unit16, single, or double whose pixel value specify intensity values [7].
 ### 2.1.2 connected components
 In a binary image, the connected components are cluster of pixel, which all have the value 1, and they can be connected in some way.

## 2.2 Text Recognition:
I mainly used two methods to attempt achieve my goals. 
 ### 2.2.1 Convolutional Neural Network:
  A 5-layer convolutional neural network has been considered at first. Referring to the TensorFlow, I tried to implement a similar structure as the MNIST model has. The model contains 5 layers:
  1.	Layer 1: a convolutional layer computes 32 features using a 5x5 filter with ReLU activation, so that 26 kernels in total. Total parameters in layer = 5*5*32 + 32(Bias terms). Input image of size 32x32x1 gives an output of 28x28x1. (Padding is used to preserve width and height as the input images from MNIST or EMNIST are the size of 28x28).
2.	Layer 2: a max pooling layer with a 2x2 filter and stride of 2. The input from previous layer of size 28x28x32 gets sub-sampled to 14x14x28. Total parameters in layer = 2*32.
3.	Layer 3: a convolutional layer. It is similar to layer 1 and it is used to compute 64 features by using a 5x5 filters. So that, total parameters in layer = 5x5x64+64. ReLU activation has also been used. The output size would be 14x14x64 (padding is added).
4.	Layer 4: a max pooling layer as same as the layer 2. It has a 2x2 filter and stride of 2. Total parameters in layer = 2*64. The image gets sub-sampled again to 7x7x64.
5.	Layer 5: a dense layer with 1024 neurons. The output would be a vector with a shape of 1x1024. Total parameters in layer = 7x7x64x1024.
6.	Then we used a dropout operation to drop elements, a logits operation to obtain probabilities for each class. Then applying argmax function to obtain the final prediction.
The total number of parameters that used in this model is 3213952.

A simple model has been used by Ram et al. This model provides good results used to recognize printed characters:
 ### Ram's Model:
