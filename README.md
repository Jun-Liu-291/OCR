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
 ### 2.2.2 Ram's Model:
  Their model will reshape an input image to a 32 x 32 pixels image. This image would be reshaped again to a vector with dimensions 1 x 1024. After this simple preprocess, an input image now has been converted to an input vector. Then, they built a very big matrix (weights) with dimensions 1024 x 62 and a bias with dimensions 1 x 62. Now, their idea is very clear:

Input Vector(1,1024) × Weights(1024,62)+Bias(1,62)=Classes(1,62)

The output is a vector of 62 classes. Each element in the output vector represents a probability of the corresponding class.  The total number of parameters is 65598.
Training model:
Ram’s model has been trained on MNIST dataset as well. Training process was based on gradient descent.
#### 2.2.2.1 Gradient Descent (GD):
The network’s weights and biases will be updated by applying gradient descent.
We often use gradient descent (GD) to minimize costs. For each iteration, we update the weight W based on En(fw):
█(E_n (f)=1/n ∑_(i=1)^n▒〖l(f(x_i ),y_i ),〗#(2.1) )
This equation is a cost function.
Then the updated weight W would be:
█(w_(t+1)=w_t-r 1/n ∑_(i=1)^n▒∇_w  Q(z_i,w_t ),#(2.2) )

Where r is the learning rate [1]. 
If the initial value w0 is chosen properly and the learning rate (gain) is chosen small enough, this algorithm can satisfy the linear convergence, that is, − log ρ ∼ t, where ρ represents the residual error.
Then disadvantages of this method can be observed: when the number of samples m is large, all samples need to be calculated for each iteration, and the training process will be slow. Due to the large number of samples, a simplified method, SGD with faster training process, had been used to train Ram’s model.
#### 2.2.2.2 Stochastic Gradient Descent (SGD):
SGD is an important simplification. In each iteration, the gradient estimation does not calculate En(fw) directly but based on a sample of zt that is randomly selected.

█(w_(t+1)=w_t-r_t ∇_w Q(z_i,w_t ),#(2.3) )

The stochastic process {wt: t= 1, …} depends on the examples randomly picked at each iteration. 
Since the loss function is not optimized on all training data, but in each iteration, the loss function is optimized by a randomly selected training data. So that the update speed of each round of parameters is greatly accelerated but problems occur.  The model accuracy would be decreased, and the model may converge to a local optimum because a single sample cannot represent the trend of the entire sample [1].

# 3. Methodology
This section provides a description of Python experiments designed to investigate the feasibility of CNN based and shallow neural network based optical character recognition. It explains dataset used for training, preprocessing procedure for input image standardization and detailed experiment configuration.

## 3.1 Datasets:
The CNN and shallow neural network both were trained on MNIST dataset and EMNIST dataset. 
### 3.1.1 MNIST dataset:
MNIST dataset is a large dataset of handwritten digits that is commonly used for training various image processing systems and it is modified by National Institute of Standards and Technology. This dataset contains digits from 0 to 9, meaning 10 patterns in total and it consists of pair, ‘image’ and its corresponding ‘label’. Each image is a gray scaled image with size 28 x 28 pixels [2]. 
