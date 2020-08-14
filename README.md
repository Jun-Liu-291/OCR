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
### 3.1.2 EMNIST dataset:
EMNIST is an expanded MNIST. The dataset has six different splits, each split consists of different numbers of classes and different numbers of characters. In this project, the EMNIST ByClass, containing 62 classes, and the EMNIST Letters, containing 26 classes, had been used for training process. The EMNIST dataset has a same format as MNIST does. It also consists of pair, ‘image’ and its ‘label’. Each image is a grey scaled image with size 28 x 28 pixels as well. In our project, training part, I used EMNIST Letters dataset to train our model for 5101 times and validate the model in the same dataset. The other split of EMNIST dataset had been used by Ram. They did not use all of the EMNIST but used some of that whose training set containing 50,000 2-tuples data with 62 classes and testing set & validation set containing 10,000 2-tuples data with 62 classes [3]. 

## 3.2 Preporcessing:
Before we start our recognition part, we need to get the images of a single character.
### 3.2.1 Grayscale image
The original images are RGB model images taken by camera in our project. To recognize the text in the images, the first thing we need to do is convert the original images to binary images, which can reduce the amount of calculation. But before convert it to binary images, we need to convert it to 8-bit grayscale images. If the number of the pixels in original image is n×m,   RGB image is an 8-bit n-by-m-by-3 data array stored in computer, while an 8-bit n-by-m data array. And a widely used equation to convert RGB image to grayscale image is,
█(gray=0.2989×R+0.5870×G+0.1141×B,#(2.4) )
which is also applied to our project.
### 3.2.2 Binarization
After we get the 8-bit grayscale image, we can convert it to a binary image. In our project, our aim is to recognize characters in the images, and in this case, for most of our original images, the background is close to white, which has the value close to 255 in the array, and the characters is close to black, which has the value close to zero. But in binary images, array only have 0 and 1, while 0 represents white and 1 represents black. As a consequence, we need to set a threshold to segment the grayscale image, let the pixels belong to characters equal to 1, and all the other background pixels equal to 0. After we get the histogram of many grayscale images, we set the value of the threshold as the end of the first wave’s value.
### 3.2.3 Denoising
Since after we get the binary image, we find that there are salt and paper noise on our image. And the best way denoising this kind of noise is to use a 3-by-3 median filter.  As we know, each pixels inside the image has 8 neighborhood, the main idea of median filter is to replace the original pixel’s value as the median value of these 9 pixels. Since for salt and paper noise, the pixels equal to 1 will appear alone, and the median value in this region will be 0, that’s how we get rid of salt and paper noise.
### 3.2.4 Rotation
Since our images are taken by hand using a smart phone or a camera, the image of text might be slightly tilted. And in this case, it might be difficult for us to separate each lines and each characters, that’s why we need to rotate the binary image to make it parallel. The algorithm we use to achieve this purpose is get the new array of each rotated images and calculate the sum of each lines’ pixels and count the number of the lines whose sum is equal to 0 and take the image who has the biggest number of lines are all zero as our parallel images.
In case the degree of tilt is big, at first we rotate the original image from -10^° to 10^° with 2^° step size. After we get the new image, we rotate the image  -1^° to 1^° with 〖0.1〗^° step size. And we can get our final parallel image.
### 3.2.5 Lines segmentation
Since the distance between lines is quit huge, and after getting the rotated parallel image, we can separate the lines by cutting along x-axis. Using the algorithm same as the rotate part, calculate the sum of each lines, and we can get the histogram of this set of data. For each line, multiple waves will be generated. Each waves represents a line of text. For lines only have lower case letters, and also without long letters like “h”, “l” and “t”, but have “i” or “j”, in this case, one line will generate to waves in the histogram. Therefore, only search for the waves will not satisfy our purpose. But the distance between these two waves can’t be compare to the distance between two different lines. So the only thing we need to avoid this case is set a threshold, if the distance between two waves is smaller than this threshold we will set this two waves as one line. And finally we set this threshold as the average distance between each waves.  
### 3.2.6 Charaters segmentation
After we successfully separate each lines, we can start to separate characters in each lines. But to complete this purpose, we can’t use the same method to separate lines, because lines are totally separate with each other, while the character are not always separated for some kinds of fonts, especially for some strings like “th”. In this case, we have to find the connected components in each lines, because for most characters and digits, they can be seen as a single connected area. And the algorithm to find connected components is row-by-row search. The main idea of this algorithm is to find each connected components and label them.
We start to search the first line of our array, we will label the first pixel whose value is 1 as label “1”, if the next pixels is also 1 we will label it the same as the previous label, until we find a 0 value pixels. At this time we would add 1 to the label number, and use this new label to label the next 1 we find. After we label all the pixels in the first line, we will go on to label the next line. The main method are same as we label the first line’s pixels, but in this case, we need to consider the pixels connect to it in the previous lane. There are two standard that are often used to determine if two pixels are connected, 4-connected pixels and 8-connected pixels. And in our project, we tried both. For the 4-connected pixels connected components, we only need to consider the pixel above our searching pixel. If it has some value, we will set all the three label as the same, and also change the label equals to the bigger label to be the smaller label, and we need to minus our label number with 1. And the same for the 8-connected value.

## 3.3 Text Recoginition:
Following the segmentation part, we want to recognize each character in segmented figure.
In this part, we initially suggested that two methods might be doable:
### 3.3.1 Convolutional Neural Network:
Referred to TensorFlow MNIST model, a 5-layer-CNN model trained on EMNIST dataset has been constructed. The structure of the CNN is shown in the background part. This model has been trained in the EMNIST Letters training dataset for 5101 times and it has been tested in the same dataset to see an accuracy.
After the training process, a validation part could be followed. Setting my own dataset as an input, or using some preprocessed image as an input, then I run the model and tried to obtain some reasonable results from the outputs. 
The initial setup was that an image, after preprocessing, will be a series of images that each image, in the series, contains individual character. Then we feed these images to the text recognition model so that we will obtain text outputs.
### 3.3.2 Ram's Model
Suggested by Ram, Das and Shamdasani, a simple model which is trained in dataset with 60k training times has been tested. This is a pretrained model. Hence, weights and biases have been set and trained properly. After loading weights and changing input variables, we set outputs of the preprocess as the input of this pretrained model, that is, we used this model to do the last step, character recognition, and tried to obtain text outputs.
