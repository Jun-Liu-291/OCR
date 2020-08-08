# OCR
Implementation of Optical Character Recognition

## Introduction
Optical character recognition or optical character reader, often known as OCR, refers to the process of analyzing and identifying image files of text data and obtaining text and layout information. The text in the image is also recognized and converted to text from image. 
 
According to the identification scenario, the OCR can be roughly divided into a dedicated OCR that identifies a specific scene and a generalized OCR that identifies multiple scenarios. For example, today's ingenious document identification and license plate recognition are typical examples of dedicated OCR. Universal OCR can be used in more complex scenarios and has greater application potential. However, since the scene of the general picture is not fixed and the layout of the text is various, it is more difficult. According to the content of the recognized picture, the scene can be divided into clear and fixed scenes and simpler scenes.
 
The typical OCR is shown as follows:
 
Input  Image Preprocessing  Text Detection  Text Recognition  Output
 
Usually, in traditional OCR technology, image preprocessing is used to correct optical imaging problems of images. Common pre-processing processes include geometric transformation (perspective, distortion, rotation, etc.), distortion correction, blur removal, image enhancement, and flat-field correction.
 
Text detection detects the location and extent of the text and its layout. It also usually includes layout analysis and text line detection. The main problems solved by text detection are where there is text and how much the text is.
 
Text recognition is based on text detection, which recognizes text content and converts text information in the image into text information. The main problem solved by text recognition is what each text is. The recognized text usually needs to be checked again to ensure its correctness.

