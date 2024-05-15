# Video-Segmintation 
  - First we want to apply edge detection and smoothing filters to the frame We will use the Gaussian Laplacian filter for edge detection and Gaussian filter for smoothing.
  - apply adaptive thresholding to the frame to segment the objs from the background.
  - Apply morphological operations to improve the segmentation (eirosion and dilation) for edge linking operation too , we apply closing operation to fill the holes in the objects and opening to separate objs.
## if you want text only 
  - Apply tesseract OCR to detect the text in the frame
  - Draw bounding boxes around the detected text and apply last steps for segment text captions only 
  - Display the frame with the bounding boxes 
![image](https://github.com/Eng-Abdelrahman-Mostafa-Mohamed/Vido-Segmintation-/assets/116603423/d9fa19a1-a431-4522-9db5-c8c8cb157243)
