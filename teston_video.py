import cv2
import numpy as np
import scipy.ndimage as nd
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'


# First we want to apply edge detection and smoothing filters to the frame We will use the Gaussian Laplacian filter for edge detection and Gaussian filter for smoothing
# apply adaptive thresholding to the frame to segment the objs from the background
# Apply morphological operations to improve the segmentation (eirosion and dilation) for edge linking operation too , we apply closing operation to fill the holes in the objects and opening to separate objs
# Apply tesseract OCR to detect the text in the frame
# Draw bounding boxes around the detected text and apply last steps for segment text captions only 
# Display the frame with the bounding boxes



#### the big problem is that the text is not clear and the bounding boxes are not accurate so we need to improve the segmentation and the OCR accuracy it's extract objs as text captions
# 

 # if we want to detect the text in the frame we need to apply the following steps
# def detect_subtitles(processed_frame):
#     d = pytesseract.image_to_data(processed_frame, output_type=pytesseract.Output.DICT)
#     subtitle_bboxes = []
#     word_bboxes = []
#     n_boxes = len(d['level'])
#     for i in range(n_boxes):
#         (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#         text = d['text'][i].strip() # for deviding sent into words 
#         if text:
#             word_bboxes.append((x, y, w, h))
#             if len(subtitle_bboxes) == 0:
#                 subtitle_bboxes.append((x, y, w, h))
#             else:
#                 last_bbox = subtitle_bboxes[-1]
#                 if abs(y - last_bbox[1]) < h:  # Assume same line if y-difference is less than height
#                     subtitle_bboxes[-1] = (min(x, last_bbox[0]), min(y, last_bbox[1]), max(w, last_bbox[2]), max(h, last_bbox[3]))
#                 else:
#                     subtitle_bboxes.append((x, y, w, h))
    
#     return subtitle_bboxes, word_bboxes

# def draw_bounding_boxes(frame, subtitle_bboxes, word_bboxes):
#     for (x, y, w, h) in subtitle_bboxes:
#         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#     for (x, y, w, h) in word_bboxes:
#         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
#     return frame


# def crop_caption_part(image):
#     subtitle_bboxes,word_boxes = detect_subtitles(image)
#     if len(subtitle_bboxes) == 0:
#         return image
#     #retrutns subtitle part of the image
#     else:
#         x,y,w,h = subtitle_bboxes[-1]
#         return image[y:y+h, x:x+w]
    
# def try_detect_and_crop_all_caption_parts(image):
#     subtitle_bboxes, word_boxes = detect_subtitles(image)
#     if len(subtitle_bboxes) == 0:
#         return image
#     else:
#         x1, y1, _, _ = subtitle_bboxes[0]
#         x2, y2, _, _ = subtitle_bboxes[-1]
#         return image[y1:y2, x1:x2]



cap = cv2.VideoCapture('Project Video.mp4')
captions=[]
all_captions=[]
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding is the method where the threshold value is calculated for smaller regions and therefore, there will be different threshold values for different regions.
    filtered_image = nd.gaussian_laplace(gray, sigma=3)
    thresh = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 7, 2)
    # cv2.imshow('thresh',thresh)
    
    kernel_opening = np.ones((2, 2), np.uint8)
    kernel_closing = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_opening, iterations=7)
    cv2.imshow('after opening',opening)
    
    #after applying opening by applying erosion then dilation  we find the hall in segmented obj in segmented regon so we abbly closing to fill the holes by applying dilation then erosion
    
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_closing, iterations=2)
    
    # # Display results
    # # cv2.imshow('Original Video', frame)
    
    cv2.imshow('Threshholded', np.invert(closing)) 
    cv2.imshow('boundry segminted', np.invert(closing)*np.invert(gray)) # we multibly for segmint obj by its boundry to show the obj only not segment all regon of obj 
    
    
    # captions.append(crop_caption_part(thresh))
    
    # all_captions.append(try_detect_and_crop_all_caption_parts(closing))
    # if(len(all_captions)==100):
    #     break
    # cv2.imshow('captions',crop_caption_part(thresh))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# cv2.imshow('captions static ',captions[0])
cv2.imshow('captions all ',all_captions[99])
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()

# import cv2
# cv2.imshow(captions[0],cmap='gray')


#finnaly as my try i found  that the text is not clear and the bounding boxes are not accurate so we need to improve the segmentation and the OCR accuracy it's extract objs as text captions