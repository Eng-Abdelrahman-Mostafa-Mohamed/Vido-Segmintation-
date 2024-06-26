import cv2
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt 

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Position (x, y):", x, y)
        return(x,y)
        

    
def detect_(cleaned_image):
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coordinates= []
    
    # Get coordinates
    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        coordinates.append(list([x, y, w, h]))
        # Draw a rectangle around the object
        cv2.rectangle(cleaned_image, (x, y), (x+w, y+h), (0, 0, 0), 4)        
    return cleaned_image ,coordinates



img=cv2.imread('image.png',0)
filtered_image = img
thresh = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY,7, 2)
kernel_opening = np.ones((5, 5), np.uint8)
kernel_closing = np.ones((2,2), np.uint8)

opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_opening, iterations=1)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_closing, iterations=2)



print(closing)
detected_image , cordinates=detect_((closing*10))
cv2.imshow('detection_image',detected_image )
# x_start_seg_part, y_start_seg_part, x_end_seg, y_end_seg = int(cordinates[0][0]), int(cordinates[0][1]), int(cordinates[-1][0]), int(cordinates[-1][1])
x_start_seg_part, y_start_seg_part, x_end_seg, y_end_seg =[0,500,900,600]
croped_part = detected_image[y_start_seg_part:y_end_seg, x_start_seg_part:x_end_seg]
cv2.imshow('cropped part', croped_part)
segmented_caption_part_image=img
segmented_caption_part_image[y_start_seg_part:y_end_seg, x_start_seg_part:x_end_seg]=croped_part
cv2.imshow('final_segmented_caption_part_image',segmented_caption_part_image)
cv2.waitKey(0)