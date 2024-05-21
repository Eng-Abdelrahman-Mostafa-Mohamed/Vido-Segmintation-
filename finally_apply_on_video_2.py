import cv2
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt 

x_start_seg_part, y_start_seg_part, x_end_seg, y_end_seg =[2,600,2000,750]

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Position (x, y):", x, y)
        return(x,y)
        
def preprocess_frame(frame):
    thresh = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY,7, 2)
    kernel_opening = np.ones((5, 5), np.uint8)
    kernel_closing = np.ones((2, 2), np.uint8)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_opening, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_closing, iterations=2)

    return closing

def detect_(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Opening_Closing seperation
    closing=preprocess_frame(gray)
    edges = closing - gray

    edges = np.where(edges >= 50, 255, 0).astype(np.uint8)

    # Set the red channel to 255 where edges are detected (making white pixels red)
    edges[:, 0] = np.where(edges == 255, 255, 0)

    # Set the green channel to 255 where edges are detected (making white pixels green)
    edges[:, :, 1] = np.where(edges == 255, 255, 0)

    # Set the blue channel to 0 (no blue component)
    edges[:, :, 2] = 0

    # #----------------------------------------------------------------------------------
    # Create an empty RGBA frame with the same dimensions
    rgba_frame = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)

    # Copy the RGB channels to the RGBA frame
    rgba_frame[:, :, :3] = frame

    # Set the alpha channel to fully opaque (255) for all pixels
    rgba_frame[:, :, 3] = 255
    #----------------------------------------------------------------------------------

    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coordinates= []
    
    # Get coordinates
    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check if the bounding box is within the specified region
        if x >= x_start_seg_part and y >= y_start_seg_part and (x + w) <= x_end_seg and (y + h) <= y_end_seg:
            coordinates.append([x, y, w, h])
            
            # Draw a rectangle around the object
            # cv2.rectangle(rgba_frame + rgba_image, (x, y), (x + w, y + h), (0, 150, 0), 4)        
            cv2.rectangle(edges, (x, y), (x + w, y + h), (0, 150, 0), 4)        

    # return rgba_frame + rgba_image, coordinates
    return rgba_image, coordinates


#-------------------------------------------
#-------------------Main--------------------
cap = cv2.VideoCapture('Project Video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    detected_image , cordinates=detect_(frame)

    # # Convert frame to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # # Opening_Closing seperation
    # closing=preprocess_frame(gray)
    # edges = closing - gray

    # edges = np.where(edges >= 50, 255, 0).astype(np.uint8)

    # # Create an RGBA image from the grayscale image
    # rgba_image = np.zeros((edges.shape[0], edges.shape[1], 4), dtype=np.uint8)

    # # Set the red channel to 255 where edges are detected (making white pixels red)
    # rgba_image[:, :, 0] = np.where(edges == 255, 255, 0)

    # # Set the green channel to 255 where edges are detected (making white pixels green)
    # rgba_image[:, :, 1] = np.where(edges == 255, 255, 0)

    # # Set the blue channel to 0 (no blue component)
    # rgba_image[:, :, 2] = 0

    # # Set the alpha channel based on the pixel intensity
    # rgba_image[:, :, 3] = np.where(edges == 0, 0, 255) 

    # img = np.ones((720, 1280, 4))*0

    # croped_part = detected_image[y_start_seg_part:y_end_seg, x_start_seg_part:x_end_seg]
    # croped_part_3d = np.expand_dims(croped_part,2)
    # segmented_caption_part_image=frame
    # segmented_caption_part_image[y_start_seg_part:y_end_seg, x_start_seg_part:x_end_seg]=croped_part_3d
    # # croped_part_rgp = cv2.cvtColor(croped_part, cv2.COLOR_GRAY2RGB)
    # cv2.rectangle(segmented_caption_part_image, (607, 674), (656, 710), (0, 150, 0), 4) 
    # cv2.imshow('final_segmented_caption_part_image',segmented_caption_part_image)
    cv2.imshow('final_segmented_caption_part_image', detected_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()