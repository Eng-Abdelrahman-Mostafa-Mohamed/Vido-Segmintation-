import cv2
import numpy as np
import scipy.ndimage as nd
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

def detect_subtitles(processed_frame):
    d = pytesseract.image_to_data(processed_frame, output_type=pytesseract.Output.DICT)
    subtitle_bboxes = []
    word_bboxes = []
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        text = d['text'][i].strip()
        if text:
            word_bboxes.append((x, y, w, h))
            if len(subtitle_bboxes) == 0:
                subtitle_bboxes.append((x, y, w, h))
            else:
                last_bbox = subtitle_bboxes[-1]
                if abs(y - last_bbox[1]) < h:  # Assume same line if y-difference is less than height
                    subtitle_bboxes[-1] = (min(x, last_bbox[0]), min(y, last_bbox[1]), max(w, last_bbox[2]), max(h, last_bbox[3]))
                else:
                    subtitle_bboxes.append((x, y, w, h))
    
    return subtitle_bboxes, word_bboxes

def draw_bounding_boxes(frame, subtitle_bboxes, word_bboxes):
    for (x, y, w, h) in subtitle_bboxes:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for (x, y, w, h) in word_bboxes:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return frame


def crop_caption_part(image):
    subtitle_bboxes,word_boxes = detect_subtitles(image)
    if len(subtitle_bboxes) == 0:
        return image
    else:
        x,y,w,h = subtitle_bboxes[-1]
        return image[y:y+h, x:x+w]



cap = cv2.VideoCapture('Project Video.mp4')
captions=[]
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    # what is adaptive thresholding : Adaptive thresholding is the method where the threshold value is calculated for smaller regions and therefore, there will be different threshold values for different regions.
    #what is there arguments : cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C) what C represent and block size represent 
    filtered_image = nd.gaussian_laplace(gray, sigma=3)
    thresh = cv2.adaptiveThreshold(filtered_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 7, 2)
    # cv2.imshow('thresh',thresh)
    
    # # Improve segmentation with morphology (optional)
    kernel_opening = np.ones((2, 2), np.uint8)
    kernel_closing = np.ones((3, 3), np.uint8)
    
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_opening, iterations=7)# what this function do  cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)? 
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_closing, iterations=2)
    # # Display results
    # # cv2.imshow('Original Video', frame)
    # # i want to segment the text from the background so i will show the text only
    
    # cv2.imshow('Thresholded', np.invert(closing))
    
    
    captions.append(crop_caption_part(thresh))
    # cv2.imshow('captions',crop_caption_part(thresh))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# import cv2
# cv2.imshow(captions[0],cmap='gray')