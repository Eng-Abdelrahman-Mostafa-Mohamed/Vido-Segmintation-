import cv2
import numpy as np
import pytesseract
import scipy.ndimage as nd

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

def preprocess_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 30, 150)

    return edges

def detect_text_regions(edges):
    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on aspect ratio to identify potential text regions
    min_aspect_ratio = 1.5
    max_aspect_ratio = 5.0
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h != 0 else 0
        if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
            text_regions.append((x, y, w, h))

    return text_regions

def draw_bounding_boxes(frame, text_regions):
    for (x, y, w, h) in text_regions:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

def extract_captions(frame, text_regions):
    captions = []
    for (x, y, w, h) in text_regions:
        caption = frame[y:y+h, x:x+w]
        captions.append(caption)

    return captions

def combine_captions(captions):
    # Determine maximum height and total width for combined caption frame
    max_height = max(caption.shape[0] for caption in captions)
    total_width = sum(caption.shape[1] for caption in captions)

    # Create an empty frame to combine all captions
    combined_frame = np.zeros((max_height, total_width, 3), dtype=np.uint8)

    # Paste each caption onto the combined frame
    x_offset = 0
    for caption in captions:
        combined_frame[0:caption.shape[0], x_offset:x_offset+caption.shape[1]] = caption
        x_offset += caption.shape[1]

    return combined_frame

# Load video
cap = cv2.VideoCapture('Project Video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    edges = preprocess_frame(frame)

    # Detect text regions
    text_regions = detect_text_regions(edges)

    # Draw bounding boxes on original frame
    frame_with_boxes = draw_bounding_boxes(frame.copy(), text_regions)

    # Extract captions from original frame
    captions = extract_captions(frame, text_regions)

    # Combine all captions into one frame
    combined_captions = combine_captions(captions)

    # Display results
    cv2.imshow('Original Frame with Boxes', frame_with_boxes)
    cv2.imshow('Combined Captions', combined_captions)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
