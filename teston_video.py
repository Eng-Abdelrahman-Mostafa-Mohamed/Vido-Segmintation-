# import cv2
# import numpy as np
# import scipy.ndimage as nd

# cap = cv2.VideoCapture('Project Video.mp4')

# for k in range(10):
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to grayscale
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame_gray = frame_gray / 255.0  # Normalize to [0, 1]
    
#     cv2.imshow('Original video', frame)
    
#     # Apply LoG filter
#     LoG = nd.gaussian_laplace(frame_gray, sigma=1)
#     zero_crossings = np.zeros_like(LoG)

#     # Detect zero crossings
#     for y in range(1, LoG.shape[0] - 1):
#         for x in range(1, LoG.shape[1] - 1):
#             if (
#                 (LoG[y-1, x] * LoG[y+1, x] < 0) or
#                 (LoG[y, x-1] * LoG[y, x+1] < 0) or
#                 (LoG[y-1, x-1] * LoG[y+1, x+1] < 0) or
#                 (LoG[y-1, x+1] * LoG[y+1, x-1] < 0)
#             ):
#                 zero_crossings[y, x] = 1

#     # Perform edge linking using morphological dilation
#     kernel = np.ones((3, 3), np.uint8)
#     dilated_edges = cv2.dilate(zero_crossings.astype(np.uint8), kernel, iterations=5)

#     # Perform segmentation by thresholding dilated edges
#     threshold = 1
#     segmented_image = (dilated_edges >= threshold).astype(np.uint8) * 255

#     # Resize segmented image to match frame_gray dimensions
#     segmented_image_resized = cv2.resize(segmented_image, (frame_gray.shape[1], frame_gray.shape[0]))

#     # Multiply frame_gray and segmented_image_resized element-wise
#     image_new = np.multiply(frame_gray, segmented_image_resized).astype(np.uint8)

#     # Display new image
#     cv2.imshow('New Image', image_new)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture object and close windows
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import numpy as np

# cap = cv2.VideoCapture('Project Video.mp4')

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Apply adaptive thresholding
#     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                  cv2.THRESH_BINARY, 11, 2)

#     # Improve segmentation with morphology (optional)
#     kernel = np.ones((3, 3), np.uint8)
#     opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
#     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

#     # Display results
#     # cv2.imshow('Original Video', frame)
#     cv2.imshow('Thresholded Text', closing)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



import cv2
import numpy as np

cap = cv2.VideoCapture('Project Video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter for noise reduction
    gray = cv2.GaussianBlur(gray,sigmaX=0.5,sigmaY=0.5, ksize=(5,5))
    gray = cv2.sopel(gray, cv2.CV_64F)
    # Adaptive thresholding with Niblack's method (optional)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                               cv2.THRESH_BINARY, 11, 2)

    k = 0.5  # Tuning parameter
    mean = cv2.blur(gray, (11, 11))
    thresh = cv2.compare(gray,mean, cv2.THRESH_BINARY)

    # Improve segmentation with morphology (optional)
    kernel = np.ones((3, 3), np.uint8)
    
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Connected Component Analysis (Optional)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(closing)
    for i in range(1, stats.shape[0]):
        x, y, w, h, area = stats[i]
        # Filter out small components (not likely text)
        if area > 200:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display results
    cv2.imshow('Original Video', frame)
    cv2.imshow('Thresholded Text', closing)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
