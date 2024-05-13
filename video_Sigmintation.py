# import cv2
# import numpy as np
# import scipy.ndimage as nd

# cap = cv2.VideoCapture('Project Video.mp4')

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to grayscale
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame_gray = frame_gray / 255.0  # Normalize to [0, 1]
    
#     # Apply LoG filter
#     LoG = nd.gaussian_laplace(frame_gray, sigma=1)
#     zero_crossings=np.zeros_like(LoG)
#     for y in range(1, LoG.shape[0] - 1):
#         for x in range(1, LoG.shape[1] - 1):
#             if (
#                 (LoG[y-1, x] * LoG[y+1, x] < 0) or
#                 (LoG[y, x-1] * LoG[y, x+1] < 0) or
#                 (LoG[y-1, x-1] * LoG[y+1, x+1] < 0) or
#                 (LoG[y-1, x+1] * LoG[y+1, x-1] < 0)
#             ):
#                 zero_crossings[y, x] = 1


    
#     threshold = 0.5 
#     segmented_image = (zero_crossings >= threshold).astype(np.uint8) * 255

#     # Display segmented image
#     cv2.imshow('Segmented Image', segmented_image)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




# # # i

# # # # Read the image
# # # image = cv2.imread('image.jpg')
# # # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # # # Thresholding
# # # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# # # # Invert the binary image
# # # binary = 255 - binary

# # # # Perform OCR on the binary image
# # # text = pytesseract.image_to_string(binary)

# # # # Extract text regions
# # # text_boxes = pytesseract.image_to_boxes(binary)

# # # # Draw text boxes on original image for visualization
# # # for box in text_boxes.splitlines():
# # #     box = box.split()
# # #     x, y, w, h = int(box[1]), int(box[2]), int(box[3]), int(box[4])
# # #     cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

# # # # Show the original image with text boxes
# # # cv2.imshow('Text Segmentation', image)
# # # cv2.waitKey(0)
# # # cv2.destroyAllWindows()






# # # import cv2
# # # import numpy as np
# # # import scipy.ndimage as nd
# # # import pytesseract

# # # cap = cv2.VideoCapture('Project Video.mp4')

# # # while True:
# # #     ret, frame = cap.read()
# # #     if not ret:
# # #         break

# # #     # Convert frame to grayscale
# # #     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # #     frame_gray = frame_gray / 255.0  # Normalize to [0, 1]

# # #     text_part_per_fram = 
# # #     # Apply LoG filter
# # #     LoG = nd.gaussian_laplace(frame_gray, sigma=1)
# # #     for y in range(1, LoG.shape[0] - 1):
# # #         for x in range(1, LoG.shape[1] - 1):
# # #             if (
# # #                 (LoG[y-1, x] * LoG[y+1, x] < 0) or
# # #                 (LoG[y, x-1] * LoG[y, x+1] < 0) or
# # #                 (LoG[y-1, x-1] * LoG[y+1, x+1] < 0) or
# # #                 (LoG[y-1, x+1] * LoG[y+1, x-1] < 0)
# # #             ):
# # #                 zero_crossings[y, x] = 1
# # #             else:
# # #                 zero_crossings[y, x] = 0

# # #     # Thresholding
# # #     threshold = 0.5  # Adjust as needed
# # #     segmented_image = (zero_crossings >= threshold).astype(np.uint8) * 255

# # #     # Display segmented image
# # #     cv2.imshow('Segmented Image', segmented_image)

# # #     if cv2.waitKey(1) & 0xFF == ord('q'):
# # #         break

# # # cap.release()
# # # cv2.destroyAllWindows()






# import cv2
# import pytesseract

# # Read the image
# image = cv2.imread('image.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Thresholding
# _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# # Invert the binary image
# binary = 255 - binary

# # Perform OCR on the binary image
# text_boxes = pytesseract.image_to_boxes(binary)

# # Extract and print the coordinates of each text region
# text_regions = []
# for box in text_boxes.splitlines():
#     box = box.split()
#     region = {
#         'x1': int(box[1]),
#         'y1': int(box[2]),
#         'x2': int(box[3]),
#         'y2': int(box[4])
#     }
#     text_regions.append(region)

# # Print the coordinates of each text region
# for i, region in enumerate(text_regions):
#     print(f"Text Region {i + 1}:")
#     print(f"Coordinates: ({region['x1']}, {region['y1']}) - ({region['x2']}, {region['y2']})")
#     print()

# # Show the original image with text boxes
# cv2.imshow('Text Segmentation', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import scipy.ndimage as nd
# import cv2
# # import pytesseract

# # Read the image
# frame = cv2.imread('image.png')
# frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # if not ret:
# #     break

# # Convert frame to grayscale
# frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# frame_gray = frame_gray / 255.0  # Normalize to [0, 1]

# # Apply LoG filter
# LoG = nd.gaussian_laplace(frame_gray, sigma=1)
# zero_crossings=np.zeros_like(LoG)
# for y in range(1, LoG.shape[0] - 1):
#     for x in range(1, LoG.shape[1] - 1):
#         if (
#             (LoG[y-1, x] * LoG[y+1, x] < 0) or
#             (LoG[y, x-1] * LoG[y, x+1] < 0) or
#             (LoG[y-1, x-1] * LoG[y+1, x+1] < 0) or
#             (LoG[y-1, x+1] * LoG[y+1, x-1] < 0)
#         ):
#             zero_crossings[y, x] = 1
# threshold = 1
# segmented_image = (zero_crossings >= threshold).astype(np.uint8) * 255

# # Display segmented image
# cv2.imshow('Segmented Image', segmented_image)
# cv2.imshow(' Image', frame)

# cv2.waitKey(0)



# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.ndimage as nd
# # Load an example image (replace with video frames)
# image = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)



# LoG = cv2.Laplacian(image, cv2.CV_16S)
# minLoG = cv2.morphologyEx(LoG, cv2.MORPH_ERODE, np.ones((3,3)))
# maxLoG = cv2.morphologyEx(LoG, cv2.MORPH_DILATE, np.ones((3,3)))
# zeroCross = np.logical_or(np.logical_and(minLoG < 0,  LoG > 0), np.logical_and(maxLoG > 0, LoG < 0))

# # Display the results
# plt.imshow(zeroCross, cmap="gray")
# plt.title("Captions Segmentation")
# plt.axis("off")
# plt.show()



import cv2
import numpy as np
import scipy.ndimage as nd
cap = cv2.VideoCapture('Project Video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
# Read the image
# frame = cv2.imread('image.png')
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Convert frame to grayscale
frame_gray = frame_gray / 255.0  # Normalize to [0, 1]

# Apply LoG filter
LoG = nd.gaussian_laplace(frame_gray, sigma=1)
zero_crossings = np.zeros_like(LoG)

# Detect zero crossings
for y in range(1, LoG.shape[0] - 1):
    for x in range(1, LoG.shape[1] - 1):
        if (
            (LoG[y-1, x] * LoG[y+1, x] < 0) or
            (LoG[y, x-1] * LoG[y, x+1] < 0) or
            (LoG[y-1, x-1] * LoG[y+1, x+1] < 0) or
            (LoG[y-1, x+1] * LoG[y+1, x-1] < 0)
        ):
            zero_crossings[y, x] = 1

# Perform edge linking using morphological dilation
kernel = np.ones((3, 3), np.uint8)
dilated_edges = cv2.dilate(zero_crossings.astype(np.uint8), kernel, iterations=5)

# Perform segmentation by thresholding dilated edges
threshold = 0.5
segmented_image = (dilated_edges >= threshold).astype(np.uint8) * 80

# Display segmented image
cv2.imshow('Segmented Image', segmented_image)
# Resize segmented image to match frame_gray dimensions
segmented_image_resized = cv2.resize(segmented_image, (frame_gray.shape[1], frame_gray.shape[0]))

# Multiply frame_gray and segmented_image_resized element-wise
image_new = np.multiply(frame_gray, segmented_image_resized).astype(np.uint8)

# Display new image
cv2.imshow('New Image', image_new)
cv2.imshow('Original Image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()



# import cv2
# import numpy as np
# import scipy.ndimage as nd
# import cv2
# # import pytesseract

# # Read the image
# frame = cv2.imread('image.png')
# frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # if not ret:
# #     break

# # Convert frame to grayscale
# frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# frame_gray = frame_gray / 255.0  # Normalize to [0, 1]

# # Apply LoG filter
# blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# # Apply Laplacian operator
# laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
# zero_crossings=np.where(np.diff(np.sign(LoG)))[0]
# # threshold = 1
# # segmented_image = (zero_crossings >= threshold).astype(np.uint8) * 255

# # Display segmented image
# cv2.imshow('zero crossing Image', zero_crossings)
# cv2.imshow(' Image', frame)

# cv2.waitKey(0)