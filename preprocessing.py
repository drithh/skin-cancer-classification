import cv2
from util import plot_images
import numpy as np



# Convert the original image to grayscale
def hair_removal(original_image, debug=False):
  kernel_size = 5  # You can adjust this parameter based on the level of noise in your image
  median_filter = cv2.medianBlur(original_image, kernel_size)
  gray_image = cv2.cvtColor(median_filter, cv2.COLOR_BGR2GRAY)

  # Kernel for the morphological filtering
  kernel = cv2.getStructuringElement(1,(17,17))

  # Perform the blackHat filtering on the grayscale image to find the  hair countours
  blackhat = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel)

  # intensify the hair countours in preparation for the inpainting  algorithm
  _,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)

  # inpaint the original image depending on the mask
  dst = cv2.inpaint(median_filter,thresh2,1,cv2.INPAINT_TELEA)

  if debug:
    result = [original_image, median_filter, blackhat, thresh2, dst]
    label = ['Gambar Asli', 'Median Filter', 'Blackhat', 'Threshold', 'Hasil']
    plot_images(result, label, 5, 1)
  
  return dst



def get_mask_dark_corners(original_image, circularity_threshold=0.5, area_threshold_ratio=0.2):
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)


    # Use Canny edge detection to find edges in the image
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the background triangles
    mask = np.zeros_like(gray)

    # Iterate through the contours and draw them on the mask
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Add the mask to the binary image
    binary_image = cv2.bitwise_and(gray, mask)

    # Invert the image to make skin lesions white and background black
    inverted_image = cv2.bitwise_not(binary_image)

    # Perform flood fill operation to remove holes in the background
    h, w = inverted_image.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inverted_image, flood_mask, (0, 0), 255)
    
    # Return the mask
    return inverted_image 
    # image_gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    # _, binary_image = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # # Create a black image with the same shape as the original image
    # img_contours = np.zeros_like(original_image)

    # # Draw contours on the black image
    # cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 1)

    # # Filter contours based on circularity
    # circular_contours = []
    # max_circularity = 0

    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     perimeter = cv2.arcLength(contour, True)

    #     # Check if the perimeter is not zero before calculating circularity
    #     if perimeter != 0:
    #         circularity = (4 * np.pi * area) / (perimeter ** 2)
    #         # Check if the contour is circle-like and the area is large enough
    #         if circularity > circularity_threshold and area > area_threshold_ratio * original_image.size:
    #             circular_contours.append(contour)

    # # Create a white mask
    # mask = np.zeros_like(image_gray)

    # if circular_contours:
    #     # Draw the circular contours on the mask
    #     cv2.drawContours(mask, circular_contours, -1, 255, thickness=cv2.FILLED)

    # else:
    #     # make mask white
    #     mask[:] = 255

    # return mask

def preprocess_image(image, debug=False):
    kernel_size = 5  # You can adjust this parameter based on the level of noise in your image
    image_median_filtered_image = cv2.medianBlur(image, kernel_size)
    image_removed_hair = hair_removal(image_median_filtered_image)

    # mask_dark_corners = get_mask_dark_corners(image_removed_hair)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_removed_hair, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # apply mask
    # binary = cv2.bitwise_and(binary, mask_dark_corners)

    # Morphological opening to remove small objects
    kernel_opening = np.ones((5, 5), np.uint8)
    binary_opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_opening, iterations=1)

    # Smoothen the lesion border using opening and closing operations
    kernel_smoothening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    binary_smooth_opening = cv2.morphologyEx(binary_opening, cv2.MORPH_OPEN, kernel_smoothening, iterations=1)
    binary_smooth_closing = cv2.morphologyEx(binary_smooth_opening, cv2.MORPH_CLOSE, kernel_smoothening, iterations=1)

    # inverse
    inverted_binary_smooth = cv2.bitwise_not(binary_smooth_closing)

    result_image = cv2.bitwise_and(image_removed_hair, image_removed_hair, mask=inverted_binary_smooth)
    
    if debug:
        result = [image_removed_hair, binary, binary_opening, binary_smooth_opening, binary_smooth_closing, result_image]
        label = ['Gambar Asli', 'Otsu Treshold', 'Opening (Square)', 'Opening (Ellpise)', 'Closing (Ellpise)', 'Hasil']
        plot_images(result, label, 6, 3)
    
    return result_image
