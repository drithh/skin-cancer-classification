import cv2
from util import plot_images
import numpy as np



# Convert the original image to grayscale
def clean_image(original_image, debug=False):
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

def preprocess_image(image, debug=False):
    image_removed_hair = clean_image(image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image_removed_hair, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


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
