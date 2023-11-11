import os
from matplotlib import pyplot as plt
import numpy as np
import cv2


def plot_images(images, image_names, columns=None, rows=None):
    rows = rows  or (int(np.sqrt(len(images) + 1)) )
    columns = columns or rows
    plt.figure(figsize=(columns * 4, rows * 4))
    
    for i, filtered_image in enumerate(images):
        # make check if i = 10 because columns will break
        plt.subplot(rows, columns, i + 1)
        plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
        plt.title(image_names[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    
    
def read_images(folder_path):
    images = []
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
    else:
        # Get a list of all files in the folder
        files = os.listdir(folder_path)

        # Iterate through each file in the folder
        for file_name in files:
            # Construct the full path to the image file
            file_path = os.path.join(folder_path, file_name)

            # Check if the file is a valid image file
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Read the image using OpenCV
                image = cv2.imread(file_path)
                images.append(image)
            else:
                print(f"Skipping non-image file: {file_name}")
    return images