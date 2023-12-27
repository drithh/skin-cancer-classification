import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix
from skimage.feature import graycoprops
from scipy.stats import skew, kurtosis, entropy
from skimage.feature import hog
from skimage import exposure
from sklearn.decomposition import PCA
from tqdm import tqdm

def calculate_texture_features(image) -> pd.Series:
    # Read image
    gray_result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
  
    # Create a mask to exclude black pixels
    mask = (gray_result > 0)
    masked_image = gray_result * mask

    # Compute GLCM
    glcm = graycomatrix(masked_image, distances=distances, angles=angles, symmetric=True, normed=True)

    # Calculate texture features
    contrast = graycoprops(glcm, 'contrast')
    energy = graycoprops(glcm, 'energy')
    homogeneity = graycoprops(glcm, 'homogeneity')
    correlation = graycoprops(glcm, 'correlation')

    # Store features in a Pandas DataFrame
    features_dict = {
        'glcm_contrast': np.nanmean(contrast),
        'glcm_energy': np.nanmean(energy),
        'glcm_homogeneity': np.nanmean(homogeneity),
        'glcm_correlation': np.nanmean(correlation)
    }

    return pd.Series(features_dict)
  
def calculate_rgb_statistic(image) -> pd.Series:
    # Split the image into its channels
    b, g, r = cv2.split(image)

    # Filter out black pixels
    non_black_indices = np.where((b > 0) & (g > 0) & (r > 0))

    # Extract non-black values for each channel
    b_non_black = b[non_black_indices]
    g_non_black = g[non_black_indices]
    r_non_black = r[non_black_indices]

    # Check if the arrays are not empty before calculating statistics
    if len(b_non_black) > 0:
        mean_b, var_b, skew_b, kurt_b, entropy_b = np.nan_to_num(np.mean(b_non_black), nan=0), np.nan_to_num(np.var(b_non_black), nan=0), np.nan_to_num(skew(b_non_black.flatten()), nan=0), np.nan_to_num(kurtosis(b_non_black.flatten()), nan=0), np.nan_to_num(entropy(b_non_black.flatten()), nan=0)
    else:
        mean_b, var_b, skew_b, kurt_b, entropy_b = 0, 0, 0, 0, 0
    
    # Repeat similar checks for other channels (g and r)
    if len(g_non_black) > 0:
        mean_g, var_g, skew_g, kurt_g, entropy_g = np.nan_to_num(np.mean(g_non_black), nan=0), np.nan_to_num(np.var(g_non_black), nan=0), np.nan_to_num(skew(g_non_black.flatten()), nan=0), np.nan_to_num(kurtosis(g_non_black.flatten()), nan=0), np.nan_to_num(entropy(g_non_black.flatten()), nan=0)
    else:
        mean_g, var_g, skew_g, kurt_g, entropy_g = 0, 0, 0, 0, 0
    
    if len(r_non_black) > 0:
        mean_r, var_r, skew_r, kurt_r, entropy_r = np.nan_to_num(np.mean(r_non_black), nan=0), np.nan_to_num(np.var(r_non_black), nan=0), np.nan_to_num(skew(r_non_black.flatten()), nan=0), np.nan_to_num(kurtosis(r_non_black.flatten()), nan=0), np.nan_to_num(entropy(r_non_black.flatten()), nan=0)
    else:
        mean_r, var_r, skew_r, kurt_r, entropy_r = 0, 0, 0, 0, 0

    # Return the calculated values
    features_dict = {
        'rgb_mean_b': mean_b, 'rgb_mean_g': mean_g, 'rgb_mean_r': mean_r,
        'rgb_var_b': var_b, 'rgb_var_g': var_g, 'rgb_var_r': var_r,
        'rgb_skew_b': skew_b, 'rgb_skew_g': skew_g, 'rgb_skew_r': skew_r,
        'rgb_kurt_b': kurt_b, 'rgb_kurt_g': kurt_g, 'rgb_kurt_r': kurt_r,
        'rgb_entropy_b': entropy_b, 'rgb_entropy_g': entropy_g, 'rgb_entropy_r': entropy_r
    }
    
    return pd.Series(features_dict)
  
def calculate_hsv_statistic(image) -> pd.Series:
    # Convert the image to HSV colorspace
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the image into its channels
    h, s, v = cv2.split(hsv_image)

    # Filter out black pixels
    non_black_indices = np.where((h > 0) & (s > 0) & (v > 0))

    # Extract non-black values for each channel
    h_non_black = h[non_black_indices]
    s_non_black = s[non_black_indices]
    v_non_black = v[non_black_indices]

    # Check if the arrays are not empty before calculating statistics
    if len(h_non_black) > 0:
        mean_h, var_h, skew_h, kurt_h, entropy_h = np.nan_to_num(np.mean(h_non_black), nan=0), np.nan_to_num(np.var(h_non_black), nan=0), np.nan_to_num(skew(h_non_black.flatten()), nan=0), np.nan_to_num(kurtosis(h_non_black.flatten()), nan=0), np.nan_to_num(entropy(h_non_black.flatten()), nan=0)
    else:
        mean_h, var_h, skew_h, kurt_h, entropy_h = 0, 0, 0, 0, 0
    
    # Repeat similar checks for other channels (s and v)
    if len(s_non_black) > 0:
        mean_s, var_s, skew_s, kurt_s, entropy_s = np.nan_to_num(np.mean(s_non_black), nan=0), np.nan_to_num(np.var(s_non_black), nan=0), np.nan_to_num(skew(s_non_black.flatten()), nan=0), np.nan_to_num(kurtosis(s_non_black.flatten()), nan=0), np.nan_to_num(entropy(s_non_black.flatten()), nan=0)
    else:
        mean_s, var_s, skew_s, kurt_s, entropy_s = 0, 0, 0, 0, 0
        
    if len(v_non_black) > 0:
        mean_v, var_v, skew_v, kurt_v, entropy_v = np.nan_to_num(np.mean(v_non_black), nan=0), np.nan_to_num(np.var(v_non_black), nan=0), np.nan_to_num(skew(v_non_black.flatten()), nan=0), np.nan_to_num(kurtosis(v_non_black.flatten()), nan=0), np.nan_to_num(entropy(v_non_black.flatten()), nan=0)
    else:
        mean_v, var_v, skew_v, kurt_v, entropy_v = 0, 0, 0, 0, 0
    
    # Return the calculated values
    features_dict = {
        'hsv_mean_h': mean_h, 'hsv_mean_s': mean_s, 'hsv_mean_v': mean_v,
        'hsv_var_h': var_h, 'hsv_var_s': var_s, 'hsv_var_v': var_v,
        'hsv_skew_h': skew_h, 'hsv_skew_s': skew_s, 'hsv_skew_v': skew_v,
        'hsv_kurt_h': kurt_h, 'hsv_kurt_s': kurt_s, 'hsv_kurt_v': kurt_v,
        'hsv_entropy_h': entropy_h, 'hsv_entropy_s': entropy_s, 'hsv_entropy_v': entropy_v
    }
    
    return pd.Series(features_dict)
  
def calculate_lab_statistic(image) -> pd.Series:
    # Convert the image to LAB colorspace
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the image into its channels
    l, a, b = cv2.split(lab_image)

    # Filter out black pixels
    non_black_indices = np.where((l > 0) & (a > 0) & (b > 0))

    # Extract non-black values for each channel
    l_non_black = l[non_black_indices]
    a_non_black = a[non_black_indices]
    b_non_black = b[non_black_indices]

    # Check if the arrays are not empty before calculating statistics
    if len(l_non_black) > 0:
        mean_l, var_l, skew_l, kurt_l, entropy_l = np.nan_to_num(np.mean(l_non_black), nan=0), np.nan_to_num(np.var(l_non_black), nan=0), np.nan_to_num(skew(l_non_black.flatten()), nan=0), np.nan_to_num(kurtosis(l_non_black.flatten()), nan=0), np.nan_to_num(entropy(l_non_black.flatten()), nan=0)
    else:
        mean_l, var_l, skew_l, kurt_l, entropy_l = 0, 0, 0, 0, 0

    # Repeat similar checks for other channels (a and b)
    if len(a_non_black) > 0:
        mean_a, var_a, skew_a, kurt_a, entropy_a = np.nan_to_num(np.mean(a_non_black), nan=0), np.nan_to_num(np.var(a_non_black), nan=0), np.nan_to_num(skew(a_non_black.flatten()), nan=0), np.nan_to_num(kurtosis(a_non_black.flatten()), nan=0), np.nan_to_num(entropy(a_non_black.flatten()), nan=0)
    else:
        mean_a, var_a, skew_a, kurt_a, entropy_a = 0, 0, 0, 0, 0

    if len(b_non_black) > 0:
        mean_b, var_b, skew_b, kurt_b, entropy_b = np.nan_to_num(np.mean(b_non_black), nan=0), np.nan_to_num(np.var(b_non_black), nan=0), np.nan_to_num(skew(b_non_black.flatten()), nan=0), np.nan_to_num(kurtosis(b_non_black.flatten()), nan=0), np.nan_to_num(entropy(b_non_black.flatten()), nan=0)
    else:
        mean_b, var_b, skew_b, kurt_b, entropy_b = 0, 0, 0, 0, 0

    
    # Return the calculated values
    features_dict = {
        'lab_mean_l': mean_l, 'lab_mean_a': mean_a, 'lab_mean_b': mean_b,
        'lab_var_l': var_l, 'lab_var_a': var_a, 'lab_var_b': var_b,
        'lab_skew_l': skew_l, 'lab_skew_a': skew_a, 'lab_skew_b': skew_b,
        'lab_kurt_l': kurt_l, 'lab_kurt_a': kurt_a, 'lab_kurt_b': kurt_b,
        'lab_entropy_l': entropy_l, 'lab_entropy_a': entropy_a, 'lab_entropy_b': entropy_b
    }
    
    return pd.Series(features_dict)

def calculate_shape_features(image) -> pd.Series:
    # Read the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate area (number of non-zero pixels)
    area = np.count_nonzero(image)

    # Calculate mean, variance, and standard deviation
    mean, std_dev = cv2.meanStdDev(image)

    # Extract values from numpy arrays
    mean = mean.flatten()
    std_dev = std_dev.flatten()

    # Calculate variance from std_dev
    variance = std_dev ** 2

    features_dict = {
        'shape_area': area,
        'shape_mean': mean[0],
        'shape_std_dev': std_dev[0],
        'shape_variance': variance[0]
    }
    
    return pd.Series(features_dict)
    
          

def calculate_hog_with_pca(image_paths) -> pd.Series:
  image_features = []
  
  # resize and grayscale images
  for image_path in tqdm(image_paths, desc='Extracting HOG Features', unit='image'):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (112, 122))
    
    # Compute HOG features
    features, _ = hog(resized_image, orientations=8, pixels_per_cell=(8, 8),
                            cells_per_block=(1, 1), visualize=True)

    # Optionally, you can enhance the visualization of HOG features
    image_features.append(features)
  
  pca = PCA(n_components=100)
  pca_result = pca.fit_transform(image_features)
  # make it to dataframe
  pca_result = pd.DataFrame(pca_result, columns=['hog_pca_%i' % i for i in range(100)])
  return pca_result
  
def get_file_path_with_label(preprocessed_path):
  train_features = pd.DataFrame()
  test_features = pd.DataFrame()

  def hotfix_skin_cancer_type_folder(skin_cancer_type_folder):
    if (skin_cancer_type_folder == "benign"):
      return 0
    elif (skin_cancer_type_folder == "malignant"):
      return 1

  for data_type_folder in os.listdir(preprocessed_path):
      for skin_cancer_type_folder in os.listdir(os.path.join(preprocessed_path, data_type_folder)):
        for image in os.listdir(os.path.join(preprocessed_path, data_type_folder, skin_cancer_type_folder)):
          image_path = os.path.join(preprocessed_path, data_type_folder, skin_cancer_type_folder, image)
          data = {"image_path": image_path, "label": hotfix_skin_cancer_type_folder(skin_cancer_type_folder)}
          if (data_type_folder == "train"):
            train_features = pd.concat([train_features, pd.DataFrame([data])], ignore_index=True)
          else:
            test_features = pd.concat([test_features, pd.DataFrame([data])], ignore_index=True)
            
  return train_features, test_features