# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 16:25:34 2024

@author: win
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lower red mask (0-10)

lower_blue = np.array([110, 180, 50]) 
upper_blue = np.array([140, 255, 255])

lower_red1 = np.array([0, 200, 100])
upper_red1 = np.array([5, 255, 255])
lower_red2 = np.array([160, 200, 100])
upper_red2 = np.array([180, 255, 255])


def rotate_resize_img(img):
    """
    Rotate and resize the image.
    
    Args:
        img: BGR image.

    Returns:
        the rotated and resized image.
    """
    width, height = img.shape[:2]
    
    # Rotate the image if width is greater than height
    if width > height:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        print("rotation")
    # Resize the image to 443x960
    resized_img = cv2.resize(img, (443, 960))  # Note that (width, height) is (960, 443)

    return resized_img
    
    
    
def color_mask(img):
    """
    args:
        img: images in cv2 matrix format
        
    return:
        image after applaying red color masking
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)    
    
    return ((mask_red, "red_ball"), (mask_blue, "blue_ball"))
    
    
    
    
def mask_processing(mask):
    """
    args:
        mask(numpy binary matrix): the mask before processing
    return:
        (numpy binary matrix): the mask after processing
    """
    
    # Apply Gaussian blur to reduce noise
    mask = cv2.GaussianBlur(mask, (9, 9), 2)
    
    # Apply morphological operations to close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Returning binary mask
    #_, mask = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)
    
    return mask

    
def get_contours(mask, color="", debuge=False):
    """
    get the contours from mask
    function for debugging
    """
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
    
    if debuge:
        contours_image = np.zeros_like(mask)
        cv2.drawContours(contours_image, contours, -1, 255, 1)
        plt.figure(figsize=(10, 6))
        plt.imshow(contours_image, cmap='gray')
        plt.title(f"Contours - {color}")
        plt.axis("off")
        plt.show()
    
def detect_circular_contours(mask, circularity_threshold=0.3, area_threshold=0.5, debuge=False):
    """
    Detect circular contours using circularity.
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circular_objects = []
    r_list = []
    
    for contour in contours:
        # Calculate area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 150:  # Avoid division by zero
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            # Check if the contour is circular based on the circularity threshold
            if circularity > circularity_threshold:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                area_r = area/((np.pi)*(radius**2))
                r_list.append(area_r)

                circular_objects.append((contour, center, radius))
     
    if r_list:
        # get the best contour closed to a circle
        ind_max = np.argmax(r_list)
          
        if debuge:
            # Create an image to visualize detected circular contours
            contours_image = np.zeros_like(mask)
            # Draw full contours and corresponding enclosing circles for the best r score
            contour, center, radius = circular_objects[ind_max]
            cv2.drawContours(contours_image, [contour], -1, 255, 1)
            cv2.circle(contours_image, center, int(radius), 255, 2)
        
            # Display the contour image for debugging
            plt.figure(figsize=(10, 6))
            plt.imshow(contours_image, cmap='gray')
            plt.title("Detected Circular Contours Based on Circularity")
            plt.axis("off")
            plt.show()
    else:
        ind_max = None
    return circular_objects, ind_max

