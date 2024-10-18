# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 15:28:47 2024

@author: win
"""

import cv2
import matplotlib.pyplot as plt
import os
import zipfile


from detection_utilites import detect_circular_contours, color_mask, mask_processing




def main(add_labels=False, debugging = True):
    
    output_dir = r"D:\labeled images"
    f_path = r"D:\Task8.1_Balls_dataset"
    
    # Variables for calculating accuracy
    TP, FP, FN = 0, 0, 0

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare the zip file
    zip_file = os.path.join(output_dir, "labeled_images.zip")
    with zipfile.ZipFile(zip_file, 'w') as zf:
        # List of image paths
        path_lst = os.listdir(f_path)
        
    

        for i in range(len(path_lst)):
            # Read the image
            img = cv2.imread(f_path+"\\"+path_lst[i])

            
            # Expected labels based on image name (e.g., "b_000", "br_004")
            expected_labels = []
            if 'b' in path_lst[i]:
                expected_labels.append('blue_ball')
            if 'r' in path_lst[i]:
                expected_labels.append('red_ball')
            
            # Get the mask for the red and blue ball
            masks = color_mask(img)
            
            radii = []
            circular_objects = []
            detected_labels = []
            
            for ind in range(len(masks)):
                
                # Preform the the mask processing to ensure no noise 
                mask = mask_processing(masks[ind][0])
            
                # Get the best circular contour for each color of ball if exist
                circular_object, ind_max = detect_circular_contours(mask, 
                                                                     circularity_threshold=0.3, 
                                                                     area_threshold=0.6, 
                                                                     debuge=False)
                # Check if circular_object exist
                if ind_max != None:
                    # Get the object coordinates for ploting
                    contour, center, radius = circular_object[ind_max]
                    
                    # Append the redius to the radii list
                    radii.append(radius)
                    circular_objects.append((contour, center, radius, masks[ind][1]))
            
            for contour, center, radius, label in circular_objects:
    
                # Select boundry about relative circle size
                if radius >= max(radii)/2:
                    # Append the label for later evalution
                    detected_labels.append(label)
                    
                    if label == 'red_ball':
                        color = (0, 0, 255) 
                    elif label == 'blue_ball':
                        color = (255, 0, 0) 
                    
                    # Draw the boundry
                    cv2.circle(img, center, int(radius), color, 5)
                    
                    if add_labels:
                        # Add the text label to the center of the ball
                        cv2.putText(img, label, (center[0] - 20, center[1] - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if debugging:
                # Plot the final compined image for debbuging         
                plt.figure(figsize=(10, 6))
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title("Combined Detection")
                plt.axis("off")
                plt.show()
        
            # Evalution
            for label in detected_labels:
                if label in expected_labels:
                    TP += 1  
                else:
                    FP += 1  
    
            for label in expected_labels:
                if label not in detected_labels:
                    FN += 1        
            # Save labeled image
            output_img_path = output_dir +"\\"+ path_lst[i]
            cv2.imwrite(output_img_path, img)
            zf.write(output_img_path, os.path.basename(output_img_path))

    # Calc the accuracy
    accuracy = TP / (TP + FP + FN) 
    # Create evaluation file
    eval_file = output_dir +"\\"+"evaluation.txt"
    with open(eval_file, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"TP: {TP}, FP: {FP}, FN: {FN}\n")

    # Add evaluation file to the zip
    with zipfile.ZipFile(zip_file, 'a') as zf:
        zf.write(eval_file, "evaluation.txt")        
        
if __name__ == '__main__':
    main()



cv2.waitKey(0)
cv2.destroyAllWindows()