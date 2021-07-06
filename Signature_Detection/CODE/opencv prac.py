#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
print(cv2.__version__)



# In[2]:





# In[118]:


import cv2
import numpy as np
import glob

#a path - "Downloads/a_signs/sign5- 24043.png" - all signatures by the same person
#b_path - "Downloads/b_signs/b_sign4 - 81820.png"  - here a few images are similar, a few are not
#c path - "Downloads/c_signs/c_sign5 - 82135.png"  - no sign is similar
print("Images of signatures should be in Standard Format - 650px * 150px with white background. Kindly ensure that your signature is in the centre of the image\n\n")
original = cv2.imread("Downloads/a_signs/sign5- 24043.png")

# Sift and Flann
sift = cv2.xfeatures2d.SIFT_create()

kp_1, desc_1 = sift.detectAndCompute(original, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Load all the images
sum = 0
avg = 0
all_images_to_compare = []
titles = []
similarity = []

for f in glob.iglob("Downloads/a_signs/*"):
    image = cv2.imread(f)
    titles.append(f)
    all_images_to_compare.append(image)
    
for image_to_compare, title in zip(all_images_to_compare, titles):
    #1) Check if 2 images are equals
    if original.shape == image_to_compare.shape:
        difference = cv2.subtract(original, image_to_compare)
        b, g, r = cv2.split(difference)
        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            print("Similarity: 100% (equal size and channels). Accurate Image is Found, as we've compared the image to itself.\n")
            continue
            
    # 2) Continue to check for similarities between the 2 images
    
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
    matches = flann.knnMatch(desc_1, desc_2, k=2)
    good_points = []
    for m, n in matches:
        if m.distance > 0.6*n.distance:
            good_points.append(m)
    number_keypoints = 0
    if len(kp_1) >= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)
    print("Title: " + title)
    percentage_similarity = len(good_points) / number_keypoints * 100
    print("Similarity: " + str(int(percentage_similarity)) + "\n")
    similarity.append(percentage_similarity)
    
for i in similarity:
    sum += i
avg = sum/len(similarity)
print("\nAverage signature accuracy is: ",avg,"%\n")

#print("Number of Images that don't match 100%: ",len(similarity),"\n")

if (avg>90):
    print("Signatures are authentic!")
else:
    print("Signatures do not seem authentic. Sorry, try again.")
    


# In[ ]:





# In[ ]:




