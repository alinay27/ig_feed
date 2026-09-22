import cv2 #For image reading and processing.
import numpy as np #For image array operations (used internally).
import matplotlib.pyplot as plt #For displaying the image visually.
import os #For potential path operations (though not directly used here).
from sklearn.cluster import KMeans #For clustering pixel colors to find the dominant color.

image_path = 'orange_kitten.png'

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Converts the image from BGR to RGB color space.

clt = KMeans(n_clusters = 1)
clt.fit(img_rgb.reshape(-1, 3)) #Reshapes the image to a 2D array of pixels for clustering.
clt.labels_
clt.cluster_centers_

avg_color = clt.cluster_centers_[0].astype(int)
print(avg_color)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))#1. Create a figure with 1 row and 2 columns of subplots

axes[0].imshow(img_rgb)
axes[0].set_title("Original Image")
axes[0].axis("off")
axes[1].imshow([[avg_color / 255]])
axes[1].set_title("Average Color: " + str(avg_color))
axes[1].axis("off")

plt.show()
