import cv2 #For image reading and processing.
import numpy as np #For image array operations (used internally).
import matplotlib.pyplot as plt #For displaying the image visually.
import os #For potential path operations (though not directly used here).

image_path = 'orange_kitten.png'

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Converts BGR to RGB for processing.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Converts BGR to Grayscale.

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image (RGB)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis("off")

plt.tight_layout()
plt.show()