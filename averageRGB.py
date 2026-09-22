import cv2 #For image reading and processing.
import numpy as np #For image array operations (used internally).
import matplotlib.pyplot as plt #For displaying the image visually.
from sklearn.cluster import KMeans #For clustering pixel colors to find the dominant color.
import streamlit as st

st.set_page_config(page_title = "Image Color Analyzer", layout = "centered")
st.title("Image Color Analyzer")
st.write("This app calculates the average color of an uploaded image. Upload an image in PNG, JPG, or JPEG!")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    files_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(files_bytes, cv2.IMREAD_COLOR) 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    pixels = img_rgb.reshape(-1, 3)
    clt = KMeans(n_clusters = 1)
    clt.fit(pixels)
    avg_color = clt.cluster_centers_[0].astype(int)

    st.subheader("Original Image")
    st.image(img_rgb, use_container_width=True)
    st.subheader("Average Color")

    color_block = np.zeros((200, 400, 3), dtype=np.uint8)
    color_block[:] = avg_color
    st.image(color_block, use_container_width=True)
    st.write(f"RGB: {avg_color[0]}, {avg_color[1]}, {avg_color[2]}")
    hex_color = "#{:02x}{:02x}{:02x}".format(avg_color[0], avg_color[1], avg_color[2])
    st.write(f"Hex: {hex_color}")
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))#Create a figure with 1 row and 2 columns of subplots
# axes[0].imshow(img_rgb)
# axes[0].set_title("Original Image")
# axes[0].axis("off")
# axes[1].imshow([avg_color / 255])
# axes[1].set_title("Average Color: " + str(avg_color))
# axes[1].axis("off")

# plt.show()
