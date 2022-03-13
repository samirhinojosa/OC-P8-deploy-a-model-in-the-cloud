## General
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Computer vision library
import cv2


def noise_reduction(image_original):
    """
    Method used to reduce noise in image

    Parameters:
    -----------------
        original_image (img): Original image in PIL format

    Returns:
    -----------------
        image_result (img): Image in PIL format

    """

    # Reading the image and ist attributes from PIL format
    image = cv2.cvtColor(np.array(image_original), cv2.COLOR_RGB2BGR)
    
    # Treating noise and transforming to RGB
    image_result = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    image_result = cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB)

    # Transforming the image to PIL format
    image_result = Image.fromarray(image_result)
    
    return image_result


def show_image_and_histogram(original_image, edited_image):
    """
    Method used to show image and its histogram

    Parameters:
    -----------------
        original_image (img): Original image in PIL format
        edited_image (img): Edited image in PIL format

    Returns:
    -----------------
        None
        Plot original and edited image with their histograms

    """
    
    # Reading the image and ist attributes from PIL format
    original_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    edited_image = cv2.cvtColor(np.array(edited_image), cv2.COLOR_RGB2BGR)
    edited_image = cv2.cvtColor(edited_image, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(12, 8))

    ax1, ax2, ax3, ax4 = fig.add_subplot(221), fig.add_subplot(222), \
        fig.add_subplot(223), fig.add_subplot(224)

    ax1.imshow(original_image)
    ax1.set_title("Original image", fontsize=14)
    ax1.grid(None)
    ax1.axis("off")

    ax2.hist(np.array(original_image).flatten(), bins=range(256),
             facecolor="#2ab0ff", edgecolor="#169acf", linewidth=0.5)
    ax2.set_title("Histogram", fontsize=14)

    ax3.imshow(edited_image)
    ax3.set_title("Image after preprocessing", fontsize=14)
    ax3.grid(None)
    ax3.axis("off")

    ax4.hist(np.array(edited_image).flatten(), bins=range(256),
             facecolor="#2BDC6C", edgecolor="#0CD355", linewidth=0.5)
    ax4.set_title("Histogram after preprocessing", fontsize=14)

    plt.tight_layout()
    plt.show()
    

def gaussian_blur(image_original):
    """
    Method used to treat the image with GaussianBlur

    Parameters:
    -----------------
        original_image (img): Original image in PIL format

    Returns:
    -----------------
        image_result (img): Image in PIL format

    """

    # Reading the image and ist attributes from PIL format
    image = cv2.cvtColor(np.array(image_original), cv2.COLOR_RGB2BGR)
    
    # Treating noise and transforming to RGB
    image_result = cv2.GaussianBlur(image, (5, 5), 0)
    image_result = cv2.cvtColor(image_result, cv2.COLOR_BGR2RGB)

    # Transforming the image to PIL format
    image_result = Image.fromarray(image_result)
    
    return image_result