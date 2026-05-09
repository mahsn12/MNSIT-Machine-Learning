import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize



def preprocess(flat_image: np.ndarray, canvas: int = 28) -> np.ndarray:
    """
    Full preprocessing pipeline:
      1. Reshape flat vector → 2-D grayscale
      2. Binarize with Otsu thresholding
      3. Center the digit (centroid → canvas center)
      4. Resize to fixed canvas
    Returns a binary uint8 image (0 / 255).
    """
    # Step 1 ── reshape
    img = flat_image.reshape(canvas, canvas).astype(np.uint8)

    # Step 2 ── Otsu binarization
    #   B(x,y) = 1 if I(x,y) > t_otsu, else 0
    _, binary = cv2.threshold(img, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    M = cv2.moments(binary)
    if M["m00"] != 0 :
        cx = M["m10"]//M["m00"]
        cy = M["m01"]//M["m00"]
        shftX = (canvas // 2) - cx
        shftY = (canvas // 2) - cy
        T = np.array([[1,0,shftX],
                      [0,1,shftY]])
        binary = cv2.warpAffine(binary,dsize=(canvas,canvas),M=T)
    return binary
    


def skeletonize_image(binary_img: np.ndarray) -> np.ndarray:
    """
    Zhang-Suen thinning via skimage.
    Input  : binary uint8 image (0 / 255)
    Output : skeleton uint8 image (0 / 255), every stroke 1-pixel wide
    """
    # skimage expects a bool array (True = foreground)
    boolOFimg = binary_img > 127
    skeleton = skeletonize(boolOFimg)
    return (skeleton*255).astype(np.uint8)

