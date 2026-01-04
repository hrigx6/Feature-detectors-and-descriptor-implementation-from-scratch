"""
utils.py — Shared utilities for basic operations.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage


def load_gray(path, resize=None):
    # load as grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: Could not load image at", path)
        return None
    
    if resize is not None:
        width = resize[0]
        height = resize[1]
        img = cv2.resize(img, (width, height))
    
    img = img.astype(np.float64)
    img = img / 255.0
    
    return img


def show(img, title=None):
    
    plt.figure(figsize=(10, 8))
    
    # check if grayscale or color
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
    
    if title is not None:
        plt.title(title)
    
    plt.axis('off')
    plt.show()

def draw_keypoints(img, keypoints, radius=5, color=(0, 255, 0)):
    output = img.copy()

    # convert float to uint8 
    if output.dtype == np.float64 or output.dtype == np.float32:
        output = (output * 255).astype(np.uint8)
    
    if len(output.shape) == 2:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    
    # draw a circle for each keypoint
    for i in range(len(keypoints)):
        x = keypoints[i][0]
        y = keypoints[i][1]
        center = (int(x), int(y))
        cv2.circle(output, center, radius, color, thickness=1)
    
    return output


def draw_matches(img1, img2, kp1, kp2, matches):
    
    img1_copy = img1.copy()
    img2_copy = img2.copy()
    
    if img1_copy.dtype == np.float64 or img1_copy.dtype == np.float32:
        img1_copy = (img1_copy * 255).astype(np.uint8)
    if img2_copy.dtype == np.float64 or img2_copy.dtype == np.float32:
        img2_copy = (img2_copy * 255).astype(np.uint8)
    
    if len(img1_copy.shape) == 2:
        img1_copy = cv2.cvtColor(img1_copy, cv2.COLOR_GRAY2BGR)
    if len(img2_copy.shape) == 2:
        img2_copy = cv2.cvtColor(img2_copy, cv2.COLOR_GRAY2BGR)
    
    output = np.hstack([img1_copy, img2_copy])
    
    # get width of img1
    width1 = img1_copy.shape[1]
    
    # loop through each match and draw a line
    for i in range(len(matches)):
        idx1 = matches[i][0]
        idx2 = matches[i][1]
        
        # get keypoint coordinates
        x1 = kp1[idx1][0]
        y1 = kp1[idx1][1]
        
        x2 = kp2[idx2][0] + width1  # offset by img1 width
        y2 = kp2[idx2][1]
        
        # convert to int tuples
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        
        # draw line
        color = (0, 255, 0)  # green
        cv2.line(output, pt1, pt2, color, thickness=1)
    
    return output

def gaussian_kernel(sigma):
    
    size = int(6 * sigma)
    if size % 2 == 0:
        size = size + 1
    if size < 3: 
        size = 3
    
    kernel = np.zeros(size)
    
    center = size // 2
    
    for i in range(size):
        x = i - center
        kernel[i] = np.exp(-(x * x) / (2 * sigma * sigma))
    
    # normalize
    total = np.sum(kernel)
    kernel = kernel / total
    
    return kernel


def gaussian_blur(img, sigma):

    kernel = gaussian_kernel(sigma)
    
    blurred = ndimage.convolve1d(img, kernel, axis=1)   #in X direction
    blurred = ndimage.convolve1d(blurred, kernel, axis=0)  # in direction Y
    
    return blurred


def compute_gradients(img):
   
    # sobel_x kernel
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    # sobel_y kernel
    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)
    
    
    Ix = ndimage.convolve(img, sobel_x)
    Iy = ndimage.convolve(img, sobel_y)
    
    return Ix, Iy


def non_max_suppression(response, radius):
   
    # calculate neighborhood size
    size = 2 * radius + 1
    
    # find local maximum in each neighborhood
    local_max = ndimage.maximum_filter(response, size=size)
    
    # create output array
    output = response.copy()
    
    # keep only values that equal the local max
    for i in range(response.shape[0]):
        for j in range(response.shape[1]):
            if response[i, j] != local_max[i, j]:
                output[i, j] = 0
    
    return output


if __name__ == "__main__":
    print("Testing utils.py...")
    print()
    
    # test gaussian kernel
    print("1. Testing gaussian_kernel()")
    kernel = gaussian_kernel(1.0)
    print("   Kernel:", kernel)
    print("   Sum:", np.sum(kernel))
    print()
    
    # test load_gray
    print("2. Testing load_gray()")
    img = load_gray("data/sample.jpg")
    if img is not None:
        print("   Image shape:", img.shape)
        print("   Min value:", img.min())
        print("   Max value:", img.max())
        print()
        
        # test show
        print("3. Testing show()")
        show(img, "Original Image")
        
        # test blur
        print("4. Testing gaussian_blur()")
        blurred = gaussian_blur(img, sigma=2.0)
        show(blurred, "Blurred (sigma=2)")
        
        # test gradients
        print("5. Testing compute_gradients()")
        Ix, Iy = compute_gradients(img)
        show(np.abs(Ix), "Gradient X (absolute)")
        show(np.abs(Iy), "Gradient Y (absolute)")
        
        # test draw_keypoints
        print("6. Testing draw_keypoints()")
        fake_keypoints = [(500, 500), (500, 350), (300, 200)]
        img_with_kp = draw_keypoints(img, fake_keypoints)
        show(img_with_kp, "Image with keypoints")
    
    print("Done!")