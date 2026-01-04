"""
shi_tomasi.py — Shi-Tomasi Corner Detector

"""

import numpy as np
from utils import load_gray, show, compute_gradients, gaussian_blur, non_max_suppression, draw_keypoints


def shi_tomasi_response(img, window_sigma=2.0):
    
    Ix, Iy = compute_gradients(img)
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    Sxx = gaussian_blur(Ixx, sigma=window_sigma)
    Syy = gaussian_blur(Iyy, sigma=window_sigma)
    Sxy = gaussian_blur(Ixy, sigma=window_sigma)
    
    half_trace = (Sxx + Syy) / 2
    diff_half = (Sxx - Syy) / 2
    sqrt_term = np.sqrt(diff_half * diff_half + Sxy * Sxy)
    
    # eigenvalues
    lambda1 = half_trace + sqrt_term  # larger eigenvalue
    lambda2 = half_trace - sqrt_term  # smaller eigenvalue
    
    # Shi-Tomasi response is the minimum eigenvalue
    response = lambda2
    
    return response


def shi_tomasi_corners(img, window_sigma=2.0, threshold=0.01, nms_radius=5):
   
    # blur and get response
    blurred = gaussian_blur(img, sigma=1.0)
    response = shi_tomasi_response(blurred, window_sigma=window_sigma)
    
    # Threshold
    max_response = np.max(response)
    threshold_value = threshold * max_response
    
    response_thresholded = response.copy()
    for i in range(response.shape[0]):
        for j in range(response.shape[1]):
            if response[i, j] < threshold_value:
                response_thresholded[i, j] = 0
    
    response_nms = non_max_suppression(response_thresholded, radius=nms_radius)

    corners = []
    for i in range(response_nms.shape[0]):
        for j in range(response_nms.shape[1]):
            if response_nms[i, j] > 0:
                x = j
                y = i
                corners.append((x, y))
    
    return corners


if __name__ == "__main__":
    print("Shi-Tomasi Corner Detector")
    print("=" * 40)
    
    # Load image
    img = load_gray("data/sample.jpg")
    if img is None:
        print("Could not load image!")
        exit()
    
    print("Image shape:", img.shape)
    
    # Detect corners
    print("Detecting corners...")
    corners = shi_tomasi_corners(img, threshold=0.06, nms_radius=10)
    print("Found", len(corners), "corners")
    
    # Show the response
    print("\nShowing Shi-Tomasi response...")
    response = shi_tomasi_response(gaussian_blur(img, 1.0))
    
    # Normalize for display
    response_display = response.copy()
    response_display = response_display - response_display.min()
    if response_display.max() > 0:
        response_display = response_display / response_display.max()
    show(response_display, "Shi-Tomasi Response (min eigenvalue)")
    
    # Draw corners
    print("Drawing corners...")
    img_with_corners = draw_keypoints(img, corners, radius=3, color=(0, 255, 0))
    show(img_with_corners, f"Shi-Tomasi Corners ({len(corners)} detected)")
    
    print("Done!")