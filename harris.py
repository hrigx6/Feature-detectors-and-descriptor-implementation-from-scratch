"""
harris.py — Harris Corner Detector from scratch

"""

import numpy as np
from utils import load_gray, show, compute_gradients, gaussian_blur, non_max_suppression, draw_keypoints


def harris_response(img, k=0.04, window_sigma=2.0):
    
    Ix, Iy = compute_gradients(img)
    
    # Compute products of gradients at each pixel
    Ixx = Ix * Ix   
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    Sxx = gaussian_blur(Ixx, sigma=window_sigma)
    Syy = gaussian_blur(Iyy, sigma=window_sigma)
    Sxy = gaussian_blur(Ixy, sigma=window_sigma)
    
    # Compute Harris response R = det(M) - k * trace(M)^2
    # For 2x2 matrix M = [[Sxx, Sxy], [Sxy, Syy]]
    det_M = Sxx * Syy - Sxy * Sxy
    trace_M = Sxx + Syy
    
    response = det_M - k * trace_M * trace_M
    
    return response


def harris_corners(img, k=0.04, window_sigma=2.0, threshold=0.01, nms_radius=5):
    
    blurred = gaussian_blur(img, sigma=1.0)
    
    # Compute Harris response
    response = harris_response(blurred, k=k, window_sigma=window_sigma)
    
    max_response = np.max(response)
    threshold_value = threshold * max_response
    
    # set weak responses to zero
    response_thresholded = response.copy()
    for i in range(response.shape[0]):
        for j in range(response.shape[1]):
            if response[i, j] < threshold_value:
                response_thresholded[i, j] = 0
    
    # Non-max suppression - keep only local maxima
    response_nms = non_max_suppression(response_thresholded, radius=nms_radius)
    
    # Extract corner coordinates
    corners = []
    for i in range(response_nms.shape[0]):       # i is row (y coordinate)
        for j in range(response_nms.shape[1]):   # j is column (x coordinate)
            if response_nms[i, j] > 0:
                # keypoints are (x, y) so we use (j, i)
                x = j
                y = i
                corners.append((x, y))
    
    return corners


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Harris Corner Detector")
    print("=" * 40)
    
    # Load image
    img = load_gray("data/sample.jpg")
    if img is None:
        print("Could not load image!")
        exit()
    
    print("Image shape:", img.shape)
    
    # Detect corners
    print("Detecting corners...")
    corners = harris_corners(img, k=0.04, threshold=0.01, nms_radius=10)
    print("Found", len(corners), "corners")
    
    # Show the Harris response (for visualization)
    print("\nShowing Harris response...")
    response = harris_response(gaussian_blur(img, 1.0))
    
    # Normalize response for display
    response_display = response.copy()
    response_display = response_display - response_display.min()  # shift to start at 0
    response_display = response_display / response_display.max()  # scale to 0-1
    show(response_display, "Harris Response (brighter = stronger corner)")
    
    # Draw corners on image
    print("Drawing corners...")
    img_with_corners = draw_keypoints(img, corners, radius=5, color=(0, 255, 0))
    show(img_with_corners, f"Harris Corners ({len(corners)} detected)")
    
    print("Done!")