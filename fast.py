"""
fast.py — FAST Corner Detector (Features from Accelerated Segment Test)

"""

import numpy as np
from utils import load_gray, show, non_max_suppression, draw_keypoints

CIRCLE_OFFSETS = [
    (-3,  0),  # position 1  (top)
    (-3,  1),  # position 2
    (-2,  2),  # position 3
    (-1,  3),  # position 4
    ( 0,  3),  # position 5  (right)
    ( 1,  3),  # position 6
    ( 2,  2),  # position 7
    ( 3,  1),  # position 8
    ( 3,  0),  # position 9  (bottom)
    ( 3, -1),  # position 10
    ( 2, -2),  # position 11
    ( 1, -3),  # position 12
    ( 0, -3),  # position 13 (left)
    (-1, -3),  # position 14
    (-2, -2),  # position 15
    (-3, -1),  # position 16
]


def get_circle_pixels(img, row, col):
    pixels = []
    for i in range(16):
        dy = CIRCLE_OFFSETS[i][0]
        dx = CIRCLE_OFFSETS[i][1]
        pixel_value = img[row + dy, col + dx]
        pixels.append(pixel_value)
    return pixels


def check_contiguous_arc(states, n_contiguous):
   
    # We need to check for wrap-around (pixel 16 is next to pixel 1)
    # So we duplicate the array: [1,2,3,...,16,1,2,3,...,16]
    extended = states + states
    
    # Check for arc of all brighter (1s)
    count_bright = 0
    for i in range(len(extended)):
        if extended[i] == 1:
            count_bright = count_bright + 1
            if count_bright >= n_contiguous:
                return True
        else:
            count_bright = 0
    
    # Check for arc of all darker (-1s)
    count_dark = 0
    for i in range(len(extended)):
        if extended[i] == -1:
            count_dark = count_dark + 1
            if count_dark >= n_contiguous:
                return True
        else:
            count_dark = 0
    
    return False


def fast_quick_reject(img, row, col, threshold):
    
    center = img[row, col]
    high = center + threshold
    low = center - threshold
    
    # Get the 4 cardinal pixels (positions 1, 5, 9, 13)
    p1  = img[row - 3, col]      # top
    p5  = img[row, col + 3]      # right
    p9  = img[row + 3, col]      # bottom
    p13 = img[row, col - 3]      # left
    
    # Count how many are brighter than high
    count_bright = 0
    if p1 > high:
        count_bright = count_bright + 1
    if p5 > high:
        count_bright = count_bright + 1
    if p9 > high:
        count_bright = count_bright + 1
    if p13 > high:
        count_bright = count_bright + 1
    
    # If at least 3 are brighter, might be a corner (can't reject)
    if count_bright >= 3:
        return True
    
    # Count how many are darker than low
    count_dark = 0
    if p1 < low:
        count_dark = count_dark + 1
    if p5 < low:
        count_dark = count_dark + 1
    if p9 < low:
        count_dark = count_dark + 1
    if p13 < low:
        count_dark = count_dark + 1
    
    # If at least 3 are darker, might be a corner (can't reject)
    if count_dark >= 3:
        return True
    
    # Can reject — not enough bright or dark pixels
    return False


def is_fast_corner(img, row, col, threshold, n_contiguous=9):
   
    center = img[row, col]
    high = center + threshold
    low = center - threshold
    
    # Get all 16 pixels on the circle
    circle_pixels = get_circle_pixels(img, row, col)
    
    # Classify each pixel: 1 = brighter, -1 = darker, 0 = similar
    states = []
    for i in range(16):
        p = circle_pixels[i]
        if p > high:
            states.append(1)    # brighter
        elif p < low:
            states.append(-1)   # darker
        else:
            states.append(0)    # similar
    
    # Check for contiguous arc
    return check_contiguous_arc(states, n_contiguous)


def fast_response(img, row, col, threshold):
    
    center = img[row, col]
    circle_pixels = get_circle_pixels(img, row, col)
    
    # Sum of absolute differences
    score = 0
    for i in range(16):
        diff = abs(circle_pixels[i] - center)
        if diff > threshold:
            score = score + diff
    
    return score


def fast_corners(img, threshold=0.1, n_contiguous=9, nms_radius=5):
    
    height = img.shape[0]
    width = img.shape[1]
    
    # Create response image for NMS
    response = np.zeros((height, width))
    
    # We can't check pixels too close to the border (need 3 pixel margin)
    margin = 3
    
    print("Scanning for FAST corners...")
    
    # Check each pixel
    for row in range(margin, height - margin):
        for col in range(margin, width - margin):
            
            # Quick rejection test first
            if not fast_quick_reject(img, row, col, threshold):
                continue
            
            # Full test
            if is_fast_corner(img, row, col, threshold, n_contiguous):
                # Compute corner strength
                response[row, col] = fast_response(img, row, col, threshold)
    
    # Non-max suppression
    response_nms = non_max_suppression(response, radius=nms_radius)
    
    # Extract corners
    corners = []
    for row in range(height):
        for col in range(width):
            if response_nms[row, col] > 0:
                x = col
                y = row
                corners.append((x, y))
    
    return corners


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("FAST Corner Detector")
    print("=" * 40)
    
    # Load image
    img = load_gray("data/sample.jpg")
    if img is None:
        print("Could not load image!")
        exit()
    
    print("Image shape:", img.shape)
    
    # Detect corners
    print("Detecting corners...")
    corners = fast_corners(img, threshold=0.32, n_contiguous=9, nms_radius=10)
    print("Found", len(corners), "corners")
    
    # Draw corners
    print("Drawing corners...")
    img_with_corners = draw_keypoints(img, corners, radius=3, color=(0, 255, 0))
    show(img_with_corners, f"FAST Corners ({len(corners)} detected)")
    
    print("Done!")