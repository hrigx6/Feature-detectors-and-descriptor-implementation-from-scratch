"""
orb.py — ORB: Oriented FAST and Rotated BRIEF

"""

import numpy as np
from utils import load_gray, show, gaussian_blur, draw_keypoints, draw_matches
from fast import fast_corners


def compute_orientation(img, x, y, radius=15):
    
    height = img.shape[0]
    width = img.shape[1]
    
    # Check bounds
    if y - radius < 0 or y + radius >= height:
        return None
    if x - radius < 0 or x + radius >= width:
        return None
    
    m_01 = 0.0  # moment for y
    m_10 = 0.0  # moment for x
    
    # Compute moments over circular patch
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            # Check if point is within circle
            if dx * dx + dy * dy <= radius * radius:
                pixel_value = img[y + dy, x + dx]
                m_10 = m_10 + dx * pixel_value
                m_01 = m_01 + dy * pixel_value
    
    # Compute angle
    angle = np.arctan2(m_01, m_10)
    
    return angle


def generate_orb_pattern(n_pairs=256, patch_size=31, seed=42):
    
    np.random.seed(seed)
    
    half = patch_size // 2
    sigma = patch_size / 5.0
    
    pattern = np.zeros((n_pairs, 4), dtype=np.float64)  # float for rotation
    
    for i in range(n_pairs):
        x1 = np.clip(np.random.normal(0, sigma), -half, half)
        y1 = np.clip(np.random.normal(0, sigma), -half, half)
        x2 = np.clip(np.random.normal(0, sigma), -half, half)
        y2 = np.clip(np.random.normal(0, sigma), -half, half)
        
        pattern[i, 0] = x1
        pattern[i, 1] = y1
        pattern[i, 2] = x2
        pattern[i, 3] = y2
    
    return pattern


def rotate_pattern(pattern, angle):
    
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    rotated = np.zeros_like(pattern)
    
    for i in range(pattern.shape[0]):
        x1 = pattern[i, 0]
        y1 = pattern[i, 1]
        x2 = pattern[i, 2]
        y2 = pattern[i, 3]
        
        # Rotate point 1
        rotated[i, 0] = cos_a * x1 - sin_a * y1
        rotated[i, 1] = sin_a * x1 + cos_a * y1
        
        # Rotate point 2
        rotated[i, 2] = cos_a * x2 - sin_a * y2
        rotated[i, 3] = sin_a * x2 + cos_a * y2
    
    return rotated


def compute_orb_descriptor(img, x, y, angle, pattern):
    
    height = img.shape[0]
    width = img.shape[1]
    half_patch = 15
    
    # Check bounds
    if y - half_patch < 0 or y + half_patch >= height:
        return None
    if x - half_patch < 0 or x + half_patch >= width:
        return None
    
    # Rotate pattern by keypoint orientation
    rotated = rotate_pattern(pattern, angle)
    
    n_pairs = pattern.shape[0]
    descriptor = np.zeros(n_pairs, dtype=np.uint8)
    
    for i in range(n_pairs):
        # Get rotated coordinates (round to int)
        dx1 = int(round(rotated[i, 0]))
        dy1 = int(round(rotated[i, 1]))
        dx2 = int(round(rotated[i, 2]))
        dy2 = int(round(rotated[i, 3]))
        
        # Absolute coordinates
        x1 = x + dx1
        y1 = y + dy1
        x2 = x + dx2
        y2 = y + dy2
        
        # Bounds check for rotated coordinates
        if x1 < 0 or x1 >= width or y1 < 0 or y1 >= height:
            continue
        if x2 < 0 or x2 >= width or y2 < 0 or y2 >= height:
            continue
        
        # Compare intensities
        if img[y1, x1] > img[y2, x2]:
            descriptor[i] = 1
    
    return descriptor


def build_scale_pyramid(img, num_levels=8, scale_factor=1.2):
    
    pyramid = []
    current = img.copy()
    current_scale = 1.0
    
    for level in range(num_levels):
        pyramid.append((current.copy(), current_scale))
        
        # Compute new size
        new_height = int(current.shape[0] / scale_factor)
        new_width = int(current.shape[1] / scale_factor)
        
        if new_height < 20 or new_width < 20:
            break
        
        # Resize (simple subsampling)
        resized = np.zeros((new_height, new_width))
        for i in range(new_height):
            for j in range(new_width):
                src_i = int(i * scale_factor)
                src_j = int(j * scale_factor)
                src_i = min(src_i, current.shape[0] - 1)
                src_j = min(src_j, current.shape[1] - 1)
                resized[i, j] = current[src_i, src_j]
        
        current = resized
        current_scale = current_scale * scale_factor
    
    return pyramid


def orb_detect_and_compute(img, n_keypoints=500, n_levels=8, scale_factor=1.2, fast_threshold=0.1):
    
    # Build pyramid
    print(f"Building {n_levels}-level pyramid...")
    pyramid = build_scale_pyramid(img, n_levels, scale_factor)
    print(f"Built {len(pyramid)} levels")
    
    # Generate sampling pattern
    pattern = generate_orb_pattern(n_pairs=256)
    
    # Detect keypoints at each level
    all_keypoints = []  # (x, y, scale, angle, response)
    
    for level, (level_img, scale) in enumerate(pyramid):
        # Blur slightly for stability
        blurred = gaussian_blur(level_img, sigma=1.0)
        
        # Detect FAST corners
        corners = fast_corners(blurred, threshold=fast_threshold, n_contiguous=9, nms_radius=5)
        
        print(f"  Level {level}: {len(corners)} FAST corners at scale {scale:.2f}")
        
        # Compute orientation for each corner
        for corner in corners:
            cx, cy = corner
            angle = compute_orientation(blurred, cx, cy, radius=15)
            
            if angle is not None:
                # Convert back to original image coordinates
                orig_x = cx * scale
                orig_y = cy * scale
                all_keypoints.append((orig_x, orig_y, scale, angle, level_img[cy, cx]))
    
    # Sort by response (intensity at corner) and keep top n_keypoints
    all_keypoints.sort(key=lambda kp: -kp[4])  # sort by response descending
    all_keypoints = all_keypoints[:n_keypoints]
    
    print(f"Kept top {len(all_keypoints)} keypoints")
    
    # Compute descriptors
    # Use blurred original image for descriptor computation
    img_blurred = gaussian_blur(img, sigma=2.0)
    
    keypoints = []
    descriptors = []
    
    for kp in all_keypoints:
        orig_x, orig_y, scale, angle, response = kp
        
        x = int(round(orig_x))
        y = int(round(orig_y))
        
        desc = compute_orb_descriptor(img_blurred, x, y, angle, pattern)
        
        if desc is not None:
            keypoints.append((x, y, scale, angle))
            descriptors.append(desc)
    
    print(f"Computed {len(descriptors)} descriptors")
    
    return keypoints, descriptors


def hamming_distance(desc1, desc2):

    distance = 0
    for i in range(len(desc1)):
        if desc1[i] != desc2[i]:
            distance = distance + 1
    return distance


def match_orb_descriptors(desc1, desc2, max_distance=64, ratio_thresh=0.75):
    
    matches = []
    
    for i in range(len(desc1)):
        d1 = desc1[i]
        
        best_dist = 999
        second_dist = 999
        best_idx = -1
        
        for j in range(len(desc2)):
            d2 = desc2[j]
            dist = hamming_distance(d1, d2)
            
            if dist < best_dist:
                second_dist = best_dist
                best_dist = dist
                best_idx = j
            elif dist < second_dist:
                second_dist = dist
        
        # Apply ratio test
        if best_idx >= 0 and best_dist <= max_distance:
            if second_dist > 0 and (best_dist / second_dist) < ratio_thresh:
                matches.append((i, best_idx, best_dist))
    
    return matches


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("ORB: Oriented FAST and Rotated BRIEF")
    print("=" * 50)
    
    # Load image
    img = load_gray("data/sample.jpg")
    if img is None:
        print("Could not load image!")
        exit()
    
    print(f"Image shape: {img.shape}")
    print()
    
    # Detect and compute
    print("Detecting ORB features...")
    keypoints, descriptors = orb_detect_and_compute(
        img, 
        n_keypoints=500,
        n_levels=4,
        fast_threshold=0.12
    )
    
    print(f"\nFinal: {len(keypoints)} keypoints with descriptors")
    
    # Show some keypoint info
    if len(keypoints) > 0:
        print("\nSample keypoints (x, y, scale, angle):")
        for i in range(min(5, len(keypoints))):
            kp = keypoints[i]
            print(f"  {i}: x={kp[0]}, y={kp[1]}, scale={kp[2]:.2f}, angle={np.degrees(kp[3]):.1f}°")
    
    # Draw keypoints (just x, y for visualization)
    kp_coords = [(kp[0], kp[1]) for kp in keypoints]
    img_with_kp = draw_keypoints(img, kp_coords, radius=3, color=(0, 255, 0))
    show(img_with_kp, f"ORB Keypoints ({len(keypoints)} detected)")
    
    # Self-matching test
    print("\nSelf-matching test...")
    matches = match_orb_descriptors(descriptors, descriptors, max_distance=0)
    print(f"Perfect self-matches: {len(matches)} (should be {len(descriptors)})")
    
    print("\nDone!")
    print("\nORB is used in ORB-SLAM, ORB-SLAM2, ORB-SLAM3")