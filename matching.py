"""
matching.py — Feature Matching

"""

import numpy as np
from utils import load_gray, show, gaussian_blur, draw_keypoints, draw_matches
from harris import harris_corners
from brief import compute_brief_descriptors


# =============================================================================
# Distance Functions
# =============================================================================

def hamming_distance(desc1, desc2):
    
    distance = 0
    for i in range(len(desc1)):
        if desc1[i] != desc2[i]:
            distance = distance + 1
    return distance


def l2_distance(desc1, desc2):
    total = 0.0
    for i in range(len(desc1)):
        diff = desc1[i] - desc2[i]
        total = total + diff * diff
    return np.sqrt(total)

def brute_force_match(descriptors1, descriptors2, distance_func, max_distance=None):
    
    matches = []
    
    for i in range(len(descriptors1)):
        desc1 = descriptors1[i]
        
        best_idx = -1
        best_dist = float('inf')
        
        # Find nearest neighbor in descriptors2
        for j in range(len(descriptors2)):
            desc2 = descriptors2[j]
            dist = distance_func(desc1, desc2)
            
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        
        # Apply max distance threshold if specified
        if best_idx >= 0:
            if max_distance is None or best_dist <= max_distance:
                matches.append((i, best_idx, best_dist))
    
    return matches



def ratio_test_match(descriptors1, descriptors2, distance_func, ratio_thresh=0.75, max_distance=None):
    
    matches = []
    
    for i in range(len(descriptors1)):
        desc1 = descriptors1[i]
        
        best_idx = -1
        best_dist = float('inf')
        second_dist = float('inf')
        
        # Find best and second best matches
        for j in range(len(descriptors2)):
            desc2 = descriptors2[j]
            dist = distance_func(desc1, desc2)
            
            if dist < best_dist:
                # Current best becomes second best
                second_dist = best_dist
                # New best
                best_dist = dist
                best_idx = j
            elif dist < second_dist:
                # Update second best
                second_dist = dist
        
        # Apply ratio test
        if best_idx >= 0 and second_dist > 0:
            ratio = best_dist / second_dist
            
            if ratio < ratio_thresh:
                # Apply max distance if specified
                if max_distance is None or best_dist <= max_distance:
                    matches.append((i, best_idx, best_dist))
    
    return matches

def cross_check_match(descriptors1, descriptors2, distance_func, max_distance=None):
    
    # Forward matching: img1 → img2
    forward_matches = {}  # idx1 → (idx2, distance)
    
    for i in range(len(descriptors1)):
        desc1 = descriptors1[i]
        best_idx = -1
        best_dist = float('inf')
        
        for j in range(len(descriptors2)):
            dist = distance_func(desc1, descriptors2[j])
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        
        if best_idx >= 0:
            forward_matches[i] = (best_idx, best_dist)
    
    # Backward matching: img2 → img1
    backward_matches = {}  # idx2 → idx1
    
    for j in range(len(descriptors2)):
        desc2 = descriptors2[j]
        best_idx = -1
        best_dist = float('inf')
        
        for i in range(len(descriptors1)):
            dist = distance_func(descriptors1[i], desc2)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        if best_idx >= 0:
            backward_matches[j] = best_idx
    
    # Cross-check: keep only mutual matches
    matches = []
    
    for i, (j, dist) in forward_matches.items():
        # Check if backward match agrees
        if j in backward_matches and backward_matches[j] == i:
            if max_distance is None or dist <= max_distance:
                matches.append((i, j, dist))
    
    return matches

def robust_match(descriptors1, descriptors2, distance_func, ratio_thresh=0.75, max_distance=None):
    
    # Forward: img1 → img2 with ratio test
    forward_matches = {}
    
    for i in range(len(descriptors1)):
        desc1 = descriptors1[i]
        
        best_idx = -1
        best_dist = float('inf')
        second_dist = float('inf')
        
        for j in range(len(descriptors2)):
            dist = distance_func(desc1, descriptors2[j])
            
            if dist < best_dist:
                second_dist = best_dist
                best_dist = dist
                best_idx = j
            elif dist < second_dist:
                second_dist = dist
        
        # Ratio test
        if best_idx >= 0 and second_dist > 0:
            if best_dist / second_dist < ratio_thresh:
                forward_matches[i] = (best_idx, best_dist)
    
    # Backward: img2 → img1 with ratio test
    backward_matches = {}
    
    for j in range(len(descriptors2)):
        desc2 = descriptors2[j]
        
        best_idx = -1
        best_dist = float('inf')
        second_dist = float('inf')
        
        for i in range(len(descriptors1)):
            dist = distance_func(descriptors1[i], desc2)
            
            if dist < best_dist:
                second_dist = best_dist
                best_dist = dist
                best_idx = i
            elif dist < second_dist:
                second_dist = dist
        
        # Ratio test
        if best_idx >= 0 and second_dist > 0:
            if best_dist / second_dist < ratio_thresh:
                backward_matches[j] = best_idx
    
    # Step 2: Cross-check
    matches = []
    
    for i, (j, dist) in forward_matches.items():
        if j in backward_matches and backward_matches[j] == i:
            if max_distance is None or dist <= max_distance:
                matches.append((i, j, dist))
    
    return matches

def matches_to_vis_format(matches):
    """
    Convert (idx1, idx2, distance) matches to (idx1, idx2) for draw_matches.
    """
    return [(m[0], m[1]) for m in matches]


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Feature Matching Demo")
    print("=" * 50)
    
    # Load two images
    # For testing, we'll use the same image (should get perfect matches)
    # In real use, you'd load two different views of the same scene
    
    img1 = load_gray("data/sample.jpg")
    if img1 is None:
        print("Could not load image!")
        exit()
    
    # Create a "second image" by shifting/cropping the first
    # This simulates a small viewpoint change
    print("Creating test image pair...")
    
    # Shift image by 20 pixels (simulates camera movement)
    shift = 20
    img2 = np.zeros_like(img1)
    img2[:, :-shift] = img1[:, shift:]  # shift left
    
    print(f"Image 1 shape: {img1.shape}")
    print(f"Image 2 shape: {img2.shape}")
    
    # Detect keypoints
    print("\nDetecting Harris corners...")
    kp1 = harris_corners(img1, threshold=0.01, nms_radius=15)
    kp2 = harris_corners(img2, threshold=0.01, nms_radius=15)
    print(f"Image 1: {len(kp1)} keypoints")
    print(f"Image 2: {len(kp2)} keypoints")
    
    # Compute BRIEF descriptors
    print("\nComputing BRIEF descriptors...")
    valid_kp1, desc1 = compute_brief_descriptors(img1, kp1, n_bits=256)
    valid_kp2, desc2 = compute_brief_descriptors(img2, kp2, n_bits=256)
    print(f"Image 1: {len(desc1)} descriptors")
    print(f"Image 2: {len(desc2)} descriptors")
    
    # Test different matching strategies
    print("\n" + "=" * 50)
    print("Testing matching strategies:")
    print("=" * 50)
    
    # 1. Brute force
    print("\n1. Brute Force (no filtering):")
    bf_matches = brute_force_match(desc1, desc2, hamming_distance, max_distance=64)
    print(f"   Matches: {len(bf_matches)}")
    
    # 2. Ratio test
    print("\n2. Ratio Test (threshold=0.75):")
    ratio_matches = ratio_test_match(desc1, desc2, hamming_distance, ratio_thresh=0.75)
    print(f"   Matches: {len(ratio_matches)}")
    
    # 3. Cross-check
    print("\n3. Cross-Check:")
    cc_matches = cross_check_match(desc1, desc2, hamming_distance, max_distance=64)
    print(f"   Matches: {len(cc_matches)}")
    
    # 4. Robust (ratio + cross-check)
    print("\n4. Robust (Ratio + Cross-Check):")
    robust_matches = robust_match(desc1, desc2, hamming_distance, ratio_thresh=0.75)
    print(f"   Matches: {len(robust_matches)}")
    
    # Visualize robust matches
    print("\nVisualizing robust matches...")
    vis_matches = matches_to_vis_format(robust_matches)
    match_img = draw_matches(img1, img2, valid_kp1, valid_kp2, vis_matches)
    show(match_img, f"Robust Matches ({len(robust_matches)} matches)")
    
    # Show match distance statistics
    if len(robust_matches) > 0:
        distances = [m[2] for m in robust_matches]
        print(f"\nMatch distances:")
        print(f"   Min: {min(distances)}")
        print(f"   Max: {max(distances)}")
        print(f"   Mean: {np.mean(distances):.1f}")
    
    print("\nDone!")
    print("\nNext step: Use RANSAC to filter matches geometrically!")