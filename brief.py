"""
brief.py — BRIEF Descriptor (Binary Robust Independent Elementary Features)

"""

import numpy as np
from utils import load_gray, show, gaussian_blur, draw_keypoints
from harris import harris_corners


# Patch size around keypoint
PATCH_SIZE = 31  # 31x31 patch, keypoint at center (15, 15)
HALF_PATCH = PATCH_SIZE // 2  # 15


def generate_brief_pattern(n_pairs=256, patch_size=31, seed=42):
    np.random.seed(seed)
    half = patch_size // 2
    
    # Using Gaussian distribution for random point generation
    sigma = patch_size / 5.0
    
    pattern = np.zeros((n_pairs, 4), dtype=np.int32)
    
    for i in range(n_pairs):
        # First point
        x1 = int(np.random.normal(0, sigma))
        y1 = int(np.random.normal(0, sigma))
        
        # Clamp to patch bounds
        x1 = max(-half, min(half, x1))
        y1 = max(-half, min(half, y1))
        
        # Second point
        x2 = int(np.random.normal(0, sigma))
        y2 = int(np.random.normal(0, sigma))
        
        x2 = max(-half, min(half, x2))
        y2 = max(-half, min(half, y2))
        
        pattern[i, 0] = x1
        pattern[i, 1] = y1
        pattern[i, 2] = x2
        pattern[i, 3] = y2
    
    return pattern


def compute_brief_descriptor(img, keypoint, pattern):
    
    x = int(keypoint[0])
    y = int(keypoint[1])
    
    height = img.shape[0]
    width = img.shape[1]
    
    # Check if patch fits within image
    if y - HALF_PATCH < 0 or y + HALF_PATCH >= height:
        return None
    if x - HALF_PATCH < 0 or x + HALF_PATCH >= width:
        return None
    
    n_pairs = pattern.shape[0]
    descriptor = np.zeros(n_pairs, dtype=np.uint8)
    
    for i in range(n_pairs):
        # Get relative coordinates
        dx1 = pattern[i, 0]
        dy1 = pattern[i, 1]
        dx2 = pattern[i, 2]
        dy2 = pattern[i, 3]
        
        # Get absolute coordinates
        x1 = x + dx1
        y1 = y + dy1
        x2 = x + dx2
        y2 = y + dy2
        
        # Compare intensities
        if img[y1, x1] > img[y2, x2]:
            descriptor[i] = 1
        else:
            descriptor[i] = 0
    
    return descriptor


def compute_brief_descriptors(img, keypoints, n_bits=256):
    
    img_blurred = gaussian_blur(img, sigma=2.0)
    
    pattern = generate_brief_pattern(n_pairs=n_bits)
    
    valid_keypoints = []
    descriptors = []
    
    for kp in keypoints:
        desc = compute_brief_descriptor(img_blurred, kp, pattern)
        
        if desc is not None:
            valid_keypoints.append(kp)
            descriptors.append(desc)
    
    return valid_keypoints, descriptors


def hamming_distance(desc1, desc2):
    
    distance = 0
    for i in range(len(desc1)):
        if desc1[i] != desc2[i]:
            distance = distance + 1
    return distance


def match_brief_descriptors(descriptors1, descriptors2, max_distance=50):

    matches = []
    
    for i in range(len(descriptors1)):
        desc1 = descriptors1[i]
        
        best_idx = -1
        best_distance = max_distance + 1
        
        # Find best match in descriptors2
        for j in range(len(descriptors2)):
            desc2 = descriptors2[j]
            
            dist = hamming_distance(desc1, desc2)
            
            if dist < best_distance:
                best_distance = dist
                best_idx = j
        
        # Keep match if distance is below threshold
        if best_idx >= 0 and best_distance <= max_distance:
            matches.append((i, best_idx))
    
    return matches


def descriptor_to_string(desc, max_chars=64):
    
    s = ""   #string
    for i in range(min(len(desc), max_chars)):
        s = s + str(desc[i])
    if len(desc) > max_chars:
        s = s + "..."
    return s


if __name__ == "__main__":
    print("BRIEF Descriptor")
    print("=" * 50)
    
    # Load image
    img = load_gray("data/sample.jpg")
    if img is None:
        print("Could not load image!")
        exit()
    
    print(f"Image shape: {img.shape}")
    
    # Detect keypoints using Harris
    print("\nDetecting keypoints with Harris...")
    keypoints = harris_corners(img, threshold=0.01, nms_radius=15)
    print(f"Found {len(keypoints)} keypoints")
    
    # Compute BRIEF descriptors
    print("\nComputing BRIEF descriptors...")
    valid_kp, descriptors = compute_brief_descriptors(img, keypoints, n_bits=256)
    print(f"Computed {len(descriptors)} descriptors")
    print(f"(Some keypoints too close to border were skipped)")
    
    # Show a sample descriptor
    if len(descriptors) > 0:
        print(f"\nSample descriptor (first 64 bits):")
        print(f"  {descriptor_to_string(descriptors[0])}")
    
    # Draw keypoints with valid descriptors
    img_with_kp = draw_keypoints(img, valid_kp, radius=3, color=(0, 255, 0))
    show(img_with_kp, f"Keypoints with BRIEF descriptors ({len(valid_kp)})")
    
    # Demo: Self-matching (should find perfect matches)
    print("\nDemo: Self-matching test...")
    matches = match_brief_descriptors(descriptors, descriptors, max_distance=0)
    print(f"Self-matches found: {len(matches)} (should equal {len(descriptors)})")
    
    print("\nDone!")
    print("\nNote: BRIEF is not rotation invariant.")
    print("ORB improves this by rotating the pattern based on keypoint orientation.")
    print("\n try Running orb.py for ORB descriptor demo.")