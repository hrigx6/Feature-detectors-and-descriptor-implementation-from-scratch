"""
ransac.py — RANSAC for Geometric Verification

Theory:
    RANSAC (Random Sample Consensus) finds a geometric model that fits
    most matches, even when many matches are wrong (outliers).
    
    Algorithm:
    1. Repeat N times:
        a. Randomly select minimal sample (4 points for homography)
        b. Fit model to these points
        c. Count inliers (matches that agree with model)
        d. If best so far, save it
    2. Refit model using all inliers
    3. Return model + inliers
    
    Why it works:
    - If 50% of matches are wrong, picking 4 randomly gives 6.25% chance
      of all 4 being correct. After 100 iterations, >99% chance of success.
    
    We implement:
    - Homography estimation (planar scenes)
    - Can be extended to Fundamental matrix (general scenes)
"""

import numpy as np
from utils import load_gray, show, draw_matches, draw_keypoints
from harris import harris_corners
from brief import compute_brief_descriptors
from matching import robust_match, hamming_distance


# =============================================================================
# Homography Estimation
# =============================================================================

def compute_homography_dlt(pts1, pts2):
    """
    Compute homography using Direct Linear Transform (DLT).
    
    Given 4+ point correspondences, find H such that:
        pts2 = H @ pts1  (in homogeneous coordinates)
    
    Args:
        pts1: Nx2 array of points in image 1
        pts2: Nx2 array of points in image 2
    
    Returns:
        H: 3x3 homography matrix
    """
    n = pts1.shape[0]
    
    if n < 4:
        return None
    
    # Build the A matrix for Ah = 0
    # Each correspondence gives 2 equations
    A = np.zeros((2 * n, 9))
    
    for i in range(n):
        x1 = pts1[i, 0]
        y1 = pts1[i, 1]
        x2 = pts2[i, 0]
        y2 = pts2[i, 1]
        
        # First equation (for x2)
        A[2*i, 0] = -x1
        A[2*i, 1] = -y1
        A[2*i, 2] = -1
        A[2*i, 3] = 0
        A[2*i, 4] = 0
        A[2*i, 5] = 0
        A[2*i, 6] = x2 * x1
        A[2*i, 7] = x2 * y1
        A[2*i, 8] = x2
        
        # Second equation (for y2)
        A[2*i+1, 0] = 0
        A[2*i+1, 1] = 0
        A[2*i+1, 2] = 0
        A[2*i+1, 3] = -x1
        A[2*i+1, 4] = -y1
        A[2*i+1, 5] = -1
        A[2*i+1, 6] = y2 * x1
        A[2*i+1, 7] = y2 * y1
        A[2*i+1, 8] = y2
    
    # Solve using SVD
    # The solution is the last column of V (corresponding to smallest singular value)
    try:
        U, S, Vt = np.linalg.svd(A)
        h = Vt[-1, :]  # Last row of Vt = last column of V
        H = h.reshape(3, 3)
        
        # Normalize so H[2,2] = 1
        if abs(H[2, 2]) > 1e-10:
            H = H / H[2, 2]
        
        return H
    except:
        return None


def apply_homography(H, pts):
    """
    Apply homography to points.
    
    Args:
        H: 3x3 homography matrix
        pts: Nx2 array of points
    
    Returns:
        transformed: Nx2 array of transformed points
    """
    n = pts.shape[0]
    transformed = np.zeros((n, 2))
    
    for i in range(n):
        x = pts[i, 0]
        y = pts[i, 1]
        
        # Convert to homogeneous coordinates
        # [x', y', w'] = H @ [x, y, 1]
        x_prime = H[0, 0] * x + H[0, 1] * y + H[0, 2]
        y_prime = H[1, 0] * x + H[1, 1] * y + H[1, 2]
        w_prime = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        
        # Convert back from homogeneous
        if abs(w_prime) > 1e-10:
            transformed[i, 0] = x_prime / w_prime
            transformed[i, 1] = y_prime / w_prime
        else:
            transformed[i, 0] = 0
            transformed[i, 1] = 0
    
    return transformed


def compute_reprojection_error(H, pts1, pts2):
    """
    Compute reprojection error for each correspondence.
    
    Error = distance between H @ pts1 and pts2
    
    Args:
        H: 3x3 homography
        pts1, pts2: Nx2 arrays of corresponding points
    
    Returns:
        errors: N array of reprojection errors (Euclidean distance)
    """
    # Transform pts1 using H
    pts1_transformed = apply_homography(H, pts1)
    
    # Compute error for each point
    n = pts1.shape[0]
    errors = np.zeros(n)
    
    for i in range(n):
        dx = pts1_transformed[i, 0] - pts2[i, 0]
        dy = pts1_transformed[i, 1] - pts2[i, 1]
        errors[i] = np.sqrt(dx * dx + dy * dy)
    
    return errors


# =============================================================================
# RANSAC
# =============================================================================

def ransac_homography(pts1, pts2, n_iterations=1000, threshold=5.0, min_inliers=10):
    """
    RANSAC for homography estimation.
    
    Args:
        pts1: Nx2 array of points in image 1
        pts2: Nx2 array of points in image 2
        n_iterations: number of RANSAC iterations
        threshold: max reprojection error to be considered inlier (pixels)
        min_inliers: minimum inliers to accept model
    
    Returns:
        best_H: best homography found (or None if failed)
        inlier_mask: boolean array indicating inliers
    """
    n = pts1.shape[0]
    
    if n < 4:
        print("Need at least 4 points for homography")
        return None, None
    
    best_H = None
    best_inlier_count = 0
    best_inlier_mask = None
    
    print(f"Running RANSAC with {n} matches, {n_iterations} iterations...")
    
    for iteration in range(n_iterations):
        # Step 1: Randomly select 4 points
        indices = np.random.choice(n, size=4, replace=False)
        
        sample_pts1 = pts1[indices]
        sample_pts2 = pts2[indices]
        
        # Step 2: Compute homography from these 4 points
        H = compute_homography_dlt(sample_pts1, sample_pts2)
        
        if H is None:
            continue
        
        # Step 3: Compute reprojection error for ALL points
        errors = compute_reprojection_error(H, pts1, pts2)
        
        # Step 4: Count inliers (error < threshold)
        inlier_mask = errors < threshold
        inlier_count = np.sum(inlier_mask)
        
        # Step 5: Update best if this is better
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask
            best_H = H
            
            # Early termination if we have enough inliers
            inlier_ratio = inlier_count / n
            if inlier_ratio > 0.9:
                print(f"  Early stop at iteration {iteration}: {inlier_count}/{n} inliers ({inlier_ratio*100:.1f}%)")
                break
    
    if best_inlier_count < min_inliers:
        print(f"  Failed: only {best_inlier_count} inliers (need {min_inliers})")
        return None, None
    
    print(f"  Best: {best_inlier_count}/{n} inliers ({best_inlier_count/n*100:.1f}%)")
    
    # Step 6: Recompute homography using ALL inliers
    inlier_pts1 = pts1[best_inlier_mask]
    inlier_pts2 = pts2[best_inlier_mask]
    
    final_H = compute_homography_dlt(inlier_pts1, inlier_pts2)
    
    if final_H is not None:
        # Recompute inlier mask with refined homography
        errors = compute_reprojection_error(final_H, pts1, pts2)
        best_inlier_mask = errors < threshold
        print(f"  After refinement: {np.sum(best_inlier_mask)}/{n} inliers")
    
    return final_H, best_inlier_mask


def calculate_iterations_needed(inlier_ratio, sample_size=4, confidence=0.99):
    """
    Calculate number of RANSAC iterations needed.
    
    Formula: N = log(1 - confidence) / log(1 - inlier_ratio^sample_size)
    
    Args:
        inlier_ratio: expected ratio of inliers (e.g., 0.5 for 50%)
        sample_size: number of points per sample (4 for homography)
        confidence: desired probability of success (e.g., 0.99)
    
    Returns:
        n_iterations: number of iterations needed
    """
    if inlier_ratio <= 0 or inlier_ratio >= 1:
        return 1000
    
    p_all_inliers = inlier_ratio ** sample_size
    
    if p_all_inliers >= 1:
        return 1
    
    n = np.log(1 - confidence) / np.log(1 - p_all_inliers)
    
    return int(np.ceil(n))


# =============================================================================
# Helper: Convert matches to point arrays
# =============================================================================

def matches_to_points(keypoints1, keypoints2, matches):
    """
    Convert matches to point arrays for RANSAC.
    
    Args:
        keypoints1: list of (x, y) from image 1
        keypoints2: list of (x, y) from image 2
        matches: list of (idx1, idx2, distance)
    
    Returns:
        pts1: Nx2 array of points from image 1
        pts2: Nx2 array of points from image 2
    """
    n = len(matches)
    pts1 = np.zeros((n, 2))
    pts2 = np.zeros((n, 2))
    
    for i in range(n):
        idx1 = matches[i][0]
        idx2 = matches[i][1]
        
        pts1[i, 0] = keypoints1[idx1][0]
        pts1[i, 1] = keypoints1[idx1][1]
        pts2[i, 0] = keypoints2[idx2][0]
        pts2[i, 1] = keypoints2[idx2][1]
    
    return pts1, pts2


def filter_matches_by_mask(matches, inlier_mask):
    """
    Filter matches using inlier mask from RANSAC.
    
    Args:
        matches: list of (idx1, idx2, distance)
        inlier_mask: boolean array
    
    Returns:
        inlier_matches: filtered list
    """
    inlier_matches = []
    
    for i in range(len(matches)):
        if inlier_mask[i]:
            inlier_matches.append(matches[i])
    
    return inlier_matches


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("RANSAC for Geometric Verification")
    print("=" * 50)
    
    # Load image
    img1 = load_gray("data/sample.jpg")
    if img1 is None:
        print("Could not load image!")
        exit()
    
    # Create transformed image (simulate viewpoint change)
    # Apply a simple shift + small rotation approximation
    print("\nCreating test image pair with known transformation...")
    shift_x = 30
    shift_y = 10
    
    img2 = np.zeros_like(img1)
    h, w = img1.shape
    
    # Shift image
    for i in range(h):
        for j in range(w):
            src_i = i + shift_y
            src_j = j + shift_x
            if 0 <= src_i < h and 0 <= src_j < w:
                img2[i, j] = img1[src_i, src_j]
    
    print(f"Applied shift: ({shift_x}, {shift_y}) pixels")
    
    # Detect keypoints
    print("\nDetecting Harris corners...")
    kp1 = harris_corners(img1, threshold=0.01, nms_radius=15)
    kp2 = harris_corners(img2, threshold=0.01, nms_radius=15)
    print(f"Image 1: {len(kp1)} keypoints")
    print(f"Image 2: {len(kp2)} keypoints")
    
    # Compute descriptors
    print("\nComputing BRIEF descriptors...")
    valid_kp1, desc1 = compute_brief_descriptors(img1, kp1, n_bits=256)
    valid_kp2, desc2 = compute_brief_descriptors(img2, kp2, n_bits=256)
    print(f"Image 1: {len(desc1)} descriptors")
    print(f"Image 2: {len(desc2)} descriptors")
    
    # Match descriptors
    print("\nMatching with ratio test + cross-check...")
    matches = robust_match(desc1, desc2, hamming_distance, ratio_thresh=0.8)
    print(f"Found {len(matches)} matches")
    
    if len(matches) < 10:
        print("Not enough matches!")
        exit()
    
    # Convert matches to point arrays
    pts1, pts2 = matches_to_points(valid_kp1, valid_kp2, matches)
    
    # Show matches before RANSAC
    print("\nShowing matches BEFORE RANSAC...")
    vis_matches_before = [(m[0], m[1]) for m in matches]
    img_before = draw_matches(img1, img2, valid_kp1, valid_kp2, vis_matches_before)
    show(img_before, f"Before RANSAC ({len(matches)} matches)")
    
    # Calculate recommended iterations
    assumed_inlier_ratio = 0.5
    recommended_iters = calculate_iterations_needed(assumed_inlier_ratio)
    print(f"\nAssuming {assumed_inlier_ratio*100:.0f}% inliers, need ~{recommended_iters} iterations")
    
    # Run RANSAC
    print("\nRunning RANSAC...")
    H, inlier_mask = ransac_homography(pts1, pts2, n_iterations=1000, threshold=5.0)
    
    if H is not None:
        print("\nEstimated Homography:")
        print(H)
        
        # The translation should be close to our applied shift
        print(f"\nEstimated translation: ({H[0,2]:.1f}, {H[1,2]:.1f})")
        print(f"Actual translation: ({shift_x}, {shift_y})")
        
        # Filter matches
        inlier_matches = filter_matches_by_mask(matches, inlier_mask)
        
        # Show matches after RANSAC
        print(f"\nShowing matches AFTER RANSAC...")
        vis_matches_after = [(m[0], m[1]) for m in inlier_matches]
        img_after = draw_matches(img1, img2, valid_kp1, valid_kp2, vis_matches_after)
        show(img_after, f"After RANSAC ({len(inlier_matches)} inliers)")
        
        # Statistics
        outliers = len(matches) - len(inlier_matches)
        print(f"\nSummary:")
        print(f"  Total matches: {len(matches)}")
        print(f"  Inliers: {len(inlier_matches)} ({len(inlier_matches)/len(matches)*100:.1f}%)")
        print(f"  Outliers rejected: {outliers} ({outliers/len(matches)*100:.1f}%)")
    else:
        print("RANSAC failed to find a good model")
    
    print("\nDone!")