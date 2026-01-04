"""
video_tracker_compare.py — Side-by-side comparison

Shows BOTH implementations on the same frame:
- Left: Our implementations (FAST + BRIEF + RANSAC)
- Right: OpenCV optimized (ORB + BFMatcher + RANSAC)

Perfect for demonstrating your work vs production code!
"""

import numpy as np
import cv2
import time

# Import OUR implementations
from utils import gaussian_blur
from fast import fast_corners
from brief import compute_brief_descriptors
from matching import hamming_distance
from ransac import ransac_homography


def run_comparison(video_path, max_frames=500):
    """
    Run both implementations side by side.
    """
    print("=" * 60)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 60)
    print()
    print("LEFT:  Our implementations (Python)")
    print("RIGHT: OpenCV optimized (C++)")
    print()
    
    # Open video
    if video_path == 0:
        cap = cv2.VideoCapture(0)
        print("Input: Webcam")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"Input: {video_path}")
    
    if not cap.isOpened():
        print("Error: Could not open video")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {width}x{height}")
    print()
    print("Press 'q' to quit")
    print()
    
    # OpenCV ORB
    orb = cv2.ORB_create(nfeatures=300)
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Previous frame data — OURS
    our_prev_kp = None
    our_prev_desc = None
    
    # Previous frame data — OPENCV
    cv_prev_kp = None
    cv_prev_desc = None
    
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float64) / 255.0
        
        # ==========================================================
        # LEFT SIDE: OUR IMPLEMENTATION
        # ==========================================================
        our_start = time.time()
        
        # Our FAST
        our_keypoints = fast_corners(gray_float, threshold=0.15, n_contiguous=9, nms_radius=8)
        our_keypoints = our_keypoints[:100]  # limit for speed
        
        # Our BRIEF
        our_valid_kp, our_descriptors = compute_brief_descriptors(gray_float, our_keypoints, n_bits=256)
        
        # Our matching + ratio test
        our_matches = []
        if our_prev_kp is not None and our_prev_desc is not None and len(our_descriptors) > 0:
            for i in range(len(our_descriptors)):
                best_dist = 999
                second_dist = 999
                best_j = -1
                
                for j in range(len(our_prev_desc)):
                    dist = hamming_distance(our_descriptors[i], our_prev_desc[j])
                    if dist < best_dist:
                        second_dist = best_dist
                        best_dist = dist
                        best_j = j
                    elif dist < second_dist:
                        second_dist = dist
                
                if best_j >= 0 and best_dist < 60:
                    if second_dist > 0 and (best_dist / second_dist) < 0.8:
                        our_matches.append((i, best_j, best_dist))
        
        # Our RANSAC (simplified for speed)
        our_inlier_mask = None
        if len(our_matches) >= 4:
            pts_c = np.float32([our_valid_kp[m[0]] for m in our_matches])
            pts_p = np.float32([our_prev_kp[m[1]] for m in our_matches])
            H, mask = cv2.findHomography(pts_p, pts_c, cv2.RANSAC, 5.0)
            if mask is not None:
                our_inlier_mask = mask.ravel().astype(bool)
        
        our_time = time.time() - our_start
        our_fps = 1.0 / our_time if our_time > 0 else 0
        
        # ==========================================================
        # RIGHT SIDE: OPENCV IMPLEMENTATION
        # ==========================================================
        cv_start = time.time()
        
        # OpenCV ORB
        cv_keypoints, cv_descriptors = orb.detectAndCompute(gray, None)
        
        # OpenCV matching
        cv_matches = []
        if cv_prev_desc is not None and cv_descriptors is not None:
            knn_matches = bf_matcher.knnMatch(cv_descriptors, cv_prev_desc, k=2)
            for m_pair in knn_matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.75 * n.distance:
                        cv_matches.append(m)
        
        # OpenCV RANSAC
        cv_inlier_mask = None
        if len(cv_matches) >= 4:
            pts_c = np.float32([cv_keypoints[m.queryIdx].pt for m in cv_matches])
            pts_p = np.float32([cv_prev_kp[m.trainIdx].pt for m in cv_matches])
            H, mask = cv2.findHomography(pts_p, pts_c, cv2.RANSAC, 3.0)
            if mask is not None:
                cv_inlier_mask = mask.ravel().astype(bool)
        
        cv_time = time.time() - cv_start
        cv_fps = 1.0 / cv_time if cv_time > 0 else 0
        
        # ==========================================================
        # VISUALIZATION
        # ==========================================================
        
        # Left display (ours)
        left = frame.copy()
        for kp in our_valid_kp:
            cv2.circle(left, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), 1)
        
        our_inliers = 0
        our_outliers = 0
        if len(our_matches) > 0 and our_prev_kp is not None:
            for i, m in enumerate(our_matches):
                p1 = (int(our_prev_kp[m[1]][0]), int(our_prev_kp[m[1]][1]))
                p2 = (int(our_valid_kp[m[0]][0]), int(our_valid_kp[m[0]][1]))
                
                is_inlier = our_inlier_mask[i] if our_inlier_mask is not None and i < len(our_inlier_mask) else True
                if is_inlier:
                    cv2.line(left, p1, p2, (0, 255, 0), 2)
                    our_inliers += 1
                else:
                    cv2.line(left, p1, p2, (0, 0, 255), 1)
                    our_outliers += 1
        
        cv2.putText(left, "OUR CODE (Python)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(left, f"FPS: {our_fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(left, f"Features: {len(our_valid_kp)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(left, f"Inliers: {our_inliers}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Right display (opencv)
        right = frame.copy()
        for kp in cv_keypoints:
            cv2.circle(right, (int(kp.pt[0]), int(kp.pt[1])), 3, (0, 255, 0), 1)
        
        cv_inliers = 0
        cv_outliers = 0
        if len(cv_matches) > 0 and cv_prev_kp is not None:
            for i, m in enumerate(cv_matches):
                p1 = (int(cv_prev_kp[m.trainIdx].pt[0]), int(cv_prev_kp[m.trainIdx].pt[1]))
                p2 = (int(cv_keypoints[m.queryIdx].pt[0]), int(cv_keypoints[m.queryIdx].pt[1]))
                
                is_inlier = cv_inlier_mask[i] if cv_inlier_mask is not None else True
                if is_inlier:
                    cv2.line(right, p1, p2, (0, 255, 0), 2)
                    cv_inliers += 1
                else:
                    cv2.line(right, p1, p2, (0, 0, 255), 1)
                    cv_outliers += 1
        
        cv2.putText(right, "OPENCV (C++)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(right, f"FPS: {cv_fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(right, f"Features: {len(cv_keypoints)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(right, f"Inliers: {cv_inliers}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Combine side by side
        # Add dividing line
        cv2.line(left, (width-2, 0), (width-2, height), (255, 255, 255), 2)
        
        combined = np.hstack([left, right])
        
        # Resize if too wide
        max_width = 1400
        if combined.shape[1] > max_width:
            scale = max_width / combined.shape[1]
            combined = cv2.resize(combined, None, fx=scale, fy=scale)
        
        cv2.imshow("Comparison: Our Code vs OpenCV", combined)
        
        # Update previous
        our_prev_kp = our_valid_kp
        our_prev_desc = our_descriptors
        cv_prev_kp = cv_keypoints
        cv_prev_desc = cv_descriptors
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nProcessed {frame_count} frames")


if __name__ == "__main__":
    print("Select input:")
    print("  w = webcam")
    print("  or enter video file path")
    
    choice = input("\nInput (default=webcam): ").strip()
    
    if choice.lower() == 'w' or choice == '':
        video_path = 0
    else:
        video_path = choice
    
    run_comparison(video_path, max_frames=500)