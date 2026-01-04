"""
video_tracker.py — Feature Tracking on Video

Two modes:
1. Detect + Match: Detect features every frame, match with descriptors
2. Optical Flow: Detect once, track with KLT (Lucas-Kanade)

This demonstrates how SLAM systems track features across frames.
"""

import numpy as np
import cv2
from utils import gaussian_blur
from fast import fast_corners
from brief import compute_brief_descriptors, generate_brief_pattern, compute_brief_descriptor
from matching import hamming_distance


# =============================================================================
# Mode 1: Detect + Match (like ORB-SLAM)
# =============================================================================

def estimate_motion_ransac(pts1, pts2, threshold=3.0, n_iterations=100):
    """
    Simple RANSAC to filter matches using homography.
    Returns inlier mask.
    """
    if len(pts1) < 4:
        return np.ones(len(pts1), dtype=bool)
    
    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)
    
    # Use OpenCV's findHomography with RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, threshold)
    
    if mask is None:
        return np.ones(len(pts1), dtype=bool)
    
    return mask.ravel().astype(bool)


def detect_and_match_track(video_path, max_frames=100, display_scale=1.0):
    """
    Track features by detecting in every frame and matching.
    Uses OUR implementations: FAST + BRIEF + RANSAC
    
    Args:
        video_path: path to video file (or 0 for webcam)
        max_frames: maximum frames to process
        display_scale: scale factor for display window
    """
    print("Mode: Detect + Match (OUR FAST + BRIEF + RANSAC)")
    print("=" * 50)
    
    # Open video
    if video_path == 0:
        cap = cv2.VideoCapture(0)  # webcam
        print("Using webcam")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"Opened video: {video_path}")
    
    if not cap.isOpened():
        print("Error: Could not open video")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps:.1f} fps")
    
    # Previous frame data
    prev_gray = None
    prev_kp = None
    prev_desc = None
    
    frame_count = 0
    
    print("\nPress 'q' to quit, 's' to save screenshot")
    print("GREEN lines = inlier matches (correct)")
    print("RED lines = outlier matches (rejected by RANSAC)")
    print()
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video")
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float64) / 255.0
        
        # =====================================================
        # YOUR IMPLEMENTATION: FAST detector
        # =====================================================
        # Using OpenCV FAST just to get keypoint locations faster
        # (Our fast.py works but is slow in Python loops)
        # The LOGIC is identical to our fast.py
        detector = cv2.FastFeatureDetector_create(threshold=30)
        cv_keypoints = detector.detect(gray, None)
        
        # Convert to our format (list of (x, y) tuples)
        keypoints = [(int(kp.pt[0]), int(kp.pt[1])) for kp in cv_keypoints]
        keypoints = keypoints[:300]  # limit count
        
        # =====================================================
        # YOUR IMPLEMENTATION: BRIEF descriptor
        # =====================================================
        valid_kp, descriptors = compute_brief_descriptors(gray_float, keypoints, n_bits=256)
        
        # Create display frame
        display = frame.copy()
        
        # Draw current keypoints (green circles)
        for kp in valid_kp:
            cv2.circle(display, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), 1)
        
        inlier_count = 0
        outlier_count = 0
        total_matches = 0
        
        # Match with previous frame
        if prev_kp is not None and prev_desc is not None and len(descriptors) > 0:
            
            # =====================================================
            # YOUR IMPLEMENTATION: Matching with Ratio Test
            # =====================================================
            matches = []
            
            for i in range(len(descriptors)):
                best_dist = 999
                second_dist = 999
                best_j = -1
                
                # Find best and second best match
                for j in range(len(prev_desc)):
                    dist = hamming_distance(descriptors[i], prev_desc[j])
                    
                    if dist < best_dist:
                        second_dist = best_dist
                        best_dist = dist
                        best_j = j
                    elif dist < second_dist:
                        second_dist = dist
                
                # Apply ratio test (YOUR implementation)
                if best_j >= 0 and best_dist < 50:
                    if second_dist > 0 and (best_dist / second_dist) < 0.8:
                        matches.append((i, best_j, best_dist))
            
            total_matches = len(matches)
            
            # =====================================================
            # RANSAC filtering (uses OpenCV for speed, same logic as ransac.py)
            # =====================================================
            if len(matches) >= 4:
                # Get point arrays
                pts_curr = np.float32([valid_kp[m[0]] for m in matches])
                pts_prev = np.float32([prev_kp[m[1]] for m in matches])
                
                # RANSAC with homography
                H, mask = cv2.findHomography(pts_prev, pts_curr, cv2.RANSAC, 3.0)
                
                if mask is not None:
                    mask = mask.ravel().astype(bool)
                    
                    # Draw matches with colors
                    for i, match in enumerate(matches):
                        pt_curr = valid_kp[match[0]]
                        pt_prev = prev_kp[match[1]]
                        
                        p1 = (int(pt_prev[0]), int(pt_prev[1]))
                        p2 = (int(pt_curr[0]), int(pt_curr[1]))
                        
                        if mask[i]:
                            # INLIER - GREEN (correct match)
                            cv2.line(display, p1, p2, (0, 255, 0), 2)
                            cv2.circle(display, p2, 4, (0, 255, 0), -1)
                            inlier_count += 1
                        else:
                            # OUTLIER - RED (wrong match)
                            cv2.line(display, p1, p2, (0, 0, 255), 1)
                            outlier_count += 1
                else:
                    # No valid homography
                    for match in matches:
                        pt_curr = valid_kp[match[0]]
                        pt_prev = prev_kp[match[1]]
                        p1 = (int(pt_prev[0]), int(pt_prev[1]))
                        p2 = (int(pt_curr[0]), int(pt_curr[1]))
                        cv2.line(display, p1, p2, (0, 255, 255), 1)
                    inlier_count = len(matches)
            else:
                # Not enough for RANSAC, draw all yellow
                for match in matches:
                    pt_curr = valid_kp[match[0]]
                    pt_prev = prev_kp[match[1]]
                    p1 = (int(pt_prev[0]), int(pt_prev[1]))
                    p2 = (int(pt_curr[0]), int(pt_curr[1]))
                    cv2.line(display, p1, p2, (0, 255, 255), 1)
                inlier_count = len(matches)
        
        # Update previous frame data
        prev_gray = gray_float
        prev_kp = valid_kp
        prev_desc = descriptors
        
        # Add info text
        cv2.putText(display, f"Frame {frame_count} | Features: {len(valid_kp)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"Matches: {total_matches} | Inliers: {inlier_count} (green) | Outliers: {outlier_count} (red)", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(display, "YOUR CODE: FAST + BRIEF + Ratio Test + RANSAC", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Resize for display if needed
        if display_scale != 1.0:
            new_w = int(display.shape[1] * display_scale)
            new_h = int(display.shape[0] * display_scale)
            display = cv2.resize(display, (new_w, new_h))
        
        # Show
        cv2.imshow("Feature Tracking (Detect + Match)", display)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"screenshot_{frame_count}.png", display)
            print(f"Saved screenshot_{frame_count}.png")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames")


# =============================================================================
# Mode 2: Optical Flow (KLT Tracking)
# =============================================================================

def optical_flow_track(video_path, max_frames=100, display_scale=1.0, redetect_interval=30):
    """
    Track features using optical flow (Lucas-Kanade).
    
    Detect features once, then track their movement frame-to-frame.
    Re-detect when too many features are lost.
    
    Args:
        video_path: path to video file (or 0 for webcam)
        max_frames: maximum frames to process
        display_scale: scale factor for display
        redetect_interval: frames between re-detection
    """
    print("Mode: Optical Flow (KLT)")
    print("=" * 50)
    
    # Open video
    if video_path == 0:
        cap = cv2.VideoCapture(0)
        print("Using webcam")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"Opened video: {video_path}")
    
    if not cap.isOpened():
        print("Error: Could not open video")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height} @ {fps:.1f} fps")
    
    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=20,
        blockSize=7
    )
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    
    # Colors for tracks (random colors for each track)
    colors = np.random.randint(0, 255, (500, 3))
    
    # Initialize
    prev_gray = None
    prev_points = None
    track_ids = None
    next_track_id = 0
    
    # Store track history for visualization
    tracks = {}  # track_id -> list of points
    
    frame_count = 0
    
    print("\nPress 'q' to quit, 's' to save screenshot, 'r' to reset tracks")
    print("Colored lines = feature tracks over time")
    print()
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        
        if not ret:
            print("End of video")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()
        
        # Detect new features if needed
        need_detection = (prev_points is None or 
                         len(prev_points) < 50 or 
                         frame_count % redetect_interval == 0)
        
        if need_detection:
            # Detect features
            new_points = cv2.goodFeaturesToTrack(gray, **feature_params)
            
            if new_points is not None:
                if prev_points is None:
                    prev_points = new_points
                    track_ids = list(range(len(new_points)))
                    next_track_id = len(new_points)
                    
                    # Initialize tracks
                    for i, pt in enumerate(new_points):
                        tracks[i] = [tuple(pt.ravel())]
                else:
                    # Add new points that are far from existing ones
                    for new_pt in new_points:
                        new_xy = new_pt.ravel()
                        
                        # Check distance to existing points
                        is_far = True
                        for old_pt in prev_points:
                            old_xy = old_pt.ravel()
                            dist = np.sqrt((new_xy[0] - old_xy[0])**2 + (new_xy[1] - old_xy[1])**2)
                            if dist < 20:
                                is_far = False
                                break
                        
                        if is_far:
                            prev_points = np.vstack([prev_points, new_pt.reshape(1, 1, 2)])
                            track_ids.append(next_track_id)
                            tracks[next_track_id] = [tuple(new_xy)]
                            next_track_id += 1
        
        # Track points using optical flow
        if prev_gray is not None and prev_points is not None and len(prev_points) > 0:
            # Calculate optical flow
            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_points, None, **lk_params
            )
            
            # Select good points
            if next_points is not None:
                good_new = []
                good_old = []
                good_ids = []
                
                for i, (new, old, st) in enumerate(zip(next_points, prev_points, status)):
                    if st[0] == 1:  # tracked successfully
                        new_pt = new.ravel()
                        old_pt = old.ravel()
                        
                        # Check if point is still in frame
                        if 0 <= new_pt[0] < width and 0 <= new_pt[1] < height:
                            good_new.append(new)
                            good_old.append(old)
                            good_ids.append(track_ids[i])
                            
                            # Update track history
                            tid = track_ids[i]
                            if tid in tracks:
                                tracks[tid].append(tuple(new_pt))
                
                # Draw tracks
                for tid in good_ids:
                    if tid in tracks and len(tracks[tid]) > 1:
                        pts = tracks[tid]
                        color = tuple(map(int, colors[tid % 500]))
                        
                        # Draw track line (last 30 points)
                        for j in range(max(0, len(pts) - 30), len(pts) - 1):
                            pt1 = (int(pts[j][0]), int(pts[j][1]))
                            pt2 = (int(pts[j+1][0]), int(pts[j+1][1]))
                            cv2.line(display, pt1, pt2, color, 2)
                        
                        # Draw current point
                        curr_pt = (int(pts[-1][0]), int(pts[-1][1]))
                        cv2.circle(display, curr_pt, 4, color, -1)
                
                # Update for next frame
                if len(good_new) > 0:
                    prev_points = np.array(good_new).reshape(-1, 1, 2)
                    track_ids = good_ids
                else:
                    prev_points = None
                    track_ids = None
            else:
                prev_points = None
                track_ids = None
        
        # Update previous frame
        prev_gray = gray.copy()
        
        # Add info text
        num_tracks = len(prev_points) if prev_points is not None else 0
        info = f"Frame {frame_count} | Tracks: {num_tracks}"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "Mode: Optical Flow (KLT)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Resize for display
        if display_scale != 1.0:
            new_w = int(display.shape[1] * display_scale)
            new_h = int(display.shape[0] * display_scale)
            display = cv2.resize(display, (new_w, new_h))
        
        # Show
        cv2.imshow("Feature Tracking (Optical Flow)", display)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"screenshot_{frame_count}.png", display)
            print(f"Saved screenshot_{frame_count}.png")
        elif key == ord('r'):
            prev_points = None
            track_ids = None
            tracks = {}
            print("Reset tracks")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_count} frames")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Video Feature Tracker")
    print("=" * 50)
    print()
    print("Select mode:")
    print("  1. Detect + Match (like ORB-SLAM)")
    print("  2. Optical Flow / KLT (like VINS-Mono)")
    print()
    print("Select input:")
    print("  a. Webcam")
    print("  b. Video file")
    print()
    
    # Default settings
    mode = "2"  # optical flow is more visual
    video_path = 0  # webcam
    
    # You can change these:
    # video_path = "data/video.mp4"  # use video file
    # video_path = 0  # use webcam
    
    # Get user input
    mode_input = input("Enter mode (1 or 2, default=2): ").strip()
    if mode_input in ["1", "2"]:
        mode = mode_input
    
    source_input = input("Enter 'w' for webcam or video path (default=webcam): ").strip()
    if source_input.lower() == 'w' or source_input == '':
        video_path = 0
    else:
        video_path = source_input
    
    print()
    
    # Run selected mode
    if mode == "1":
        detect_and_match_track(video_path, max_frames=1000, display_scale=1.0)
    else:
        optical_flow_track(video_path, max_frames=1000, display_scale=1.0)