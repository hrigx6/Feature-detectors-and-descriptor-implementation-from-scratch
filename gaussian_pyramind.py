"""
gaussian_pyramid.py — Gaussian Pyramid and Difference of Gaussians (DoG)

"""

import numpy as np
from utils import load_gray, show, gaussian_blur


def build_gaussian_pyramid(img, num_octaves=4, num_scales=5, sigma_base=1.6):
    # After num_scales levels, sigma doubles, so k^(num_scales-1) = 2
    k = 2 ** (1.0 / (num_scales - 1))
    
    # Compute sigma values for one octave
    sigmas = []
    for i in range(num_scales):
        sigma = sigma_base * (k ** i)
        sigmas.append(sigma)
    
    print(f"Scale factor k = {k:.3f}")
    print(f"Sigmas per octave: {[f'{s:.2f}' for s in sigmas]}")
    
    # Build pyramid
    pyramid = []
    current_img = img.copy()
    
    for octave in range(num_octaves):
        print(f"Building octave {octave}, image size: {current_img.shape}")
        
        octave_images = []
        
        for scale in range(num_scales):
            sigma = sigmas[scale]
            
            # Blur the image
            blurred = gaussian_blur(current_img, sigma)
            octave_images.append(blurred)
        
        pyramid.append(octave_images)
        
        # Downsample for next octave (take every 2nd pixel)
        # Use the image at scale index (num_scales-1)//2 as base for next octave
        height = current_img.shape[0]
        width = current_img.shape[1]
        
        new_height = height // 2
        new_width = width // 2
        
        if new_height < 8 or new_width < 8:
            print(f"Stopping at octave {octave} — image too small")
            break
        
        # Simple downsampling: take every other pixel
        downsampled = np.zeros((new_height, new_width))
        for i in range(new_height):
            for j in range(new_width):
                downsampled[i, j] = current_img[i * 2, j * 2]
        
        current_img = downsampled
    
    return pyramid, sigmas


def build_dog_pyramid(gaussian_pyramid):
    
    dog_pyramid = []
    
    for octave in range(len(gaussian_pyramid)):
        octave_dogs = []
        octave_gaussians = gaussian_pyramid[octave]
        num_scales = len(octave_gaussians)
        
        # DoG = difference between adjacent scales
        for i in range(num_scales - 1):
            dog = octave_gaussians[i + 1] - octave_gaussians[i]
            octave_dogs.append(dog)
        
        dog_pyramid.append(octave_dogs)
        print(f"Octave {octave}: {len(octave_dogs)} DoG images")
    
    return dog_pyramid


def find_dog_extrema(dog_pyramid, threshold=0.03):
    
    keypoints = []
    
    for octave in range(len(dog_pyramid)):
        dogs = dog_pyramid[octave]
        num_levels = len(dogs)
        
        # We need level-1, level, level+1 — so skip first and last
        for level in range(1, num_levels - 1):
            dog_below = dogs[level - 1]
            dog_current = dogs[level]
            dog_above = dogs[level + 1]
            
            height = dog_current.shape[0]
            width = dog_current.shape[1]
            
            # Check each pixel (skip border)
            for row in range(1, height - 1):
                for col in range(1, width - 1):
                    
                    center = dog_current[row, col]
                    
                    # Skip weak responses
                    if abs(center) < threshold:
                        continue
                    
                    # Gather all 26 neighbors
                    neighbors = []
                    
                    # 8 neighbors in current level
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue  # skip center itself
                            neighbors.append(dog_current[row + dr, col + dc])
                    
                    # 9 neighbors in level below
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            neighbors.append(dog_below[row + dr, col + dc])
                    
                    # 9 neighbors in level above
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            neighbors.append(dog_above[row + dr, col + dc])
                    
                    # Check if center is extremum (max or min)
                    is_max = True
                    is_min = True
                    
                    for n in neighbors:
                        if n >= center:
                            is_max = False
                        if n <= center:
                            is_min = False
                    
                    if is_max or is_min:
                        keypoints.append((octave, level, row, col, center))
    
    return keypoints


def keypoints_to_image_coords(keypoints, img_shape):
    
    coords = []
    
    for kp in keypoints:
        octave = kp[0]
        scale = kp[1]
        row = kp[2]
        col = kp[3]
        
        # Scale up coordinates based on octave
        # Octave 0 = original size, octave 1 = half size, etc.
        scale_factor = 2 ** octave
        
        x = int(col * scale_factor)
        y = int(row * scale_factor)
        
        # Make sure we're within image bounds
        if x < img_shape[1] and y < img_shape[0]:
            coords.append((x, y))
    
    return coords


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Gaussian Pyramid and DoG Keypoints")
    print("=" * 50)
    
    # Load image
    img = load_gray("data/sample.jpg")
    if img is None:
        print("Could not load image!")
        exit()
    
    print(f"Image shape: {img.shape}")
    print()
    
    # Build Gaussian pyramid
    print("Building Gaussian pyramid...")
    pyramid, sigmas = build_gaussian_pyramid(img, num_octaves=4, num_scales=5)
    print()
    
    # Show some pyramid levels
    print("Showing pyramid samples...")
    show(pyramid[0][0], "Octave 0, Scale 0 (least blur)")
    show(pyramid[0][-1], "Octave 0, Scale 4 (most blur in octave)")
    if len(pyramid) > 1:
        show(pyramid[1][0], "Octave 1, Scale 0 (downsampled)")
    
    # Build DoG pyramid
    print("\nBuilding DoG pyramid...")
    dog_pyramid = build_dog_pyramid(pyramid)
    print()
    
    # Show a DoG image
    print("Showing DoG sample...")
    dog_sample = dog_pyramid[0][1]
    # Normalize for display
    dog_display = dog_sample - dog_sample.min()
    if dog_display.max() > 0:
        dog_display = dog_display / dog_display.max()
    show(dog_display, "DoG (Octave 0, Level 1)")
    
    # Find keypoints
    print("Finding DoG extrema...")
    keypoints = find_dog_extrema(dog_pyramid, threshold=0.03)
    print(f"Found {len(keypoints)} raw keypoints")
    
    # Convert to image coordinates
    coords = keypoints_to_image_coords(keypoints, img.shape)
    print(f"Converted to {len(coords)} image coordinates")
    
    # Draw keypoints
    from utils import draw_keypoints
    img_with_kp = draw_keypoints(img, coords, radius=3, color=(0, 255, 0))
    show(img_with_kp, f"DoG Keypoints ({len(coords)} detected)")
    
    print("Done!")