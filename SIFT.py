import numpy as np
from utils import gaussian_blur, compute_gradients
from utils import load_gray, show, draw_keypoints

def build_sift_gaussian_pyramid(img, num_octaves=4, s=3, sigma0=1.6):

    k = 2 ** (1.0 / s)
    levels = s + 3
    sigmas = [sigma0 * (k ** i) for i in range(levels)]

    gauss_pyr = []
    current = img.copy()

    for o in range(num_octaves):
        octave_imgs = []
        for i in range(levels):
            octave_imgs.append(gaussian_blur(current, sigmas[i]))
        gauss_pyr.append(octave_imgs)

        # downsample base image for next octave
        base = octave_imgs[s]            # SIFT uses a blurred level before downsampling
        h, w = base.shape
        if h // 2 < 8 or w // 2 < 8:
            break
        current = base[::2, ::2]         # simple decimation

    return gauss_pyr, sigmas


def build_dog_pyramid(gauss_pyr):
    dog_pyr = []
    for octave_imgs in gauss_pyr:
        dogs = []
        for i in range(len(octave_imgs) - 1):
            dogs.append(octave_imgs[i + 1] - octave_imgs[i])
        dog_pyr.append(dogs)
    return dog_pyr


def is_edge_like(dog, y, x, r=10):
    """
    SIFT edge rejection using Hessian ratio test.
    """
    Dxx = dog[y, x+1] - 2*dog[y, x] + dog[y, x-1]
    Dyy = dog[y+1, x] - 2*dog[y, x] + dog[y-1, x]
    Dxy = (dog[y+1, x+1] - dog[y+1, x-1] - dog[y-1, x+1] + dog[y-1, x-1]) / 4.0

    det = Dxx * Dyy - Dxy * Dxy
    if det <= 1e-12:
        return True

    trace = Dxx + Dyy
    ratio = (trace * trace) / det
    thresh = ((r + 1) ** 2) / r
    return ratio > thresh


def find_dog_keypoints(dog_pyr, sigmas, contrast_thresh=0.03, edge_r=10):
   
    keypoints = []

    for o, dogs in enumerate(dog_pyr):
        num_levels = len(dogs)

        # need below/current/above -> skip first and last
        for level in range(1, num_levels - 1):
            dog_below = dogs[level - 1]
            dog_cur = dogs[level]
            dog_above = dogs[level + 1]

            h, w = dog_cur.shape

            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    v = dog_cur[y, x]
                    if abs(v) < contrast_thresh:
                        continue

                    # gather 26 neighbors and check extremum
                    is_max = True
                    is_min = True

                    # current level 8 neighbors
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            n = dog_cur[y + dy, x + dx]
                            if n >= v:
                                is_max = False
                            if n <= v:
                                is_min = False

                    # below + above 9 each
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            n1 = dog_below[y + dy, x + dx]
                            n2 = dog_above[y + dy, x + dx]
                            if n1 >= v or n2 >= v:
                                is_max = False
                            if n1 <= v or n2 <= v:
                                is_min = False

                    if not (is_max or is_min):
                        continue

                    # edge rejection
                    if is_edge_like(dog_cur, y, x, r=edge_r):
                        continue

                    sigma = sigmas[level]  # sigma at this level in the octave
                    keypoints.append((o, level, y, x, sigma))

    return keypoints


def assign_orientation(gauss_img, y, x, sigma, num_bins=36):
   
    Ix, Iy = compute_gradients(gauss_img)
    mag = np.sqrt(Ix * Ix + Iy * Iy)
    ang = np.arctan2(Iy, Ix)  # [-pi, pi]

    radius = int(3 * sigma)
    h, w = gauss_img.shape
    if y - radius < 1 or y + radius >= h - 1 or x - radius < 1 or x + radius >= w - 1:
        return None

    hist = np.zeros(num_bins, dtype=np.float64)
    weight_sigma = 1.5 * sigma
    denom = 2 * (weight_sigma * weight_sigma)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            yy = y + dy
            xx = x + dx

            wgt = np.exp(-(dx*dx + dy*dy) / denom)
            m = mag[yy, xx] * wgt

            a = ang[yy, xx]
            if a < 0:
                a += 2 * np.pi

            b = int((a / (2*np.pi)) * num_bins) % num_bins
            hist[b] += m

    best = int(np.argmax(hist))
    theta = (best / num_bins) * 2 * np.pi
    return theta


def sift_descriptor(gauss_img, y, x, sigma, theta, grid=4, bins=8):
    
    Ix, Iy = compute_gradients(gauss_img)
    mag = np.sqrt(Ix * Ix + Iy * Iy)
    ang = np.arctan2(Iy, Ix)

    win = int(8 * sigma)  # half window
    h, w = gauss_img.shape
    if y - win < 1 or y + win >= h - 1 or x - win < 1 or x + win >= w - 1:
        return None

    c = np.cos(theta)
    s = np.sin(theta)

    desc = np.zeros((grid, grid, bins), dtype=np.float64)

    size = 2 * win + 1
    cell_size = size / grid

    weight_sigma = 8 * sigma
    denom = 2 * (weight_sigma * weight_sigma)

    for dy in range(-win, win + 1):
        for dx in range(-win, win + 1):
            yy = y + dy
            xx = x + dx

            # rotate coordinates into keypoint frame
            rx =  c*dx + s*dy
            ry = -s*dx + c*dy

            cx = (rx + win) / cell_size
            cy = (ry + win) / cell_size
            if cx < 0 or cx >= grid or cy < 0 or cy >= grid:
                continue

            # rotate gradient angle relative to theta
            a = ang[yy, xx] - theta
            while a < 0:
                a += 2*np.pi
            while a >= 2*np.pi:
                a -= 2*np.pi

            b = int((a / (2*np.pi)) * bins) % bins
            wgt = np.exp(-(dx*dx + dy*dy) / denom)

            desc[int(cy), int(cx), b] += mag[yy, xx] * wgt

    vec = desc.flatten().astype(np.float32)

    # normalize + clamp + renormalize (SIFT style)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    vec = np.clip(vec, 0, 0.2)
    vec = vec / (np.linalg.norm(vec) + 1e-12)

    return vec


def octave_to_image_coords(o, y, x):
    scale = 2 ** o
    return int(x * scale), int(y * scale)


def sift_detect_and_compute(img, num_octaves=4, s=3, sigma0=1.6,
                            contrast_thresh=0.03, edge_r=10):
    gauss_pyr, sigmas = build_sift_gaussian_pyramid(img, num_octaves=num_octaves, s=s, sigma0=sigma0)
    dog_pyr = build_dog_pyramid(gauss_pyr)

    raw = find_dog_keypoints(dog_pyr, sigmas, contrast_thresh=contrast_thresh, edge_r=edge_r)

    keypoints = []
    descriptors = []

    for (o, level, y, x, sigma) in raw:
        gauss_img = gauss_pyr[o][level]  # same level index
        theta = assign_orientation(gauss_img, y, x, sigma)
        if theta is None:
            continue
        desc = sift_descriptor(gauss_img, y, x, sigma, theta)
        if desc is None:
            continue

        X, Y = octave_to_image_coords(o, y, x)
        keypoints.append((X, Y))
        descriptors.append(desc)

    return keypoints, np.array(descriptors, dtype=np.float32)


if __name__ == "__main__":
    print("Testing SIFT.py...")
    print()

    img = load_gray("data/sample.jpg")
    kps, desc = sift_detect_and_compute(img, contrast_thresh=0.03, edge_r=10)

    print("keypoints:", len(kps), "descriptors:", desc.shape)  # desc should be (N, 128)

    img_kp = draw_keypoints(img, kps, radius=3, color=(0, 255, 0))
    show(img_kp, f"SIFT keypoints ({len(kps)})")