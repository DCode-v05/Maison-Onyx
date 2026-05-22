"""
Image Similarity Check Using Pixel-Based Methods (No ML Models)
Compares two images using multiple classical computer vision techniques.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import sys


def load_and_resize(img_path, size=(256, 256)):
    """Load image and resize to standard size for fair comparison."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    img = cv2.resize(img, size)
    return img


# ==========================================================================
# Method 1: Mean Squared Error (MSE)
# ==========================================================================
def mse_similarity(img1, img2):
    """
    Lower MSE = more similar. 0 = identical.
    Very sensitive to lighting, rotation, position.
    """
    err = np.mean((img1.astype("float") - img2.astype("float")) ** 2)
    # Convert to similarity score 0-1 (1 = identical)
    similarity = 1 / (1 + err / 1000)
    return similarity, err


# ==========================================================================
# Method 2: Structural Similarity Index (SSIM)
# ==========================================================================
def ssim_similarity(img1, img2):
    """
    Range: -1 to 1. Higher = more similar.
    Considers structure, luminance, contrast.
    More robust than MSE.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score


# ==========================================================================
# Method 3: Histogram Comparison
# ==========================================================================
def histogram_similarity(img1, img2):
    """
    Compares color distributions.
    Range: 0 to 1 with correlation method. Higher = more similar.
    Robust to position changes, sensitive to color/lighting.
    """
    # Convert to HSV (better for color comparison than BGR)
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    
    # Calculate histograms
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    
    # Normalize
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    # Compare using correlation
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score


# ==========================================================================
# Method 4: Perceptual Hash (pHash)
# ==========================================================================
def phash_similarity(img1, img2, hash_size=16):
    """
    Generates a fingerprint of the image.
    Very robust to small changes, scaling, minor edits.
    Returns similarity 0-1 (1 = identical hash).
    """
    def compute_phash(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize to small size
        resized = cv2.resize(gray, (hash_size, hash_size))
        # Compute DCT
        dct = cv2.dct(np.float32(resized))
        # Take top-left 8x8 (low frequencies = main structure)
        dct_low = dct[:8, :8]
        # Compute median and create binary hash
        median = np.median(dct_low)
        return (dct_low > median).flatten()
    
    hash1 = compute_phash(img1)
    hash2 = compute_phash(img2)
    
    # Hamming distance: count of differing bits
    hamming_distance = np.sum(hash1 != hash2)
    # Convert to similarity (0-1)
    similarity = 1 - (hamming_distance / len(hash1))
    return similarity


# ==========================================================================
# Method 5: ORB Feature Matching (keypoint-based)
# ==========================================================================
def orb_similarity(img1, img2):
    """
    Detects keypoints and matches them between images.
    Robust to rotation, scaling, partial occlusion.
    Best for objects with distinct features (like jewelry patterns).
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=500)
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0, 0
    
    # Match descriptors using Brute Force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Count "good" matches (distance below threshold)
    good_matches = [m for m in matches if m.distance < 50]
    
    # Similarity = ratio of good matches to total keypoints
    total_keypoints = min(len(kp1), len(kp2))
    similarity = len(good_matches) / total_keypoints if total_keypoints > 0 else 0
    
    return min(similarity, 1.0), len(good_matches)


# ==========================================================================
# Combined Score (weighted ensemble)
# ==========================================================================
def combined_similarity(img1, img2):
    """Combine all methods with weights for robust scoring."""
    mse_score, _ = mse_similarity(img1, img2)
    ssim_score = ssim_similarity(img1, img2)
    hist_score = histogram_similarity(img1, img2)
    phash_score = phash_similarity(img1, img2)
    orb_score, _ = orb_similarity(img1, img2)
    
    # Weights — tune based on your use case
    weights = {
        'mse':   0.10,
        'ssim':  0.25,
        'hist':  0.20,
        'phash': 0.20,
        'orb':   0.25,
    }
    
    combined = (
        weights['mse']   * mse_score +
        weights['ssim']  * max(0, ssim_score) +  # SSIM can be negative
        weights['hist']  * max(0, hist_score) +
        weights['phash'] * phash_score +
        weights['orb']   * orb_score
    )
    
    return combined, {
        'mse': mse_score,
        'ssim': ssim_score,
        'histogram': hist_score,
        'phash': phash_score,
        'orb': orb_score,
    }


# ==========================================================================
# Main
# ==========================================================================
def compare_images(template_path, input_path, threshold=0.70):
    """Full comparison pipeline."""
    print(f"\nComparing:\n  Template: {template_path}\n  Input:    {input_path}\n")
    
    img1 = load_and_resize(template_path)
    img2 = load_and_resize(input_path)
    
    combined_score, scores = combined_similarity(img1, img2)
    
    print("Individual scores:")
    print(f"  MSE similarity:       {scores['mse']:.4f}")
    print(f"  SSIM:                 {scores['ssim']:.4f}")
    print(f"  Histogram (HSV):      {scores['histogram']:.4f}")
    print(f"  Perceptual hash:      {scores['phash']:.4f}")
    print(f"  ORB feature matching: {scores['orb']:.4f}")
    print(f"\nCombined score: {combined_score:.4f}")
    print(f"Threshold:      {threshold:.4f}")
    
    if combined_score >= threshold:
        print("Result: PASS (Match)")
    else:
        print("Result: FAIL (No match)")
    
    return combined_score, scores


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        template = sys.argv[1]
        new_input = sys.argv[2]
        threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.70
    else:
        # Default test paths — edit these
        template = "template.jpg"
        new_input = "input.jpg"
        threshold = 0.70
    
    compare_images(template, new_input, threshold)