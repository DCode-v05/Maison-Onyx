import cv2

# Path to input image
image_path = "master.png"

# Read image
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (9, 9), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 100, 200)

# Assume display resolution = 1920x1080
SCREEN_W = 1920
SCREEN_H = 1080

# Keep margin so windows fit properly
MAX_W = SCREEN_W // 2       # 960 px each window
MAX_H = int(SCREEN_H * 0.8) # 864 px

def resize_to_fit(img, max_w, max_h):
    h, w = img.shape[:2]

    scale = min(max_w / w, max_h / h, 1.0)  
    # Prevent enlarging small images

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

# Resize images
image_display = resize_to_fit(image, MAX_W, MAX_H)
edges_display = resize_to_fit(edges, MAX_W, MAX_H)

# Create resizable windows
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Canny Edges", cv2.WINDOW_NORMAL)

cv2.imshow("Original Image", image_display)
cv2.imshow("Canny Edges", edges_display)

cv2.waitKey(0)
cv2.destroyAllWindows()