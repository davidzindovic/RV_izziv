import cv2 as cv
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

path_do_videjev='C:\\Users\\david\\Desktop\\izziv main\\'

ime_videja="64210323_video_7"
ime_videja_2="64210323_video_3"

pravilna_anotacija_path='C:\\Users\\david\\Desktop\\izziv main\\rocna_anotacija\\'

ime_pravilna_anotacija="64210323_video_7"
ime_pravilna_anotacija_2="64210323_video_3"
debug_video=0

# funkcija za pridobitev slike iz videja na podlagi številke frame-a:
def get_video_frame(video_path, frame_number):
    """(Previous implementation remains the same)"""
    
    #PC doma:
    #video_dir = 'C:\\Users\\David Zindović\\Desktop\\Fax-Mag\\RV\\izziv\\izziv main\\'
    
    #Laptop šola:
    #video_dir = 'C:\\Users\\Vegova\\Documents\\zindo\\'
    
    #Laptop osebni:
    #video_dir = 'C:\\Users\\david\\Desktop\\izziv main\\'
    
    video_dir=path_do_videjev

    video_path = video_dir + video_path + ".mp4"
    cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        if debug_video==1:
            print(f"Error: Could not open video file {video_path}")
        return None
    
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    if frame_number < 0 or frame_number >= total_frames:
        if debug_video==1:
            print(f"Error: Frame {frame_number} is out of range (0-{total_frames-1})")
        cap.release()
        return None
    
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    else:
        if debug_video==1:
            print(f"Error: Could not read frame {frame_number}")
        return None

def detect_and_visualize_colored_objects(image, min_objects=1, color_tolerance=40, contour_thickness=2):
    """
    Detects colored objects in an image within ROI using combined color and edge detection.
    
    Parameters:
    - image: Input image (numpy array)
    - min_objects: Minimum number of objects needed to consider a color significant
    - color_tolerance: Allowed variation in color detection (0-255)
    - contour_thickness: Thickness of the drawn contours
    
    Returns:
    - Dictionary with color names and counts of detected objects
    - Displays visualization windows with detected objects
    """
    
    if image is None:
        print("Error: Could not read image")
        return {}
    
    # Define ROI (x, y, width, height)
    roi_x, roi_y, roi_w, roi_h = 150, 50, 200, 400
    
    # Extract ROI from the image
    roi = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    # Create copies for visualization
    original_img = image.copy()
    contour_img = image.copy()
    mask_visualization = np.zeros_like(image)
    combined_visualization = image.copy()
    
    # Convert ROI to LAB color space (better for color perception)
    lab_roi = cv.cvtColor(roi, cv.COLOR_BGR2LAB)
    
    # Edge detection on ROI
    gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray_roi, 50, 150)
    
    # Apply morphological closing to edges
    kernel = np.ones((3,3), np.uint8)
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    
    # Find contours from edges
    edge_contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Reshape the ROI to be a list of pixels for color clustering
    pixels = roi.reshape((-1, 3))
    
    # Find dominant colors using K-means clustering
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(pixels)
    
    # Get the dominant colors (BGR format)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    
    # Filter out neutral colors (whites, grays, blacks)
    def is_neutral(color):
        lab_color = cv.cvtColor(np.uint8([[color]]), cv.COLOR_BGR2LAB)[0][0]
        # Check for low saturation (grays) or very high/low lightness (white/black)
        return lab_color[1] < 30 and lab_color[2] < 30 or lab_color[0] < 30 or lab_color[0] > 220
    
    filtered_colors = [color for color in dominant_colors if not is_neutral(color)]
    
    if not filtered_colors:
        print("No significant colors found in the ROI (only neutrals detected)")
        return {}
    
    # Standard colors for visualization
    STANDARD_COLORS = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'orange': (0, 165, 255),
        'purple': (128, 0, 128),
        'pink': (203, 192, 255)
    }
    
    # Name the colors approximately
    def approximate_color_name(bgr_color):
        b, g, r = bgr_color
        if (r > g*1.5) and (r > b*1.5):
            return 'red', STANDARD_COLORS['red']
        elif (g > r*1.5) and (g > b*1.5):
            return 'green', STANDARD_COLORS['green']
        elif (b > r*1.5) and (b > g*1.5):
            return 'blue', STANDARD_COLORS['blue']
        elif (r > g*1.2) and (g > b*1.2):
            return 'yellow', STANDARD_COLORS['yellow']
        elif (r > g*1.2) and (g > b) and not (g > b*1.2):
            return 'orange', STANDARD_COLORS['orange']
        elif (b > g) and (r > g):
            return 'purple', STANDARD_COLORS['purple']
        elif (r > b) and (b > g):
            return 'pink', STANDARD_COLORS['pink']
        else:
            return f"rgb({r},{g},{b})", tuple(map(int, (b, g, r)))
    
    # Prepare results dictionary
    results = defaultdict(int)
    color_info = {}
    
    for color in filtered_colors:
        color_name, contour_color = approximate_color_name(color)
        target_color = color.astype(int)
        
        # Convert target color to LAB
        target_lab = cv.cvtColor(np.uint8([[target_color]]), cv.COLOR_BGR2LAB)[0][0]
        
        # Define color range bounds in LAB space
        lower_bound = np.array([
            max(0, target_lab[0] - 30),  # Lightness
            max(0, target_lab[1] - color_tolerance),  # A
            max(0, target_lab[2] - color_tolerance)   # B
        ])
        
        upper_bound = np.array([
            min(255, target_lab[0] + 30),
            min(255, target_lab[1] + color_tolerance),
            min(255, target_lab[2] + color_tolerance)
        ])
        
        # Create mask for the color range in ROI (LAB space)
        mask_roi = cv.inRange(lab_roi, lower_bound, upper_bound)
        
        # Combine with edge information
        combined_mask_roi = cv.bitwise_and(mask_roi, edges)
        
        # Apply morphological operations
        kernel = np.ones((3,3), np.uint8)
        combined_mask_roi = cv.morphologyEx(combined_mask_roi, cv.MORPH_CLOSE, kernel, iterations=2)
        combined_mask_roi = cv.morphologyEx(combined_mask_roi, cv.MORPH_OPEN, kernel, iterations=1)
        
        # Create full image mask with ROI
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = combined_mask_roi
        
        # Find contours - using RETR_EXTERNAL to get only outer contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and solidity (to remove noise and incomplete shapes)
        min_contour_area = 30  # Small area for pins
        valid_contours = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > min_contour_area:
                # Check solidity (area vs convex hull area)
                hull = cv.convexHull(cnt)
                hull_area = cv.contourArea(hull)
                if hull_area > 0:
                    solidity = float(area)/hull_area
                    if solidity > 0.7:  # Only keep fairly solid shapes
                        epsilon = 0.01 * cv.arcLength(cnt, True)
                        approx = cv.approxPolyDP(cnt, epsilon, True)
                        valid_contours.append(approx)
        
        if len(valid_contours) >= min_objects:
            results[color_name] = len(valid_contours)
            color_info[color_name] = target_color.tolist()
            
            # Draw all valid contours on the full image
            cv.drawContours(contour_img, valid_contours, -1, contour_color, contour_thickness)
            cv.drawContours(combined_visualization, valid_contours, -1, contour_color, contour_thickness)
            
            # Create a colored mask for visualization
            color_mask = cv.bitwise_and(image, image, mask=mask)
            mask_visualization = cv.add(mask_visualization, color_mask)
    
    # Draw ROI rectangle on visualizations
    cv.rectangle(original_img, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255, 255, 255), 2)
    cv.rectangle(contour_img, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255, 255, 255), 2)
    cv.rectangle(combined_visualization, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255, 255, 255), 2)
    
    # Display results
    print("Detected objects by color in ROI:")
    for color, count in results.items():
        rgb_color = color_info[color][::-1]  # Convert BGR to RGB for display
        print(f"{color} (RGB: {rgb_color}): {count} objects")
    
    # Show visualization windows
    cv.imshow('Original Image with ROI', original_img)
    cv.imshow('Edge Detection', edges)
    cv.imshow('Color-Based Detection', contour_img)
    cv.imshow('Combined Color+Edge Detection', combined_visualization)
    cv.imshow('Color Masks', mask_visualization)
    
    # Create a legend image
    legend = np.zeros((100 + len(results)*25, 300, 3), dtype=np.uint8)
    y_offset = 30
    for i, (color_name, count) in enumerate(results.items()):
        _, contour_color = approximate_color_name(color_info[color_name])
        cv.putText(legend, f"{color_name}: {count}", (10, y_offset + i*25), 
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv.circle(legend, (250, y_offset + i*25 - 5), 8, contour_color, -1)
    
    cv.imshow('Detection Legend', legend)
    
    # Resize windows for better viewing
    for winname in ['Original Image with ROI', 'Edge Detection', 'Color-Based Detection', 
                   'Combined Color+Edge Detection', 'Color Masks', 'Detection Legend']:
        cv.namedWindow(winname, cv.WINDOW_NORMAL)
        cv.resizeWindow(winname, 800, 600)
    
    print("\nPress any key on the image windows to continue...")
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return dict(results)

def find_rotated_rectangles(img):
    # Define ROI
    roi_x, roi_y, roi_w, roi_h = 150, 50, 200, 400
    roi = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    # Convert to HSV for color filtering
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    
    # Define range for bright colors (adjust these values based on your needs)
    lower_bright = np.array([0, 0, 200])
    upper_bright = np.array([180, 50, 255])
    
    # Threshold the HSV image to get only bright colors
    mask = cv.inRange(hsv, lower_bright, upper_bright)
    
    # Apply morphological operations
    kernel = np.ones((3,3), np.uint8)
    eroded = cv.erode(mask, kernel, iterations=1)
    dilated = cv.dilate(eroded, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Prepare output image
    output = img.copy()
    
    # Process each contour
    for cnt in contours:
        # Skip small contours
        #if cv.contourArea(cnt) < 500:
        #    continue
            
        # Get rotated rectangle
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int32(box)
        
        # Adjust coordinates to original image space
        box[:, 0] += roi_x
        box[:, 1] += roi_y
        
        # Draw rotated rectangle
        cv.drawContours(output, [box], 0, (0, 255, 0), 2)
    
    # Draw ROI rectangle
    cv.rectangle(output, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255, 0, 0), 2)
    
    # Show results
    cv.imshow('Detected Rotated Rectangles', output)
    cv.waitKey(0)
    cv.destroyAllWindows()


def find_dark_rotated_rectangles(img):
    # Define ROI
    roi_x, roi_y, roi_w, roi_h = 150, 50, 200, 400
    roi = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
    
    # Convert to grayscale and invert (since we want dark objects on white background)
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY_INV)  # Threshold for dark objects
    
    # Noise removal with morphological opening
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area (dilation to expand foreground boundaries)
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    
    # Distance transform to find sure foreground
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Unknown region (subtract sure_fg from sure_bg)
    unknown = cv.subtract(sure_bg, sure_fg)
    
    # Marker labelling for watershed
    _, markers = cv.connectedComponents(sure_fg)
    markers += 1  # Add 1 to all labels so background is 1, not 0
    markers[unknown == 255] = 0  # Mark unknown region as 0
    
    # Apply watershed
    markers = cv.watershed(roi, markers)
    roi[markers == -1] = [0, 255, 0]  # Mark watershed boundaries in green
    
    # Find contours on the sure_fg mask
    contours, _ = cv.findContours(sure_fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Prepare output image
    output = img.copy()
    
    for cnt in contours:
        # Skip contours that are too small or too large (adjust as needed)
        area = cv.contourArea(cnt)
        if area < 100 or area > 2000:
            continue
        
        # Get rotated rectangle
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        
        # Adjust coordinates to original image space
        box[:, 0] += roi_x
        box[:, 1] += roi_y
        
        # Draw rotated rectangle
        cv.drawContours(output, [box], 0, (0, 0, 255), 2)  # Red rectangles
    
    # Draw ROI rectangle (blue)
    cv.rectangle(output, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
    
    # Show results
    cv.imshow('Detected Dark Rotated Rectangles', output)
    cv.waitKey(0)
    cv.destroyAllWindows()
# Example usage:
# img = cv.imread('your_image.jpg')
# find_rotated_rectangles(img)



def detect_objects_in_roi(image, roi_x=150, roi_y=50, roi_w=200, roi_h=400):
    """
    Detect and segment objects inside a specified ROI using:
    - filter2D (edge enhancement)
    - distanceTransform (object separation)
    - Watershed (segmentation)
    
    Args:
        image (numpy.ndarray): Input BGR image.
        roi_x, roi_y, roi_w, roi_h: ROI coordinates and dimensions.
    
    Returns:
        numpy.ndarray: Image with segmented objects highlighted.
    """
    if image is None:
        raise ValueError("Input image is invalid.")

    # Extract ROI
    roi = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    # Convert to grayscale & blur
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    cv.imshow("g",gray)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Edge enhancement (custom kernel)
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    edges = cv.filter2D(blurred, -1, kernel)

    # Thresholding to get binary image
    _, binary = cv.threshold(edges, 100, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Noise removal (morphological opening)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)
    cv.imshow("2",opening)
    # Sure background area (dilation)
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Distance transform to find sure foreground
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    cv.imshow("11",dist_transform)
    _, sure_fg = cv.threshold(dist_transform, 0.5 * dist_transform.max(), 255, cv.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)
    cv.imshow("1",sure_fg)
    # Unknown region (subtract sure_fg from sure_bg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling for Watershed
    _, markers = cv.connectedComponents(sure_fg)
    markers += 1  # Background = 1
    markers[unknown == 255] = 0  # Unknown = 0

    # Apply Watershed
    markers = cv.watershed(roi, markers)

    # Highlight boundaries (-1 = watershed lines)
    roi[markers == -1] = [0, 0, 255]  # Red borders

    # Overlay result on original image
    result = image.copy()
    result[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = roi

    # Display
    cv.imshow("Segmented Objects in ROI", result)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return result



def detect_objects_in_roi_improved(image, roi_x=150, roi_y=50, roi_w=200, roi_h=400):
    """
    Improved object detection in ROI with better handling of small dark objects.
    
    Args:
        image (numpy.ndarray): Input BGR image.
        roi_x, roi_y, roi_w, roi_h: ROI coordinates and dimensions.
    
    Returns:
        numpy.ndarray: Image with segmented objects highlighted.
    """
    if image is None:
        raise ValueError("Input image is invalid.")

    # Extract ROI
    roi = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
    
    # Convert to LAB color space (better for brightness separation)
    lab = cv.cvtColor(roi, cv.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    # Merge back and convert to grayscale
    enhanced_lab = cv.merge([enhanced_l, a_channel, b_channel])
    enhanced_gray = cv.cvtColor(enhanced_lab, cv.COLOR_LAB2BGR)
    enhanced_gray = cv.cvtColor(enhanced_gray, cv.COLOR_BGR2GRAY)
    
    cv.imshow("w",enhanced_gray)
    # Blur to reduce noise
    blurred = cv.GaussianBlur(enhanced_gray, (5, 5), 0)
    cv.imshow("q",blurred)
    # Adaptive thresholding (better for varying illumination)
    binary = cv.adaptiveThreshold(
        blurred, 255, 
        cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv.THRESH_BINARY_INV, 11, 2
    )
    cv.imshow("bb",binary)
    # Morphological operations to remove noise and enhance objects
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)

    # Sure background (dilation)
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Distance transform to find sure foreground
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(
        dist_transform, 
        0.6 * dist_transform.max(),  # Lower threshold for small objects
        255, 
        cv.THRESH_BINARY
    )
    sure_fg = np.uint8(sure_fg)

    # Unknown region (subtract sure_fg from sure_bg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling for Watershed
    _, markers = cv.connectedComponents(sure_fg)
    markers += 1  # Background = 1
    markers[unknown == 255] = 0  # Unknown = 0

    # Apply Watershed
    markers = cv.watershed(roi, markers)

    # Highlight boundaries (-1 = watershed lines)
    roi[markers == -1] = [0, 0, 255]  # Red borders

    # Optional: Filter small contours (remove noise)
    contours, _ = cv.findContours(
        sure_fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    min_contour_area = 100  # Adjust based on object size
    for cnt in contours:
        if 200>cv.contourArea(cnt) > min_contour_area:
            #print("a")
            cv.drawContours(roi, [cnt], -1, (0, 255, 0), 1)  # Green contours

    # Overlay result on original image
    result = image.copy()
    result[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = roi

    # Display
    cv.imshow("Improved Object Detection in ROI", result)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return result




import cv2 as cv
import numpy as np

def detect_dark_objects(image_path):
    # Read the image
    img = image_path
    if img is None:
        print("Error: Could not read the image.")
        return
    
    # Define ROI (x, y, width, height)
    roi = (150, 50, 200, 400)
    x, y, w, h = roi
    
    # Extract ROI from the image
    roi_img = img[y:y+h, x:x+w].copy()
    
    # Convert ROI to grayscale
    gray = cv.cvtColor(roi_img, cv.COLOR_BGR2GRAY)

    # Enhanced contrast for dark regions
    alpha = 0.5  # Contrast control (1.0-3.0)
    beta = 50    # Brightness control (-100 to 100)
    enhanced = cv.convertScaleAbs(gray, alpha=alpha, beta=beta)
    
    # Apply adaptive thresholding for better edge detection
    thresh = cv.adaptiveThreshold(enhanced, 255, #!!!!!!!!!!!
                                 cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv.THRESH_BINARY_INV, 11, 2)
    

    # Combine with Canny edge detection
    blurred = cv.GaussianBlur(enhanced, (9, 9), 0)
    edges = cv.Canny(blurred, 40, 100)
    combined_edges = cv.bitwise_or(thresh, edges)

    # Find contours in ROI
    contours, _ = cv.findContours(combined_edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Process each contour
    object_counter = 0
    obj_display = img.copy()
    for contour in contours:
        # Calculate contour area and filter
        area = cv.contourArea(contour)
        if area < 60 or area > 300:
            continue

        # Create a mask for the current contour
        mask = np.zeros_like(gray)
        cv.drawContours(mask, [contour], -1, 255, -1)
        
        # Calculate mean pixel value within the contour
        mean_val = cv.mean(gray, mask=mask)[0]
        
        if mean_val < 140:
            object_counter += 1
            # Create a copy of the original image for this object
            
            # Translate contour coordinates back to original image
            translated_contour = contour + np.array([x, y])
            
            # Draw the contour on the image (green)
            cv.drawContours(obj_display, [translated_contour], -1, (0, 255, 0), 2)
    
    cv.imshow("a",obj_display)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Example usage:
# detect_dark_objects('your_image.jpg')

# Example usage:
if __name__ == "__main__":
    # Call the function with just an image path
    detect_dark_objects(get_video_frame(ime_videja_2,1))
    #find_rotated_rectangles(get_video_frame(ime_videja,1))
    #results = detect_and_visualize_colored_objects(get_video_frame(ime_videja,1), min_objects=1, color_tolerance=40, contour_thickness=2)
