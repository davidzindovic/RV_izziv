import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math


debug = 0


def get_video_frame(video_path, frame_number):
    """(Previous implementation remains the same)"""
    video_dir = 'C:\\Users\\david\\Desktop\\fax\\RV\\izziv_30.04.2025\\videji\\'
    video_path = video_dir + video_path + ".mp4"
    cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None
    
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    if frame_number < 0 or frame_number >= total_frames:
        print(f"Error: Frame {frame_number} is out of range (0-{total_frames-1})")
        cap.release()
        return None
    
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    else:
        print(f"Error: Could not read frame {frame_number}")
        return None

def create_offset_grid_from_lines(image, parallel_line_groups, side, show_result=True):
    """
    Creates a 3x3 grid of points with proper perpendicular offsets from two sets of lines.
    
    Args:
        image: Input image
        parallel_line_groups: List of at least two parallel line groups
        side: na kateri strani (polovica ekrana - zgoraj spodaj) so luknje 3x3
        show_result: Whether to visualize the result
        
    Returns:
        tuple: (visualization_image, grid_points, first_point_coords, horizontal_offset, vertical_offset)
               where grid_points is a 3x3 array of (x,y) coordinates
    """
    if len(parallel_line_groups) < 2:
        print("Need at least two groups of parallel lines (horizontal and vertical)")
        return image, None, None, None, None
    
    if side=="spodaj":
        horizontal_offset=32
        vertical_offset=30
        point_spacing_x=38
        point_spacing_y=40
    elif side=="zgoraj":
        horizontal_offset=45
        vertical_offset=23
        point_spacing_x=35
        point_spacing_y=25        

    # Make a copy of the image for visualization
    vis_image = image.copy()

    # Get horizontal and vertical line groups (modify indices if needed)

    horizontal_lines = parallel_line_groups[1]  # First group for horizontal lines
    vertical_lines = parallel_line_groups[0]    # Second group for vertical lines
    
    # Calculate average angles
    def get_avg_angle(lines):
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line
            angle = np.arctan2(y2 - y1, x2 - x1)
            angles.append(angle)
        return np.mean(angles)
    
    horizontal_angle = get_avg_angle(horizontal_lines)
    vertical_angle = get_avg_angle(vertical_lines)
    
    # Calculate perpendicular angles
    horizontal_perp = horizontal_angle + np.pi/2  # Perpendicular to horizontal lines
    #vertical_perp = vertical_angle + np.pi/2      # Perpendicular to vertical lines
    vertical_perp= horizontal_angle
    vertical_angle=horizontal_perp

    if debug==1:
        print("H: "+str(np.rad2deg(horizontal_angle))+" | Hp: "+str(np.rad2deg(horizontal_perp)))
        print("V: "+str(np.rad2deg(vertical_angle))+" | Vp: "+str(np.rad2deg(vertical_perp)))

    # Find reference lines for offsets
    # For horizontal offset: use the left-most vertical line
    left_vertical = min(vertical_lines, key=lambda l: min(l[0], l[2]))
    # For vertical offset: use the bottom-most horizontal line

    if side=="spodaj":
        bottom_horizontal = max(horizontal_lines, key=lambda l: max(l[1], l[3]))
    elif side=="zgoraj":
        bottom_horizontal = min(horizontal_lines, key=lambda l: min(l[1], l[3]))
    
    # Get reference points
    # For horizontal offset: bottom point of left vertical line
    x_vert_ref = min(left_vertical[0], left_vertical[2])

    if side=="spodaj":
        y_vert_ref = max(left_vertical[1], left_vertical[3])
    elif side=="zgoraj":
        y_vert_ref = min(left_vertical[1], left_vertical[3])
    
    # For vertical offset: left point of bottom horizontal line
    x_horz_ref = min(bottom_horizontal[0], bottom_horizontal[2])
    
    y_horz_ref = max(bottom_horizontal[1], bottom_horizontal[3])

    # Calculate grid origin point with both offsets
    #x_start = x_vert_ref + horizontal_offset * np.cos(horizontal_perp)
    #y_start = y_horz_ref - vertical_offset * np.sin(vertical_perp)
    x_start = x_vert_ref + horizontal_offset * np.cos(horizontal_angle)

    if side=="spodaj":
        y_start = y_horz_ref - vertical_offset * np.sin(vertical_angle)-2*point_spacing_y* np.sin(vertical_angle)
    elif side=="zgoraj":
        y_start = y_horz_ref + vertical_offset * np.sin(vertical_angle)

    # Create 3x3 grid
    grid_points = []
    for row in range(3):
        row_points = []
        for col in range(3):
            # Calculate point position
            #x = x_start + col * point_spacing * np.cos(vertical_angle)
            #y = y_start + row * point_spacing * np.sin(horizontal_angle)
            x = x_start + col * point_spacing_x * np.cos(horizontal_angle)
            y = y_start + row * point_spacing_y * np.sin(vertical_angle)
            row_points.append((int(x), int(y)))
        grid_points.append(row_points)
    
    # Convert to numpy array
    grid_points = np.array(grid_points)
    
    # Visualization
    if show_result:
        # Draw original lines
        for line in horizontal_lines:
            cv.line(vis_image, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)  # Green for horizontal
        for line in vertical_lines:
            cv.line(vis_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)  # Red for vertical
        
        # Draw offset reference lines
        # Horizontal offset line (perpendicular to horizontal lines)
        x_h_end = x_vert_ref + horizontal_offset * np.cos(horizontal_perp)
        y_h_end = y_vert_ref + horizontal_offset * np.sin(horizontal_perp)
        cv.line(vis_image, (int(x_vert_ref), int(y_vert_ref)), (int(x_h_end), int(y_h_end)), 
                (255, 255, 0), 1, cv.LINE_AA)
        
        # Vertical offset line (perpendicular to vertical lines)
        x_v_end = x_horz_ref + vertical_offset * np.cos(vertical_perp)
        y_v_end = y_horz_ref + vertical_offset * np.sin(vertical_perp)
        cv.line(vis_image, (int(x_horz_ref), int(y_horz_ref)), (int(x_v_end), int(y_v_end)), 
                (255, 0, 255), 1, cv.LINE_AA)
        
        # Draw grid points
        for row in range(3):
            for col in range(3):
                x, y = grid_points[row, col]
                cv.circle(vis_image, (x, y), 5, (0, 255, 255), -1)  # Yellow points
                cv.putText(vis_image, f"{row},{col}", (x+10, y+10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw reference points
        cv.circle(vis_image, (int(x_vert_ref), int(y_vert_ref)), 5, (255, 0, 0), -1)  # Blue - vertical ref
        if debug==1:
            print("BLUE dot: x="+str(x_vert_ref)+" y="+str(y_vert_ref))
        cv.circle(vis_image, (int(x_horz_ref), int(y_horz_ref)), 5, (0, 0, 255), -1)  # Red - horizontal ref
        if debug==1:
            print("RED dot: x="+str(x_horz_ref)+" y="+str(y_horz_ref))
        cv.circle(vis_image, (int(x_start), int(y_start)), 5, (0, 255, 0), -1)        # Green - grid origin
        
        cv.imshow("Grid with Perpendicular Offsets", vis_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    return vis_image, grid_points, grid_points[0, 0], horizontal_offset, vertical_offset

def detect_bright_objects(image, brightness_threshold=230, min_area=50, show_results=True, circularity_threshold=0.7, max_contour_gap=10):
    """
    Detects bright white objects in an image using thresholding and edge detection,
    including almost complete circular contours.
    
    Args:
        image: Input BGR image
        brightness_threshold: Threshold value for white detection (0-255)
        min_area: Minimum area for a contour to be considered
        show_results: Whether to display intermediate processing steps
        circularity_threshold: How circular the contour must be (0-1, 1=perfect circle)
        max_contour_gap: Maximum gap to bridge in almost complete contours (pixels)
        
    Returns:
        list: List of (x,y) center points of detected bright objects
        numpy.ndarray: Thresholded image
        numpy.ndarray: Edge detection result
    """
    # Convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Threshold to find bright areas
    _, thresholded = cv.threshold(gray, brightness_threshold, 255, cv.THRESH_BINARY)
    
    # Perform edge detection on the thresholded image
    edges = cv.Canny(thresholded, 50, 150)
    
    # Close small gaps in edges to help complete almost-finished contours
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    closed_edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv.findContours(closed_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Prepare visualization image
    visualization = image.copy()
    center_points = []
    
    # Process each contour
    for contour in contours:
        # Filter by area
        area = cv.contourArea(contour)
        if area < min_area:
            continue
            
        # Calculate circularity (4*pi*area/perimeter^2)
        perimeter = cv.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        # Only process circular contours
        if circularity < circularity_threshold:
            continue
            
        # Get bounding circle
        (x, y), radius = cv.minEnclosingCircle(contour)
        center = (int(x), int(y))
        center_points.append(center)
        
        # Draw on visualization
        cv.drawContours(visualization, [contour], -1, (0, 255, 0), 2)  # Green contour
        cv.circle(visualization, center, 3, (0, 0, 255), -1)  # Red center point
        cv.putText(visualization, f"{center}", (center[0]+10, center[1]), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw circularity info
        cv.putText(visualization, f"{circularity:.2f}", (center[0]+10, center[1]+30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Display results if requested
    if show_results:
        cv.imshow("Original Image", image)
        cv.imshow("Thresholded (White Areas)", thresholded)
        cv.imshow("Edge Detection", edges)
        cv.imshow("Closed Edges", closed_edges)
        cv.imshow("Detected Bright Objects", visualization)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    return center_points, thresholded, edges

def detect_parallel_lines_in_roi(image, roi, rho=1, theta=np.pi/180, threshold=50, 
                               min_line_length=50, max_line_gap=20, 
                               angle_threshold=np.pi/18, show_result=False):
    """
    Detects lines using Hough Transform within a specified ROI and groups them by similar angles.
    
    Args:
        image: Input image (grayscale or color)
        roi: Region of interest as (x, y, width, height)
        rho: Distance resolution of accumulator in pixels
        theta: Angle resolution of accumulator in radians
        threshold: Accumulator threshold parameter
        min_line_length: Minimum line length
        max_line_gap: Maximum allowed gap between line segments
        angle_threshold: Angle difference threshold for considering lines parallel (in radians)
        show_result: Whether to display the detected lines
    
    Returns:
        List of groups of parallel lines, where each group is a list of lines in format [x1, y1, x2, y2]
        (coordinates are relative to the full image)
    """
    # Extract ROI coordinates
    x, y, w, h = roi
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Extract ROI from image
    roi_image = gray[y:y+h, x:x+w]
    
    # Edge detection within ROI
    edges = cv.Canny(roi_image, 50, 150, apertureSize=3)
    
    # Line detection using Hough Transform within ROI
    lines = cv.HoughLinesP(edges, rho, theta, threshold, 
                          minLineLength=min_line_length, 
                          maxLineGap=max_line_gap)
    
    if lines is None:
        return []
    
    # Convert lines from ROI coordinates to full image coordinates
    lines_full_image = []
    for line in lines:
        x1_roi, y1_roi, x2_roi, y2_roi = line[0]
        lines_full_image.append([x1_roi + x, y1_roi + y, x2_roi + x, y2_roi + y])
    
    # Calculate angles for each line
    line_angles = []
    for line in lines_full_image:
        x1, y1, x2, y2 = line
        angle = math.atan2(y2 - y1, x2 - x1)
        # Normalize angle to be between 0 and pi
        if angle < 0:
            angle += np.pi
        line_angles.append(angle)
    
    # Group lines by similar angles
    angle_groups = defaultdict(list)
    for i, angle in enumerate(line_angles):
        # Find the closest existing group
        found_group = False
        for group_angle in angle_groups:
            # Check if angle is within threshold of any group angle
            if abs(angle - group_angle) < angle_threshold or \
               abs(angle - group_angle - np.pi) < angle_threshold or \
               abs(angle - group_angle + np.pi) < angle_threshold:
                angle_groups[group_angle].append(i)
                found_group = True
                break
        
        # If no matching group found, create a new one
        if not found_group:
            angle_groups[angle].append(i)
    
    # Create parallel line groups (in full image coordinates)
    parallel_line_groups = []
    for group in angle_groups.values():
        parallel_lines = [lines_full_image[i] for i in group]
        parallel_line_groups.append(parallel_lines)
    
    # Visualize if requested
    if show_result:
        display_image = image.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                 (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        
        # Draw ROI rectangle
        cv.rectangle(display_image, (x, y), (x+w, y+h), (255, 255, 255), 2)
        
        # Draw lines
        for i, group in enumerate(parallel_line_groups):
            color = colors[i % len(colors)]
            for line in group:
                x1, y1, x2, y2 = line
                cv.line(display_image, (x1, y1), (x2, y2), color, 2)
        
        cv.imshow("Detected Parallel Lines in ROI", display_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    return parallel_line_groups



def detect_shapes(image, region_of_interest=None, min_area=100, max_area=10000, canny_low=50, canny_high=150):
    """Enhanced version that better detects trapezoidal shapes"""
    if image is None or image.size == 0:
        print("Error: Invalid image input")
        return None, None, None
    
    if region_of_interest:
        x, y, w, h = region_of_interest
        roi = image[y:y+h, x:x+w].copy()
    else:
        roi = image.copy()
        x, y = 0, 0
    
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, canny_low, canny_high)
    
    # Use RETR_LIST to get all contours, not just external ones
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    shapes = {
        'small_circles': [],
        'big_circles': [],
        'pins': [],
        'pin_in_bowl': [],
        'trapezoids': []
    }
    
    pins_centers=[]
    for contour in contours:
        area = cv.contourArea(contour)
        if area < min_area or area > max_area:
            continue
            
        perimeter = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.04 * perimeter, True)
        
        (cx, cy), radius = cv.minEnclosingCircle(contour)
        abs_cx = int(cx) + x
        abs_cy = int(cy) + y
        
        # Only process shapes that are mostly within the ROI
        if region_of_interest:
            x_roi, y_roi, w_roi, h_roi = region_of_interest
            contour_abs = contour + np.array([x, y])
            x_cont, y_cont, w_cont, h_cont = cv.boundingRect(contour_abs)
            
            # Calculate overlap percentage
            dx = min(x_cont + w_cont, x_roi + w_roi) - max(x_cont, x_roi)
            dy = min(y_cont + h_cont, y_roi + h_roi) - max(y_cont, y_roi)
            if dx <= 0 or dy <= 0:
                continue  # No overlap
            overlap_area = dx * dy
            if overlap_area / area < 0.7:  # At least 70% of shape must be in ROI
                continue
        
        rect = cv.minAreaRect(contour)
        width = rect[1][0]
        height = rect[1][1]
        aspect_ratio = max(width, height) / (min(width, height) + 1e-5)
        
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-5)
        mask = np.zeros_like(gray)
        cv.drawContours(mask, [contour], -1, 255, -1)
        mean_color = cv.mean(roi, mask=mask)[:3]
        color_name = classify_color(mean_color)
        
        contour_absolute = contour + np.array([x, y])

        shape_info = {
            'center': (abs_cx, abs_cy),
            'radius': int(radius),
            'contour': contour_absolute,
            'area': area,
            'color_name': color_name,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'bounding_rect': cv.boundingRect(contour_absolute)
        }

        if (aspect_ratio > 2 or 160 < area < 300):
            shapes['pin_in_bowl'].append(shape_info)
        elif cv.isContourConvex(approx) and 0.8 < circularity < 1.2 and area < 1000:
            shapes['small_circles'].append(shape_info)
        elif cv.isContourConvex(approx) and circularity > 0.8 and area >= 1000:
            shapes['big_circles'].append(shape_info)
        elif (aspect_ratio > 1 or 120 < area < 161):
            shapes['pins'].append(shape_info)
            pins_centers.append([abs_cx,abs_cy])
        elif (aspect_ratio > 1 and 300 < area):
            shapes['trapezoids'].append(shape_info)

    return pins_centers,shapes, (x, y, w, h) if region_of_interest else None, edges



def classify_color(hsv_values):
    """Classifies color based on HSV values"""
    hue, sat, val = hsv_values
    if val < 30: return "black"
    elif sat < 50:
        return "white" if val > 200 else "gray"
    if hue < 15: return "red"
    elif hue < 45: return "orange"
    elif hue < 75: return "yellow"
    elif hue < 105: return "green"
    elif hue < 135: return "teal"
    elif hue < 165: return "blue"
    elif hue < 195: return "purple"
    elif hue < 255: return "pink"
    return "red"

def visualize_detection(slika, shapes, roi_rect=None):
    """
    Enhanced visualization with separate area display boxes
    """
    display = slika.copy()
    
    if roi_rect:
        x, y, w, h = roi_rect
        cv.rectangle(display, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv.putText(display, "Search Region", (x, y-10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    shape_colors = {
        'small_circles': (0, 255, 255),  # Yellow
        'big_circles': (255, 0, 0),       # Blue
        'pins': (0, 255, 0),              # Green
        'pin_in_bowl':(255, 255, 0),
        'trapezoids': (0, 0, 255)         # Red
    }
    
    for shape_type in shapes:
        for shape in shapes[shape_type]:
            # Draw the contour
            cv.drawContours(display, [shape['contour']], -1, shape_colors[shape_type], 2)
            
            # Draw center point
            cv.circle(display, shape['center'], 2, (0, 0, 255), 3)
            
            # Get bounding rectangle coordinates
            x, y, w, h = shape['bounding_rect']
            
            # Create position for area box (right side of bounding rect)
            area_box_x = x + w + 10
            area_box_y = y
            
            # Format area text
            area_text = f"Area: {shape['area']:.0f}"
            
            # Calculate text size
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            (text_width, text_height), _ = cv.getTextSize(area_text, font, scale, thickness)
            
            # Draw area box background
            cv.rectangle(display,
                        (area_box_x - 5, area_box_y - text_height - 5),
                        (area_box_x + text_width + 5, area_box_y + 5),
                        (255, 255, 255), -1)
            
            # Draw area box border
            cv.rectangle(display,
                        (area_box_x - 5, area_box_y - text_height - 5),
                        (area_box_x + text_width + 5, area_box_y + 5),
                        (0, 0, 0), 1)
            
            # Draw area text
            cv.putText(display, area_text,
                      (area_box_x, area_box_y),
                      font, scale, (0, 0, 0), thickness)
            
            # Original label (shape type and color)
            label = f"{shape_type}: {shape['color_name']}"
            (label_width, label_height), _ = cv.getTextSize(label, font, scale, thickness)
            
            # Position original label above the contour
            label_x = shape['center'][0] - label_width // 2
            label_y = shape['center'][1] - shape['radius'] - 10
            
            # Draw label background
            cv.rectangle(display,
                        (label_x - 2, label_y - label_height - 2),
                        (label_x + label_width + 2, label_y + 2),
                        (255, 255, 255), -1)
            
            # Draw label
            cv.putText(display, label,
                      (label_x, label_y),
                      font, scale, (0, 0, 0), thickness)
    
    return display

def detect_object_position_bias(image, shapes, threshold_ratio=0.5, show_result=False):
    """
    Determines if there are more detected objects towards the bottom or top of the image.
    
    Args:
        image: Input image (used only for visualization if show_result=True)
        shapes: Dictionary of detected shapes (output from detect_shapes function)
        threshold_ratio: Ratio of image height to consider as top/bottom regions (default 1/3)
        show_result: Whether to display visualization of the regions and counts
    
    Returns:
        str: 'top' if more objects in top region, 'bottom' if more in bottom region, 
             'equal' if counts are equal, or 'none' if no objects detected
    """
    if not shapes or all(len(v) == 0 for v in shapes.values()):
        return 'none'
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate threshold positions
    top_threshold = height * threshold_ratio
    bottom_threshold = height * (1 - threshold_ratio)
    
    # Count objects in top and bottom regions
    top_count = 0
    bottom_count = 0
    
    for shape_type in shapes:
        for shape in shapes[shape_type]:
            y = shape['center'][1]
            if y < top_threshold:
                top_count += 1
            elif y > bottom_threshold:
                bottom_count += 1
    
    # Determine result
    if top_count > bottom_count:
        result = 'spodaj'
    elif bottom_count > top_count:
        result = 'zgoraj'
    else:
        result = 'sredina'
    
    # Visualization
    if show_result:
        vis = image.copy()
        
        # Draw threshold lines
        cv.line(vis, (0, int(top_threshold)), (width, int(top_threshold)), 
               (0, 255, 255), 2)  # Yellow for top threshold
        cv.line(vis, (0, int(bottom_threshold)), (width, int(bottom_threshold)), 
               (0, 165, 255), 2)  # Orange for bottom threshold
        
        # Add text showing counts
        cv.putText(vis, f"Top: {top_count}", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv.putText(vis, f"Bottom: {bottom_count}", (10, 70), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        cv.putText(vis, f"Result: {result}", (width//2 - 100, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv.imshow("Object Position Analysis", vis)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    return result

if __name__ == "__main__":
    # Load image
    for i in range(1):
        slika = get_video_frame("64210323_video_5", i+200)
        if slika is None:
            print("Failed to load image")
        else:
            # Convert to BGR for processing
            image_bgr = cv.cvtColor(slika, cv.COLOR_RGB2BGR)
            
            # Define search region (x, y, width, height)
            roi = (150, 50, 200, 400)
            
            # Set parameters
            min_area = 50
            max_area = 3000
            canny_low = 50
            canny_high = 150

            # Detect shapes within ROI
            pin_centers,shapes, roi_rect, edges = detect_shapes(image_bgr, 
                                                  region_of_interest=roi,
                                                  min_area=min_area,
                                                  max_area=max_area,
                                                  canny_low=canny_low,
                                                  canny_high=canny_high)
            
            stran_lukenj = detect_object_position_bias(image_bgr, shapes, show_result=True)
            print("Luknje za pine so "+stran_lukenj)
                
            # Visualize
            result = visualize_detection(image_bgr, shapes, roi)
            
            # Detect parallel lines in ROI
            parallel_line_groups = detect_parallel_lines_in_roi(
            image_bgr, 
            roi,
            angle_threshold=np.pi/18,  # 10 degrees
            show_result=True
            )

            if parallel_line_groups and len(parallel_line_groups) >= 2:
                # Create offset grid
                vis_img, grid_points, first_point, h_off, v_off = create_offset_grid_from_lines(
                    image_bgr, 
                    parallel_line_groups,
                    stran_lukenj,
                )
                
                # Print results
                if debug==1:
                    print("First point coordinates:", first_point)
                    print("Horizontal offset from vertical line:", h_off)
                    print("Vertical offset from horizontal line:", v_off)
                    print("\nFull 3x3 grid:")
                    print(grid_points)

            centers, threshold_img, edge_img = detect_bright_objects(slika)

            print(f"Found {len(centers)} bright objects at positions:")
            for i, center in enumerate(centers):
                print(f"Object {i+1}: {center}")

            print("PINS:")
            print(pin_centers[:])

            # Display edges and results
            cv.imshow("Detected Edges in ROI", edges)
            cv.imshow(f"Shape Detection in ROI (Area: {min_area}-{max_area})", result)
            cv.waitKey(0)
            cv.destroyAllWindows()
