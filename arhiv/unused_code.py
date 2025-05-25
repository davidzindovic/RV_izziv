
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
    if show_result and prikazi_vmesne_korake==1:
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
        if prikazi_vmesne_korake==1:
            cv.imshow("Detected Parallel Lines in ROI", display_image)
            cv.waitKey(0)
            cv.destroyAllWindows()
    
    return parallel_line_groups


#main:
                """
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
                """
