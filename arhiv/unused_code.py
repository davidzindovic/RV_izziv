

def get_transformacijska_matrika(image):
    """
    Pod transformacijsko matriko na podlagi spremembe pozicije ali orientacije glavne plošče
    
    Args:
        image: slika, na kateri želimo iskati oblike
        region_of_interest: območje na sliki, znotraj katerega želimo iskati oblike
        canny_low: spodnja meja za Canny algoritem za detekcijo robov
        canny_high: zgornja meja za Canny algoritem za detekcijo robov

        pins_centers,shapes, (x, y, w, h)
    Returns:
        pins_centers: Središča najdenih objektov (pinov v 3x3 mreži lukenj)
        shapes: Podrobne informacije (parametri kot so aspect ratio ipd.) najdene oblike
        (x, y, w, h): touple, ki hrani podatke območja, znotraj katerega smo iskali (x in y začetne točke, w=širina,h=višina)
    """

    slika_rob=image.copy()
    #roi = (150, 50, 200, 400)
    x0=150
    y0=50
    width=200
    height=400
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _,thr=cv.threshold(gray,200,255,cv.THRESH_BINARY)
    new_thr = thr[y0:y0+height, x0:x0+width].copy()
    canny= cv.Canny(new_thr, 80, 160, apertureSize=3)


    contours,_=cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    najmanjsa_povrsina=0
    n_p_index=0
    n_p_cnt=0
    for c in contours:
        #contours[n_p_cnt]=c + np.array([x0, y0])
        area=cv.contourArea(c)
        if area>najmanjsa_povrsina:
            najmanjsa_povrsina=area
            n_p_index=n_p_cnt
        n_p_cnt+=1
    
    rotrect = cv.minAreaRect(contours[n_p_index]+ np.array([x0, y0]))
    
    box = cv.boxPoints(rotrect)
    box = np.intp(box)
    
    sredisce_boarda=rotrect[0]
    kot_rotacije=rotrect[2]

    if debug_iskanje_tr_mat==1:
        print ("Boarda ima središče v: "+str(sredisce_boarda)+", zarotiran je pa za: "+str(kot_rotacije))

    # get center line from box
    # note points are clockwise from bottom right
    x1 = (box[0][0] + box[1][0]) // 2
    y1 = (box[0][1] + box[1][1]) // 2
    x2 = (box[2][0] + box[3][0]) // 2
    y2 = (box[2][1] + box[3][1]) // 2

    if debug_iskanje_tr_mat==1:
        # draw rotated rectangle on copy of img as result
        cv.drawContours(slika_rob, [box], 0, (0,0,255), 2)
        cv.line(slika_rob, (x1,y1), (x2,y2), (255,0,0), 2)

        cv.imshow("Sivinska v funk robov",gray)
        #cv.imshow("Upragovljanje v funk robov",new_thr)
        cv.imshow("Canny v funk robov",canny)
        cv.imshow("Najden board v funk robov",slika_rob)

        cv.waitKey(0)

    return sredisce_boarda,kot_rotacije

# Funkcija za iskanje transformacijske matrike potrebne zaradi
# premika main boarda (za preslikavo točk 3x3 grida)
def sprememba_robov(image):#transf matrika overkil? sort?
    global debug_robovi_kvadra
    global num_za_povp_robov, robovi, robovi_povp, old_robovi_povp

    repack_old_robovi_povp=[]
    repack_robovi_povp=[]
    transformacijska_matrika=[]

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


    blurred = cv.GaussianBlur(gray, (7, 7), 2)
    edges = cv.Canny(blurred, 80, 160, apertureSize=3)
    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80, minLineLength=50, maxLineGap=60)

    # Visualize raw lines (for debugging)
    line_img = np.zeros_like(image)

    local_list=[]
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1>150 and x2<350 and y1>50 and y2<350:
            cv.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            local_list.append([x1,y1,x2,y2])

            """
            if abs(x1-x2)<50 and abs(y1-y2)>100:
                vert.append((x1,y1,x2,y2))
                print("V angle: "+str(math.degrees(math.atan(abs((x2-x1)/(y2-y1))))))
                cv.line(line_kvad, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if 200>abs(x1-x2)>100 and abs(y1-y2)<20:
                hor.append((x1,y1,x2,y2))
                print("H angle: "+str(math.degrees(math.atan(abs((y2-y1)/(x2-x1))))))
                cv.line(line_kvad, (x1, y1), (x2, y2), (255, 0, 0), 2)
            """


    robovi.append(local_list)

    if debug_robovi_kvadra==1:
        cv.imshow("Raw Hough Lines", line_img)  # Debug window
        cv.waitKey(0)
    num_za_povp_robov+=1
    move_radius=5
    max_frames_za_povp=3

    if num_za_povp_robov==max_frames_za_povp:
        num_za_povp_robov=0
        robovi_povp=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        for st_zajema in range(len(robovi)):
            for r in range((len(robovi[st_zajema]))):
                for r_l in range(len(robovi[st_zajema][r])):
                    for r_primerjava in range(len(robovi[st_zajema])):
                        if r_primerjava!=r:
                            for r_x in range(len(robovi[st_zajema][r_primerjava])):
                                if (r_l==0 or r_l==2) and (r_x==0 or r_x==2) and r<len(robovi_povp):
                                    if (robovi[st_zajema][r][r_l]+move_radius)>robovi[st_zajema][r_primerjava][r_x]>(robovi[st_zajema][r][r_l]-move_radius):
                                        robovi_povp[r][r_l]+=robovi[st_zajema][r_primerjava][r_x]

                                if (r_l==1 or r_l==3) and (r_x==1 or r_x==3) and r<len(robovi_povp):
                                    if (robovi[st_zajema][r][r_l]+move_radius)>robovi[st_zajema][r_primerjava][r_x]>(robovi[st_zajema][r][r_l]-move_radius):
                                        robovi_povp[r][r_l]+=robovi[st_zajema][r_primerjava][r_x]

        for a in range(len(robovi_povp)):
            for b in range(len(robovi_povp[a])):
                robovi_povp[a][b]/=max_frames_za_povp
        
        cifra=0
        old_vsota=0
        for vsota_1 in range(len(old_robovi_povp)):
            for vsota_2 in range(len(old_robovi_povp[vsota_1])):
                old_vsota+=old_robovi_povp[vsota_1][vsota_2]
                cifra+=1
        old_vsota=old_vsota/cifra

        cifra=0
        vsota=0
        for vsota_1 in range(len(robovi_povp)):
            for vsota_2 in range(len(robovi_povp[vsota_1])):
                vsota+=robovi_povp[vsota_1][vsota_2]
                cifra+=1
        vsota=vsota/cifra

        if len(old_robovi_povp)>0 and old_vsota>0:
            for re in range(len(old_robovi_povp)):
                for re_index in range(int(len(old_robovi_povp[re])/2)):
                    repack_old_robovi_povp.append([old_robovi_povp[re][2*re_index],old_robovi_povp[re][2*re_index+1]])
            repack_old_robovi_povp=repack_old_robovi_povp[0:3]
            repack_old_robovi_povp=np.array(repack_old_robovi_povp)
            repack_old_robovi_povp=np.float32(repack_old_robovi_povp)

        if len(robovi_povp)>0:
            for re in range(len(robovi_povp)):
                for re_index in range(int(len(robovi_povp[re])/2)):
                    repack_robovi_povp.append([robovi_povp[re][2*re_index],robovi_povp[re][2*re_index+1]])
            repack_robovi_povp=repack_robovi_povp[0:3]#točno 3 točke!
            repack_robovi_povp=np.array(repack_robovi_povp)
            repack_robovi_povp=np.float32(repack_robovi_povp)

        #print("stevilo elementov: "+str(len(robovi_povp))+" | "+str(len(old_robovi_povp)))
        #print("podatkovni tip:" +str(type(repack_old_robovi_povp))+" | "+str(type(repack_robovi_povp)))
        if len(robovi_povp)==len(old_robovi_povp) and len(old_robovi_povp)>0 and old_vsota>0 and vsota>0:
            transformacijska_matrika=cv.getAffineTransform(repack_old_robovi_povp,repack_robovi_povp)

        old_robovi_povp=robovi_povp
        robovi=[] 
        if debug_robovi_kvadra==1:
            print("Transformacijska matrika: ")
            print(transformacijska_matrika)
            print("Tocke robov prej:")
            print(repack_old_robovi_povp)
            print("tocke robov zdej:")
            print(repack_robovi_povp)
            print("----------------------")      
        return transformacijska_matrika
    
    old_robovi_povp=robovi_povp

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
