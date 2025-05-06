import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math

#conf file naj vsebuje:
#ime_videja1,ime_videja2
#zacetna_tocka_x,zacetna_tocka_y,sirina,visina <--roi točke
#direktorij_z_videji

# pin in bowl argumente popravi - ish

#logika:
#ne vidim pinov v bowlu? -> pobiranje
#->vidim pine v bowlu in sem prej pobiral? -> prenos pina
#->-> sem prej prenašal pin in je zdej nov pin zaznan v 3x3 gridu? -> odlaganje pina
#->->-> sem prej odložil pin in še vedno vidim pine v bowlu? -> prazna roka

#dodej logiko na koncu da če so vsi pini pobrani da prazna roka pomeni zadnji action -> od ___ do zadnji frame mozen

debug = 0
debug_prikaz=0          # za prikaz framea z zamudo 1
debug_outlines=0        # sam za prikaz obrob in shapeov ko najde nekaj
debug_pins=0            # za podatke o najdenih pinih
debug_sprotno_stanje=0  # za izpis/izris sprotnega stanja pinov
debug_video=0           # za izpis podatkov o videju (npr. ko ne more dobiti slike)

# seznam možnih stanj (informativno, neuporabljeno)
stanja=["pobiranje","prenos","odlaganje","prazna roka"]

# spremenljivka, ki beleži zadnjo izvedeno "akcijo" za logiko
last_action="prazna roka"

# trenutni .json file:
zgodovina=[]

# spremeljivka za shranjevanje zaključnega frame-a zadnje "akcije",
# saj se start/stop frame (številki) ponastavita po koncu akcije
last_action_frame=0

prikazi_vmesne_korake=0 # spremeljivka za omogočanje prikaza vmesnih korakov/stanj
                        # slike. Sprememljivko spreminja program


# funkcija za pridobitev slike iz videja na podlagi številke frame-a:
def get_video_frame(video_path, frame_number):
    """(Previous implementation remains the same)"""
    
    #PC doma:
    video_dir = 'C:\\Users\\David Zindović\\Desktop\\Fax-Mag\\RV\\izziv\\izziv main\\'
    
    #Laptop šola:
    #video_dir = 'C:\\Users\\Vegova\\Documents\\zindo\\'
    
    #Laptop osebni:
    #
    
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

# funkcija za zaznavanje osvetljenih lukenj za pine (krogci):
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
        center = [int(x), int(y)]
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
    if show_results and prikazi_vmesne_korake==1:
        cv.imshow("Original Image", image)
        cv.imshow("Thresholded (White Areas)", thresholded)
        cv.imshow("Edge Detection", edges)
        cv.imshow("Closed Edges", closed_edges)
        cv.imshow("Detected Bright Objects", visualization)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    return center_points, thresholded, edges

# funkcija za določanje oblike objektov v sliki 
# (pini, pini v bowlu iz enega/drugega pogleda):
def detect_shapes(image, region_of_interest=None, min_area=100, max_area=10000, canny_low=50, canny_high=150):

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
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1)
    #ZADNJI PARAMETER:CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE ,CHAIN_APPROX_TC89_L1 CHAIN_APPROX_TC89_KCOS 
    
    shapes = {
        'small_circles': [],
        'big_circles': [],
        'pins': [],
        'pin_in_bowl': [],
        'pin_in_bowl_2': [],
        'trapezoids': []
    }
    
    pins_centers=[]
    pins_in_bowl_centers=[]
    pins_in_bowl_2_centers=[]
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
        
        # Prepoznavanje oblik glede na pogoje:
        if (aspect_ratio > 1 and aspect_ratio<3 and 90 < area < 170 and circularity>0.3):
            shapes['pins'].append(shape_info)
            pins_centers.append([abs_cx,abs_cy])
            if debug_pins:
                print("pin: "+str(aspect_ratio)+"|"+str(area)+"|"+str(circularity)+"|"+str(radius))
        
        elif (170< area < 300 and circularity<0.7 and 10<radius<35):
            shapes['pin_in_bowl'].append(shape_info)
            pins_in_bowl_centers.append([abs_cx,abs_cy])
            if debug_pins:
                print("pin in bowl: "+str(aspect_ratio)+"|"+str(area)+"|"+str(circularity)+"|"+str(radius))
        
        # za kamero ki ima pin bowl bližje:
        elif (220<area<600 and radius>8):#20<area<600 and radius>8
            shapes['pin_in_bowl_2'].append(shape_info)
            pins_in_bowl_2_centers.append([abs_cx,abs_cy])
            if debug_pins:
                print("pin in bowl 2: "+str(aspect_ratio)+"|"+str(area)+"|"+str(circularity)+"|"+str(radius))

    if debug_pins:
        print("")


    return pins_in_bowl_2_centers,pins_in_bowl_centers,pins_centers,shapes, (x, y, w, h) if region_of_interest else None, edges

# funkcija za določanje barve pina:
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

# funkcija za pripravo slike za prikaz zaznanih oblik:
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
        'pin_in_bowl_2':(255, 0, 0),
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

# funkcija za določanje, če je na spodnji ali zgornji polovici
# slike 3x3 grid lukenj za pine:
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
    if show_result and prikazi_vmesne_korake==1:
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

    radij_tocnosti=20 # radij znotraj katerega je lahko sredisce najdenega pina, pri čemer je središče radija središče zaznane osvetljene luknje

    setup=0           # spremenljivka za setup - opravljanje kalibracije s pomočjo lučk v luknjah

    x=0               # oporna spremenljivka za bolj berljivo kodo
    y=1               # oporna spremenljivka za bolj berljivo kodo

    # tabela vstavljenih pinov (3x3 grid) gledano iz kamere,
    # ki ima luknje za vstavljanje na zgornji polovici slike
    inserted_pins_zgoraj=[[0,0,0],
                          [0,0,0],
                          [0,0,0]]
                   
    # tabela vstavljenih pinov (3x3 grid) gledano iz kamere,
    # ki ima luknje za vstavljanje na spodnji polovici slike
    inserted_pins_spodaj=[[0,0,0],
                          [0,0,0],
                          [0,0,0]]
                          
    frame_cnt=0
    
    # tabeli za shranjevanje zadnjega stanja tabel vstavljenih pinov:
    old_inserted_pins_zgoraj=[[0,0,0],[0,0,0],[0,0,0]]
    old_inserted_pins_spodaj=[[0,0,0],[0,0,0],[0,0,0]]
    
    # spremenljivka za spremljanje največjega števila vstavljenih pinov
    # (za fazo pobiranja iz lukenj)
    odvzemanje_tracker_max=0
    odvzemanje_tracker_max_2=0
    
    # spremenljivka za preprečevanje nepravilnih oz. predčasnih sprememb podatkov
    odvzemanje_flag_once=0

    # tabela pinov v posodici
    # (podatek iz slike kamere, ki ima posodico na spodnji polovici slike)
    posodica_pod_roko=[]
    
    # spremenljivki za spremljanje začetnega iz končnega frame-a,
    # ko je bowl s pin-i pokrit z roko
    start_frame_pokrivanja_bowla=0
    stop_frame_pokrivanja_bowla=0

    # spremenljivka za štetje pinov:
    pin_count=0
    
    # spremenljivka (string) za beleženje trenutne "akcije"
    action="vstavljanje"

    # spremenljivka za beleženje številke frame-a
    frame=0
    
    # spremenljivka za potek zanke:
    video="play"

    bowl_cnt=0

    action_assigned=False
    action_start_frame=0
    action_end_frame=0

# zanka za sprehod čez vse slike videja:
    #for frame in range(550):
    while video!="stop":
    
        #shranimo sliki iz obeh kamer izbranega poskusa:
        slika = get_video_frame("64210323_video_5", frame+1)
        slika_2 = get_video_frame("64210323_video_1", frame+1)
        
        if slika is None:
            if debug_video==1:
                print("Failed to load image, 1. video, frame: "+str(frame))
            video="stop"
            action="konec" # preprečimo obdelavo slike
        if slika_2 is None:
            if debug_video==1:
                print("Failed to load image, 2. video, frame: "+str(frame))
            video="stop"
            action="konec" # preprečimo obdelavo slike
            
        elif action!="konec":
            # Convert to BGR for processing
            image_bgr = cv.cvtColor(slika, cv.COLOR_RGB2BGR)
            image_bgr_2 = cv.cvtColor(slika_2, cv.COLOR_RGB2BGR)
            
            # pred začetkom je potrebno pridobiti podatke (iz obeh kamer)
            # glede središč osvetljenih lukenj
            if setup==0 or setup==2:
                centers, threshold_img, edge_img = detect_bright_objects(slika)
                centers_2, threshold_img_2, edge_img_2 = detect_bright_objects(slika_2)
                if len(centers)>=9 and len(centers_2)>=9: # pocaka da ima 9 ref tock na vsakem videju (ne vemo še katera je spodnja stran)
                    setup=1
            #print(str(len(centers))+"|"+str(len(centers_2))+"|"+str(setup))

            if setup==1 or setup==3:
                
                # Define search region (x, y, width, height)
                roi = (150, 50, 200, 400)
                
                # Set parameters
                min_area = 50
                max_area = 3000
                canny_low = 50
                canny_high = 150

                # Detect shapes within ROI
                pin_in_bowl_2_centers,pin_in_bowl_centers,pin_centers,shapes, roi_rect, edges = detect_shapes(image_bgr, 
                                                    region_of_interest=roi,
                                                    min_area=min_area,
                                                    max_area=max_area,
                                                    canny_low=canny_low,
                                                    canny_high=canny_high)
                
                pin_in_bowl_2_centers_2,pin_in_bowl_centers_2,pin_centers_2,shapes_2, roi_rect_2, edges_2 = detect_shapes(image_bgr_2, 
                                                    region_of_interest=roi,
                                                    min_area=min_area,
                                                    max_area=max_area,
                                                    canny_low=canny_low,
                                                    canny_high=canny_high)
                
                # Ko najdemo središča lukenj, lahko določimo kje ima
                # katera kamera ima 3x3 grid na spodnji/zgornji strani
                if setup==1 or setup==0:
                    stran_lukenj = detect_object_position_bias(image_bgr, shapes, show_result=True)
                    stran_lukenj_2 = detect_object_position_bias(image_bgr_2, shapes_2, show_result=True)

                    # po zaznavanju postavimo spremenljivko setup na primerno vrednost
                    if setup==1:
                        setup=3
                    else:
                        setup=2

                if prikazi_vmesne_korake==1 or debug_outlines==1:
                    # Visualize
                    result = visualize_detection(image_bgr, shapes, roi)
                    result_2 = visualize_detection(image_bgr_2, shapes_2, roi)

                if debug==1:
                    print(f"Found {len(centers)} bright objects at positions:")
                    for i, center in enumerate(centers):
                        print(f"Object {i+1}: {center}")

                    print("PIN centers:")
                    print(pin_centers)

                if prikazi_vmesne_korake==1:
                    # Display edges and results
                    cv.imshow("Detected Edges in ROI - cam1", edges)
                    cv.imshow(f"Shape Detection in ROI - cam1 (Area: {min_area}-{max_area})", result)

                    cv.imshow("Detected Edges in ROI - cam2", edges_2)
                    cv.imshow(f"Shape Detection in ROI - cam2 (Area: {min_area}-{max_area})", result_2)

                    cv.waitKey(0)
                    cv.destroyAllWindows()

                prikazi_vmesne_korake=0


                # na podlagi kamere odločimo katere točke za pine uporabiti oz. katera kamera gleda odložene pine
                # kamera, ki ima bowl blizje gleda vse shapes notri
                if stran_lukenj=="spodaj":
                    pin_centers_spodaj=pin_centers
                    pin_centers_zgoraj=pin_centers_2
                    centers_spodaj_temp=centers
                    centers_zgoraj_temp=centers_2
                    posodica_pod_roko=pin_in_bowl_2_centers_2+pin_in_bowl_centers_2
                elif stran_lukenj_2=="spodaj":
                    pin_centers_spodaj=pin_centers_2
                    pin_centers_zgoraj=pin_centers
                    centers_spodaj_temp=centers_2
                    centers_zgoraj_temp=centers
                    posodica_pod_roko=pin_in_bowl_2_centers+pin_in_bowl_centers

                # sortiranje sredinskih točk lukenj za pine:
                centers_spodaj_temp.sort()
                centers_zgoraj_temp.sort()
                # opozorilo: .sort v 2D seznamu sortira glede na velikost prve koordinate v paru

                centers_spodaj_final=[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
                centers_zgoraj_final=[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
                
                s_max_spodaj=0          # največja y koordinata točk sredin lukenj (3x3 grid-a)
                s_min_spodaj=100000     # najmanjša y koordinata točk sredin lukenj
                s_mid_spodaj=0          # srednja y koordinata točk sredin lukenj
                
                # določanje primerne velikosti s_max,s_min,s_mid:
                for s in range(len(centers_spodaj_temp)):
                    if centers_spodaj_temp[s][1]>s_max_spodaj:
                        s_max_spodaj=centers_spodaj_temp[s][1]
                    if centers_spodaj_temp[s][1]<s_min_spodaj:
                        s_min_spodaj=centers_spodaj_temp[s][1]
                    if (s_max_spodaj-5)>centers_spodaj_temp[s][1]>(s_min_spodaj+5):
                        s_mid_spodaj=centers_spodaj_temp[s][1]
                # opozorilo: bi lahko bilo izven main loopa?

                s_max_zgoraj=0
                s_min_zgoraj=100000
                s_mid_zgoraj=0
                
                for s in range(len(centers_zgoraj_temp)):
                    if centers_zgoraj_temp[s][1]>s_max_zgoraj:
                        s_max_zgoraj=centers_zgoraj_temp[s][1]
                    if centers_zgoraj_temp[s][1]<s_min_zgoraj:
                        s_min_zgoraj=centers_zgoraj_temp[s][1]
                    if (s_max_zgoraj-5)>centers_zgoraj_temp[s][1]>(s_min_zgoraj+5):
                        s_mid_zgoraj=centers_zgoraj_temp[s][1]

                # spremenljivke za indexiranje
                top=0
                mid=0
                bottom=0 

                # določanje razporeditve točke (primerno glede na pozicijo kamere)
                for ss in range(len(centers_spodaj_temp)):

                    if (s_max_spodaj+5)>(centers_spodaj_temp[ss][1])>(s_max_spodaj-5):
                        #print("bottom: "+str(bottom)+"|"+str((s_max+5))+">"+str(centers_temp[ss][1])+">"+str(s_max-5))
                        centers_spodaj_final[bottom+6]=centers_spodaj_temp[ss] # zadnji 3je
                        bottom=bottom+1
                    elif (s_mid_spodaj+5)>(centers_spodaj_temp[ss][1])>(s_mid_spodaj-5):
                        #print("mid: "+str(mid)+"|"+str((s_mid+5))+">"+str(centers_temp[ss][1])+">"+str(s_mid-5))
                        centers_spodaj_final[mid+3]=centers_spodaj_temp[ss] # srednji 3je
                        mid=mid+1
                    elif (s_min_spodaj+5)>(centers_spodaj_temp[ss][1])>(s_min_spodaj-5):
                        #print("top: "+str(top)+"|"+str((s_min+5))+">"+str(centers_temp[ss][1])+">"+str(s_min-5))
                        centers_spodaj_final[top+0]=centers_spodaj_temp[ss] # prvi 3je
                        top=top+1

                # spremenljivke za indexiranje
                top=0
                mid=0
                bottom=0 
                
                for ss in range(len(centers_zgoraj_temp)):
                    
                    if (s_max_zgoraj+5)>(centers_zgoraj_temp[ss][1])>(s_max_zgoraj-5):
                        #print("bottom: "+str(bottom)+"|"+str((s_max+5))+">"+str(centers_temp[ss][1])+">"+str(s_max-5))
                        centers_zgoraj_final[2-bottom]=centers_zgoraj_temp[ss] # zadnji 3je
                        bottom=bottom+1
                    elif (s_mid_zgoraj+5)>(centers_zgoraj_temp[ss][1])>(s_mid_zgoraj-5):
                        #print("mid: "+str(mid)+"|"+str((s_mid+5))+">"+str(centers_temp[ss][1])+">"+str(s_mid-5))
                        centers_zgoraj_final[5-mid]=centers_zgoraj_temp[ss] # srednji 3je
                        mid=mid+1
                    elif (s_min_zgoraj+5)>(centers_zgoraj_temp[ss][1])>(s_min_zgoraj-5):
                        #print("top: "+str(top)+"|"+str((s_min+5))+">"+str(centers_temp[ss][1])+">"+str(s_min-5))
                        centers_zgoraj_final[8-top]=centers_zgoraj_temp[ss] # prvi 3je
                        top=top+1

                # procesiranje slik izvedemo le, če je setup dokončno opravljen:
                if setup==3:
                    
                    # v primeru odvzemanje se v vsakem ciklu ponastavi seznam vstavljenih pinov
                    if action=="odvzemanje" and len(pin_centers_spodaj)>0:
                        inserted_pins_spodaj=[[0,0,0],[0,0,0],[0,0,0]]
                        
                    # za vsak vstavljen pin najdemo primerno točko iz 3x3 grid-a lukenj:
                    for pins in range(len(pin_centers_spodaj)):
                        tracking_cnt=0
                        for luknje in range(len(centers_spodaj_final)):
                            pin_index=pins
                            if pin_centers_spodaj[pin_index][x]>(centers_spodaj_final[luknje][x]-radij_tocnosti/2) and pin_centers_spodaj[pin_index][x]<(centers_spodaj_final[luknje][x]+radij_tocnosti/2) and pin_centers_spodaj[pin_index][y]>(centers_spodaj_final[luknje][y]-radij_tocnosti/2) and pin_centers_spodaj[pin_index][y]<(centers_spodaj_final[luknje][y]+radij_tocnosti/2) :

                                if inserted_pins_spodaj[math.floor(luknje/3)][math.floor(luknje%3)]!=1:
                                    inserted_pins_spodaj[math.floor(luknje/3)][math.floor(luknje%3)]=1
                                
                    if action=="odvzemanje" and len(pin_centers_zgoraj)>0:
                        inserted_pins_zgoraj=[[0,0,0],[0,0,0],[0,0,0]]
                    for pins in range(len(pin_centers_zgoraj)):
                        tracking_cnt=0
                        for luknje in range(len(centers_zgoraj_final)):
                            pin_index=pins
                            
                            if pin_centers_zgoraj[pin_index][x]>(centers_zgoraj_final[luknje][x]-radij_tocnosti/2) and pin_centers_zgoraj[pin_index][x]<(centers_zgoraj_final[luknje][x]+radij_tocnosti/2) and pin_centers_zgoraj[pin_index][y]>(centers_zgoraj_final[luknje][y]-radij_tocnosti/2) and pin_centers_zgoraj[pin_index][y]<(centers_zgoraj_final[luknje][y]+radij_tocnosti/2) :
                                luknje=len(centers_zgoraj_final)-1-luknje
                                if inserted_pins_zgoraj[math.floor(luknje/3)][math.floor(luknje%3)]!=1:
                                    inserted_pins_zgoraj[math.floor(luknje/3)][math.floor(luknje%3)]=1
                               
                    

                    sprememba=0
                    for i in range(9):
                        if inserted_pins_spodaj[math.floor(i/3)][math.floor(i%3)]>old_inserted_pins_spodaj[math.floor(i/3)][math.floor(i%3)]:
                            sprememba=1

                    if sprememba==1 and action=="vstavljanje":
                    #if (inserted_pins!=old_inserted_pins):
                        #if debug_vstavljanje==1:

                        if debug_sprotno_stanje==1:
                            print("-----------------")
                            print("Frame:" + str(frame))
                            print("faza: vstavljanje")
                            for c in range(3):
                                print(inserted_pins_zgoraj[c])
                            print("~~~~~~~~~~~~~~~")
                            for c in range(3):
                                print(inserted_pins_spodaj[c])
                            print("-----------------")
                        
                        if action_assigned==False:
                            last_action="odlaganje"

                            action_start_frame=last_action_frame+1
                            action_end_frame=frame-1
                            if action_end_frame<action_start_frame:
                                action_end_frame=action_start_frame

                            zgodovina.append(["prenos",action_start_frame,action_end_frame])#,"a",last_action_frame,start_frame_pokrivanja_bowla

                            last_action_frame=action_end_frame

                            action_start_frame=last_action_frame+1
                            action_end_frame=frame
                            if action_end_frame<action_start_frame:
                                action_end_frame=action_start_frame

                            zgodovina.append([last_action,action_start_frame,action_end_frame])#,"a",last_action_frame,start_frame_pokrivanja_bowla

                            last_action_frame=action_end_frame
                            action_assigned=True

                        if debug_prikaz==1:
                            prikazi_vmesne_korake=1
                        
                    #   if action=="odvzemanje":
                    #       old_inserted_pins=np.logical_and(old_inserted_pins,inserted_pins)
                    #   if action=="vstavljanje":
                        old_inserted_pins_spodaj=np.logical_or(old_inserted_pins_spodaj,inserted_pins_spodaj)
                        old_inserted_pins_zgoraj=np.logical_or(old_inserted_pins_zgoraj,inserted_pins_zgoraj)
                        
                        if debug_outlines==1:
                            cv.imshow("Detected Edges in ROI - cam1", edges)
                            cv.imshow(f"Shape Detection in ROI - cam1 (Area: {min_area}-{max_area})", result)

                            cv.imshow("Detected Edges in ROI - cam2", edges_2)
                            cv.imshow(f"Shape Detection in ROI - cam2 (Area: {min_area}-{max_area})", result_2)

                            cv.waitKey(0)
                            cv.destroyAllWindows()

                        sprememba=0
                    
                    if action=="vstavljanje":
                        pin_count=0
                    elif action=="odvzemanje":
                        pin_count=9
                
                    for p in range(9):
                        if inserted_pins_spodaj[math.floor(p/3)][math.floor(p%3)]==1 and action=="vstavljanje":
                            pin_count=pin_count+1
                        elif inserted_pins_spodaj[math.floor(p/3)][math.floor(p%3)]==0 and action=="odvzemanje":
                            pin_count=pin_count-1
                    

                    if action=="odvzemanje" and pin_count>odvzemanje_tracker_max:
                        odvzemanje_flag_once=odvzemanje_flag_once+1
                        if odvzemanje_flag_once>2:
                            odvzemanje_tracker_max=pin_count
                            odvzemanje_flag_once=0
                    elif pin_count<odvzemanje_tracker_max:
                        sprememba=1
                        odvzemanje_tracker_max=pin_count
                        #old_inserted_pins=np.logical_and(old_inserted_pins,inserted_pins)

                    if sprememba==1 and action=="odvzemanje":
                        if debug_prikaz==1:
                            prikazi_vmesne_korake=1
                        if debug_sprotno_stanje==1:
                            print("-----------------")
                            print("frame: "+str(frame))
                            print("faza: odvzemanje")
                            #print("stanje: "+str(pin_count)+"|"+str(odvzemanje_tracker_max))
                            for c in range(3):
                                print(inserted_pins_zgoraj[c])
                            print("~~~~~~~~~~~~~~~")
                            for cd in range(3):
                                print(inserted_pins_spodaj[cd])
                            print("-----------------")

                        if action_assigned==False:
                            last_action="pobiranje"

                            action_start_frame=last_action_frame+1
                            action_end_frame=frame-1
                            if action_end_frame<action_start_frame:
                                action_end_frame=action_start_frame

                            zgodovina.append(["prazna roka",action_start_frame,action_end_frame])#,"b",last_action_frame,start_frame_pokrivanja_bowla

                            last_action_frame=action_end_frame

                            action_start_frame=last_action_frame+1
                            action_end_frame=frame
                            if action_end_frame<action_start_frame:
                                action_end_frame=action_start_frame

                            zgodovina.append([last_action,action_start_frame,action_end_frame])#,"b",last_action_frame,start_frame_pokrivanja_bowla

                            last_action_frame=action_end_frame
                            action_assigned=True

                        if debug_outlines==1:
                            cv.imshow("Detected Edges in ROI - cam1", edges)
                            cv.imshow(f"Shape Detection in ROI - cam1 (Area: {min_area}-{max_area})", result)

                            cv.imshow("Detected Edges in ROI - cam2", edges_2)
                            cv.imshow(f"Shape Detection in ROI - cam2 (Area: {min_area}-{max_area})", result_2)

                            cv.waitKey(0)
                            cv.destroyAllWindows()
                    sprememba=0

                    # Logika za pokrivanje bowla z roko:
                    if len(posodica_pod_roko)==0 and start_frame_pokrivanja_bowla==0 and ((last_action=="odlaganje" and action=="vstavljanje")or(last_action=="pobiranje" and action=="odvzemanje")):
                        #bowl_cnt=bowl_cnt+1
                        #if bowl_cnt>2:
                        start_frame_pokrivanja_bowla=frame
                        #    bowl_cnt=0

                        #print("POKRIVA")
                        #stop_frame_pokrivanja_bowla=start_frame_pokrivanja_bowla
                    if len(posodica_pod_roko)!=0 and start_frame_pokrivanja_bowla!=0:
                        stop_frame_pokrivanja_bowla=frame#-1
                        if (stop_frame_pokrivanja_bowla-start_frame_pokrivanja_bowla)>2:
                            if debug==1:
                                print("roka pokriva bowl! start frame: "+str(start_frame_pokrivanja_bowla)+", stop frame: "+str(stop_frame_pokrivanja_bowla))
                            if action=="vstavljanje" and action_assigned==False:

                                action_start_frame=last_action_frame+1
                                action_end_frame=start_frame_pokrivanja_bowla#-1
                                if action_end_frame<action_start_frame:
                                    action_end_frame=action_start_frame

                                zgodovina.append(["prazna roka",action_start_frame,action_end_frame])#,"c",last_action_frame,start_frame_pokrivanja_bowla

                                last_action_frame=action_end_frame
                                last_action="pobiranje"
                                
                            elif action=="odvzemanje" and action_assigned==False:

                                action_start_frame=last_action_frame+1
                                action_end_frame=start_frame_pokrivanja_bowla#-1
                                if action_end_frame<action_start_frame:
                                    action_end_frame=action_start_frame

                                zgodovina.append(["prenos",action_start_frame,action_end_frame])#,"c",last_action_frame,start_frame_pokrivanja_bowla
                                
                                last_action_frame=action_end_frame
                                last_action="odlaganje"
                                
                            
                            if action_assigned==False:

                                action_start_frame=last_action_frame+1
                                action_end_frame=stop_frame_pokrivanja_bowla
                                if action_end_frame<action_start_frame:
                                    action_end_frame=action_start_frame

                                zgodovina.append([last_action,action_start_frame,action_end_frame])#,"c",last_action_frame,start_frame_pokrivanja_bowla
 
                                last_action_frame=action_end_frame
                                stop_frame_pokrivanja_bowla=0
                                start_frame_pokrivanja_bowla=0
                                action_assigned=True
                        else:
                            stop_frame_pokrivanja_bowla=0
                            start_frame_pokrivanja_bowla=0
                    
                    
                    
                    if pin_count==9 and action=="vstavljanje":
                        if debug_sprotno_stanje==1:
                            print("Vse pini so bili vstavljeni, frame: "+str(frame))
                        action="odvzemanje"
                    elif pin_count==0 and action=="odvzemanje" and len(pin_centers_spodaj)==0:
                        if debug_sprotno_stanje==1:
                            print("Vsi pini so bili pobrani, frame: "+str(frame))
                        action="konec"
                    elif action=="konec":
                        if debug_sprotno_stanje==1:
                            print("konec, frame: "+str(frame))

        frame=frame+1
        action_assigned=False
        
    print(zgodovina)
