import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import json
import config_izziv_main


#logika:
#ne vidim pinov v bowlu? -> pobiranje
#->vidim pine v bowlu in sem prej pobiral? -> prenos pina
#->-> sem prej prenašal pin in je zdej nov pin zaznan v 3x3 gridu? -> odlaganje pina
#->->-> sem prej odložil pin in še vedno vidim pine v bowlu? -> prazna roka

#to do popravek pri logiki: 
#1. kalibracija (koliko pinov je v bowlu in kje so)
#2. ko ne vidim pinov na znanih pozicijah->pobiranje
#3. vidim vsaj 7 pinov (n_pred_pobiranjem-1-1) in so v bowlu (glede na radij) -> prenašanje
#4. dobim info da je en pin odložen v 3x3 grid->konec odlaganja -> skalibriram bowl za ref točke spet
#5. ponvaljaj korake 2-4

#todo: zrihtaj spremembe robov in transf matriko (Ni NUJNO), zrihtaj barve (ni nujno)

#----------------iz configa--------------
debug = config_izziv_main.debug
debug_prikaz=config_izziv_main.debug_prikaz           
debug_outlines=config_izziv_main.debug_outlines            
debug_pins=config_izziv_main.debug_pins
debug_sprotno_stanje=config_izziv_main.debug_sprotno_stanje
debug_video=config_izziv_main.debug_video
debug_prikazi_vsak_frame=config_izziv_main.debug_prikazi_vsak_frame
debug_akcije=config_izziv_main.debug_akcije
debug_false_positive=config_izziv_main.debug_false_positive
debug_anotacija=config_izziv_main.debug_anotacija
debug_robovi_kvadra=config_izziv_main.debug_robovi_kvadra
debug_bowl=config_izziv_main.debug_bowl

path_do_videjev=config_izziv_main.path_do_videjev

ime_videja=config_izziv_main.ime_videja
ime_videja_2=config_izziv_main.ime_videja_2

pravilna_anotacija_path=config_izziv_main.pravilna_anotacija_path

ime_pravilna_anotacija=config_izziv_main.ime_pravilna_anotacija
ime_pravilna_anotacija_2=config_izziv_main.ime_pravilna_anotacija_2
#----------------------------------------

# seznam možnih stanj (informativno, neuporabljeno)
stanja=["prijemanje_pina","prenos_pina","odlaganje_pina","prazna_roka"]

# spremenljivka, ki beleži zadnjo izvedeno "akcijo" za logiko
last_action=""

# spremenljivka, ki hrani numericno vrednost zadnje akcije (v zaporedju kot so v seznamu)
last_action_num=1

# trenutni .json file:
zgodovina=[]

# robovi main boarda (za povprecenje robov vsakih 5 frameov):
robovi=[]
robovi_povp=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
old_robovi_povp=[]
num_za_povp_robov=0

# tabela za false positive (SW-real):
false_positive_tabela=[[0,0], #T-P,F-P     T=True,F=False
                       [0,0]] #F-N,T-N     P=Positive,N=Negative

# spremenljivke za false_positive_tabelo:
num_true_positive=0
num_false_positive=0
num_false_negative=0
num_true_negative=0

# spremenljivke za število pinov:
num_pins_verified=0
num_pins_hypo=0

# spremenljivka za število pinov v bowlu na podlagi števila postavljeni pinov:
num_pin_in_bowl=0

# spremeljivka za shranjevanje zaključnega frame-a zadnje "akcije",
# saj se start/stop frame (številki) ponastavita po koncu akcije
last_action_frame=0

prikazi_vmesne_korake=0 # spremeljivka za omogočanje prikaza vmesnih korakov/stanj
                        # slike. Sprememljivko spreminja program

# zastavica za razporeditev lukenj v 3x3 mrežo
razporedi_centre_lukenj=True

#zastavica, ki se proži ob premiku ali rotaciji plošče
premaknjen_board=False 

# spremenljivka, ki hrani podatek o središču bowla
center_bowla_blizje_kameri=[]

slika_height=0
slika_2_height=0

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
        return None, None
    
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    if frame_number < 0 or frame_number >= total_frames:
        if debug_video==1:
            print(f"Error: Frame {frame_number} is out of range (0-{total_frames-1})")
        cap.release()
        return None, None
    
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB),total_frames
    else:
        if debug_video==1:
            print(f"Error: Could not read frame {frame_number}")
        return None, None

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
def detect_shapes(image, region_of_interest=None, min_area=100, max_area=10000, canny_low=100, canny_high=120):

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

    blurred = cv.GaussianBlur(gray, (3, 3), 2)
    #edges = cv.Canny(blurred, 80, 160, apertureSize=3)

    #blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, canny_low, canny_high,apertureSize=3)
    
    # dodano iz interneta za odebelitev robov/omogočanje lažjega iskanja zaprtih kontur
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    dilated = cv.dilate(edges, kernel)

    # Use RETR_LIST to get all contours, not just external ones
    contours, _ = cv.findContours(dilated, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
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
    #pins_in_bowl_centers=[]
    #pins_in_bowl_2_centers=[]
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
                print("pin: "+str(aspect_ratio)+"|"+str(area)+"|"+str(circularity)+"|"+str(radius)+"|"+str(perimeter)+"|"+str(color_name))

    if debug_pins:
        print("")

    return pins_centers,shapes, (x, y, w, h) if region_of_interest else None, edges

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

def get_transformacijska_matrika(image):
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

    print ("Boarda ima središče v: "+str(sredisce_boarda)+", zarotiran je pa za: "+str(kot_rotacije))

    # get center line from box
    # note points are clockwise from bottom right
    x1 = (box[0][0] + box[1][0]) // 2
    y1 = (box[0][1] + box[1][1]) // 2
    x2 = (box[2][0] + box[3][0]) // 2
    y2 = (box[2][1] + box[3][1]) // 2

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

# Funkcija za iskanje pinov v bowlu
def detect_dark_objects(image):
    # Read the image
    og=image.copy()
    if og is None:
        print("Error: Could not read the image.")
        return
    
    # Convert ROI to grayscale
    gray_og = cv.cvtColor(og, cv.COLOR_BGR2GRAY)
    
    # Enhanced contrast for dark regions
    alpha = 2  # Contrast control (1.0-3.0)
    beta = -100    # Brightness control (-100 to 100)

    enhanced_og = cv.convertScaleAbs(gray_og, alpha=alpha, beta=beta)

    blurred_og=cv.GaussianBlur(enhanced_og, (5, 5), 0)

    _,thresh_og=cv.threshold(blurred_og,254,255,cv.THRESH_BINARY)

    if debug_pins==1:
        cv.imshow("Bowl threshold",thresh_og)

    return thresh_og

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
    if top_count >= bottom_count:
        result = 'spodaj'
    elif bottom_count > top_count:
        result = 'zgoraj'
    #else:
    #    result = 'sredina'
    
    # Visualization
    if show_result and (prikazi_vmesne_korake==1 or debug_prikaz==1):
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

# funkcija za vpis anotacije v json datoteko:
def create_annotation_json(filename, annotations_array):
    """
    Creates a JSON file with video annotations.
    
    Args:
        filename (str): The name of the output JSON file (without extension)
        annotations_array (list): List of lists where each inner list contains
                                 [frame_start, frame_stop, event]
    """
    # Prepare the data structure
    data = {
        "video_name": f"{filename}.mp4",
        "annotations": []
    }
    
    # Populate the annotations
    for annotation in annotations_array:
        if len(annotation) != 3:
            continue  # skip invalid entries
            
        event, frame_start, frame_stop = annotation
        data["annotations"].append({
            "frame_start": frame_start,
            "frame_stop": frame_stop,
            "event": event
        })
    
    # Create the filename with .json extension
    json_filename = f"{filename}.json"
    
    # Write to file (overwrite if exists)
    with open(json_filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    return json_filename

def luknje_v_3x3_gridu_POV(centers_spodaj_temp,centers_zgoraj_temp):

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
            centers_spodaj_final[bottom+6]=centers_spodaj_temp[ss] # zadnji 3je
            bottom=bottom+1
        elif (s_mid_spodaj+5)>(centers_spodaj_temp[ss][1])>(s_mid_spodaj-5):
            centers_spodaj_final[mid+3]=centers_spodaj_temp[ss] # srednji 3je
            mid=mid+1
        elif (s_min_spodaj+5)>(centers_spodaj_temp[ss][1])>(s_min_spodaj-5):
            centers_spodaj_final[top+0]=centers_spodaj_temp[ss] # prvi 3je
            top=top+1

    # spremenljivke za indexiranje
    top=0
    mid=0
    bottom=0 

    for ss in range(len(centers_zgoraj_temp)):
        
        if (s_max_zgoraj+5)>(centers_zgoraj_temp[ss][1])>(s_max_zgoraj-5):
            centers_zgoraj_final[2-bottom]=centers_zgoraj_temp[ss] # zadnji 3je
            bottom=bottom+1
        elif (s_mid_zgoraj+5)>(centers_zgoraj_temp[ss][1])>(s_mid_zgoraj-5):
            centers_zgoraj_final[5-mid]=centers_zgoraj_temp[ss] # srednji 3je
            mid=mid+1
        elif (s_min_zgoraj+5)>(centers_zgoraj_temp[ss][1])>(s_min_zgoraj-5):
            centers_zgoraj_final[8-top]=centers_zgoraj_temp[ss] # prvi 3je
            top=top+1
    
    return centers_spodaj_final,centers_zgoraj_final,

def primerjava_anotacij(fresh_json,correct_json):
    # Opening JSON file
    file_generiran = open(fresh_json)
    file_pravilen = open(correct_json)

    # returns JSON object as a dictionary
    json_generiran = json.load(file_generiran)
    json_pravilen = json.load(file_pravilen)

    # Iterating through the json list
    podatki_generirani=json_generiran['annotations']
    podatki_pravilni=json_pravilen['annotations']

    toleranca_faljenega_framea=20
    p=0
    num_true_positive=0
    num_false_positive=0
    num_true_negative=0
    num_false_negative=0


    for g in range(len(podatki_generirani)):
        tp=False
        fp=False
        tn=False
        fn=False
        #for action in ["frame_start","frame_stop"]:
        while p<len(podatki_pravilni) and (tp==False and fp==False and tn==False and fn==False):
            akcija_g=podatki_generirani[g].get("event")
            akcija_p=podatki_pravilni[p].get("event")
            if (int(podatki_generirani[g].get("frame_start"))-toleranca_faljenega_framea)<=int(podatki_pravilni[p].get("frame_start")) and (int(podatki_generirani[g].get("frame_stop"))+toleranca_faljenega_framea)>=int(podatki_pravilni[p].get("frame_stop")) and (akcija_g==akcija_p):
                tp=True
            #else:
            #    tn=True
            p=p+1
        if tp==False:
            tn=True

        if tp==True:
            num_true_positive+=1
        elif fp==True:
            num_false_positive+=1
        elif tn==True:
            num_true_negative+=1
        elif fn==True:
            num_false_negative+=1

        tp=False
        fp=False
        tn=False
        fn=False

    print("Analiza anotacije:")
    print("                Pravilna anotacija")
    print("                ------------------")
    print("               |Positive|Negative")
    print("Moja     |True |"+str(num_true_positive)+" | "+str(num_true_negative))
    print("Anotacija|False|"+str(num_false_positive)+" | "+str(num_false_negative))
    print("----------------------------------")

    num_true_positive=0
    num_false_positive=0
    num_true_negative=0
    num_false_negative=0

    # Closing file
    file_generiran.close()
    file_pravilen.close()

def primerjava_change_eventov(fresh_json,correct_json):
    # Opening JSON file
    file_generiran = open(fresh_json)
    file_pravilen = open(correct_json)

    # returns JSON object as a dictionary
    json_generiran = json.load(file_generiran)
    json_pravilen = json.load(file_pravilen)

    # Iterating through the json list
    podatki_generirani=json_generiran['annotations']
    podatki_pravilni=json_pravilen['annotations']

    toleranca_faljenega_framea=20

    num_true_positive=0
    num_false_positive=0
    num_true_negative=0
    num_false_negative=0
    
    
    # sprememba eventa mora biti enaka (in okviren time_frame)
    # TP - res se je zgodila sprememba event
    # TN - ni se zgodila sprememba program pa misli da se je
    # FP - zgodila se je sprememba, program pa misli da se ni
    # FN - ni se zgodila sprememba, program prav tako misli da se ni


    for f in range(podatki_generirani[-1].get("frame_stop")):   #moramo najti vsak podatek za vsak frame

        event_log_generiran=[0,0,False,"",""] #prejsnji_frame,zdejsnji_frame,a_se_je_zgodila_sprememba,prejsnji_event,zdejsnji event
        event_log_pravilen=[0,0,False,"",""]

        current_frame=f+1

        event_log_generiran[0]=current_frame-1
        event_log_generiran[1]=current_frame

        event_log_pravilen[0]=current_frame-1
        event_log_pravilen[1]=current_frame

        event_found=False
        event_cnt=0
        while not event_found and event_cnt<len(podatki_generirani) and event_cnt<len(podatki_pravilni):
            if (current_frame-1)>=podatki_generirani[event_cnt].get("frame_start") and (current_frame-1)<=podatki_generirani[event_cnt].get("frame_stop") and event_log_generiran[3]=="":
                event_log_generiran[3]=podatki_generirani[event_cnt].get("event")
            if (current_frame)>=podatki_generirani[event_cnt].get("frame_start") and (current_frame)<=podatki_generirani[event_cnt].get("frame_stop") and event_log_generiran[4]=="":
                event_log_generiran[4]=podatki_generirani[event_cnt].get("event")
            
            if (current_frame-1)>=podatki_pravilni[event_cnt].get("frame_start") and (current_frame-1)<=podatki_pravilni[event_cnt].get("frame_stop") and event_log_pravilen[3]=="":
                event_log_pravilen[3]=podatki_pravilni[event_cnt].get("event")
            if (current_frame)>=podatki_pravilni[event_cnt].get("frame_start") and (current_frame)<=podatki_pravilni[event_cnt].get("frame_stop") and event_log_pravilen[4]=="":
                event_log_pravilen[4]=podatki_pravilni[event_cnt].get("event")

            event_cnt+=1

            if event_log_generiran[3]!="" and event_log_generiran[4]!="" and event_log_pravilen[3]!="" and event_log_pravilen[4]!="":
                event_found=True

        #program misli da se je zgodila sprememba in se je v resnici:
        if event_log_generiran[3]!=event_log_generiran[4] and event_log_pravilen[3]!=event_log_pravilen[4]:#and event_log_generiran[3]==event_log_pravilen[3] and event_log_generiran[4]==event_log_pravilen[4]
            num_true_positive+=1
        
        #program misli da se je zgodila sprememba in se ni v resnici:  
        elif event_log_generiran[3]!=event_log_generiran[4] and event_log_pravilen[3]==event_log_pravilen[4]:
            num_true_negative+=1

        #program misli da se ni zgodila sprememba in se je v resnici:  
        elif event_log_generiran[3]==event_log_generiran[4] and event_log_pravilen[3]!=event_log_pravilen[4]:
            num_false_positive+=1

        #program misli da se ni zgodila sprememba in se ni v resnici:  
        elif event_log_generiran[3]==event_log_generiran[4] and event_log_pravilen[3]==event_log_pravilen[4]:
            num_false_negative+=1


    print("Analiza sprememb eventov:")
    print("")
    print("                Pravilna anotacija")
    print("                ------------------")
    print("               |Positive|Negative")
    print("Moja     |True |"+str(num_true_positive)+" | "+str(num_true_negative))
    print("Anotacija|False|"+str(num_false_positive)+" | "+str(num_false_negative))
    print("----------------------------------")

    # Closing file
    file_generiran.close()
    file_pravilen.close()
            
if __name__ == "__main__":

    radij_tocnosti=20 # radij znotraj katerega je lahko sredisce najdenega pina, pri čemer je središče radija središče zaznane osvetljene luknje

    setup=0           # spremenljivka za setup - opravljanje kalibracije s pomočjo lučk v luknjah

    x=0               # oporna spremenljivka za bolj berljivo kodo
    y=1               # oporna spremenljivka za bolj berljivo kodo
 
    frame_cnt=0
    
    inserted_pins_zgoraj=[[0,0,0],[0,0,0],[0,0,0]]
    inserted_pins_spodaj=[[0,0,0],[0,0,0],[0,0,0]]

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
    
    # spremenljivka za belezenje  globalnega najvecjega, začasnega najvecjega in trenutnega 
    # stevila zaznanih objektov v posodici v za zadnjih 5 frame-ih
    stevilo_objektov_v_posodici_max=-1
    stevilo_objektov_v_posodici_max_tmp=0
    stevilo_objektov_v_posodici=0
    avrg_posodica=1
    avrg_posodica_backup=1

    # spremenljivka za prvo določitev vrednosti max za posodico
    posodica_flag=False

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

    radij_bowla=0
    pin_in_bowl_data_flag=True
    number_of_pins_in_bowl_ago=0
    number_of_pins_in_bowl_ago_ago=0
    bowl_factor_osnova=3500/9
    bowl_factor=0 # "površina" enega pina
    zeljeno_stevilo_bowl_pinov=9 # koliko pinov pričakujemo kot naslednji pravilni odgovor
    zeljeno_stevilo_tracker=0   # spremenljivka za spremljanje koliko frameov smo imeli željeno število pinov v bowlu
    trenutno_stevilo_pinov_v_bowlu=0 # potrjeno število pinov
    current_state="ni roka"

    actual_dolzina_videja=0

    action_assigned=False
    action_start_frame=0
    action_end_frame=0


# zanka za sprehod čez vse slike videja:
    #for frame in range(550):
    while video!="stop":
    
        #shranimo sliki iz obeh kamer izbranega poskusa:
        slika,dolzina_videja = get_video_frame(ime_videja, frame+1)
        slika_2,dolzina_videja_2 = get_video_frame(ime_videja_2, frame+1)
        
        if dolzina_videja is not None and dolzina_videja_2 is not None:
            if dolzina_videja>dolzina_videja_2:
                actual_dolzina_videja=dolzina_videja_2
            elif dolzina_videja<=dolzina_videja_2:
                actual_dolzina_videja=dolzina_videja

        if debug_prikazi_vsak_frame==1:
            cv.imshow(slika)
            cv.imshow(slika_2)

        if slika is not None and slika_2 is not None:
            slika_height, slika_width = slika.shape[:2]
            slika_2_height, slika_2_width = slika_2.shape[:2]
            slikaaa=slika_2.copy()
        
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

            if pin_in_bowl_data_flag==True:
                po_thr=detect_dark_objects(slika)
                po_thr_2=detect_dark_objects(slika_2)
                pin_in_bowl_data_flag=False

            #poiščemo transformacijski matriki obej kamer za preslikavo točk bowla in lukenj za pine
            #na podlagi robov main boarda
            #tr_mat=sprememba_robov(slika)
            #tr_mat_2=sprememba_robov(slika_2)

            # preizkusi kaj se zgodi če je roka nad boardom
            tr_mat=get_transformacijska_matrika(slika)
            tr_mat_2=get_transformacijska_matrika(slika_2)

            if setup==1 or setup==3:

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
                
                pin_centers_2,shapes_2, roi_rect_2, edges_2 = detect_shapes(image_bgr_2, 
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

                    if stran_lukenj!='none'or stran_lukenj_2!='none':
                        setup=3

                        if stran_lukenj=="spodaj":
                            stran_lukenj_2="zgoraj"
                        elif stran_lukenj=="zgoraj":
                            stran_lukenj_2="spodaj"
                        elif stran_lukenj_2=="spodaj":
                            stran_lukenj="zgoraj"
                        elif stran_lukenj_2=="zgoraj":
                            stran_lukenj="spodaj"

                # procesiranje slik izvedemo le, če je setup dokončno opravljen:
                if setup==3:
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

                    if prikazi_vmesne_korake==1 or debug_prikazi_vsak_frame==1:
                        # Display edges and results
                        cv.imshow("Edges | Frame: "+str(frame) +" - cam1", edges)
                        cv.imshow("Shapes | Frame: "+str(frame)+" - cam1", result)

                        cv.imshow("Edges | Frame: "+str(frame) +" - cam2", edges_2)
                        cv.imshow("Shapes | Frame: "+str(frame)+" - cam2", result_2)

                    prikazi_vmesne_korake=0

                    #-------------------------DEL kode za določanje stanja v 3x3 gridu------------------------
                    
                    # na podlagi kamere odločimo katere točke za pine uporabiti oz. katera kamera gleda odložene pine
                    # kamera, ki ima bowl blizje gleda vse shapes notri

                    if stran_lukenj=="spodaj":
                        pin_centers_spodaj=pin_centers
                        pin_centers_zgoraj=pin_centers_2
                        centers_spodaj_temp=centers
                        centers_zgoraj_temp=centers_2
                        #posodica_pod_roko=pin_in_bowl_2_centers_2
                        #main_pin_bowl_area=area_pin_bowl_2
                        main_thr=po_thr_2

                    elif stran_lukenj_2=="spodaj":
                        pin_centers_spodaj=pin_centers_2
                        pin_centers_zgoraj=pin_centers
                        centers_spodaj_temp=centers_2
                        centers_zgoraj_temp=centers
                        #posodica_pod_roko=pin_in_bowl_2_centers
                        #main_pin_bowl_area=area_pin_bowl
                        main_thr=po_thr

                    if razporedi_centre_lukenj==True:
                        #enkrat razporedimo centre lukenj v 3x3 grid
                        centers_spodaj_final,centers_zgoraj_final=luknje_v_3x3_gridu_POV(centers_spodaj_temp,centers_zgoraj_temp)
                        razporedi_centre_lukenj=False

                    if premaknjen_board==True:
                        #
                        # sem paše transormacija 3x3 grida lukenj s transf matriko

                        #dx=centers_zgoraj_final[1][0]-centers_zgoraj_final[4][0]
                        #dy=centers_zgoraj_final[1][1]-centers_zgoraj_final[4][1]
                        #radij_bowla=1.5*math.sqrt(math.pow(dx,2)+math.pow(dy,2))
                        #center_bowla_blizje_kameri=[centers_zgoraj_final[1][0]+dx,centers_zgoraj_final[1][1]+3.25*dy]

                        premaknjen_board=False

                    dx=centers_zgoraj_final[1][0]-centers_zgoraj_final[4][0]
                    dy=centers_zgoraj_final[1][1]-centers_zgoraj_final[4][1]
                    radij_bowla=1.5*math.sqrt(math.pow(dx,2)+math.pow(dy,2))
                    center_bowla_blizje_kameri=[centers_zgoraj_final[1][0]+dx,centers_zgoraj_final[1][1]+3.25*dy]
                        
                    
                    # prikaz najdenih pinov v bowlu in obseg bowla
                    if debug_bowl==1:
                        index=0
                        for a in centers_zgoraj_final:
                            cv.circle(slikaaa, (a[0], a[1]), 7, (0, 225, index*20), -1)
                            index+=1
                        cv.circle(slikaaa, (int(center_bowla_blizje_kameri[0]), int(center_bowla_blizje_kameri[1])), 3, (0, 0, 255), -1)
                        cv.circle(slikaaa, (int(center_bowla_blizje_kameri[0]), int(center_bowla_blizje_kameri[1])), int(radij_bowla), (0, 0, 255), 3)
                        cv.imshow("centr",slikaaa)
                    
                    #------------------------število pinov v bowlu glede na število črnih pikslov---------------------------
                    num_crni_piksli_bowl=0
                    #print(frame)
                    for y_coord in range(main_thr.shape[0]):
                        for x_coord in range(main_thr.shape[1]):
                            razdalja_centr_bowl_pin=math.sqrt(math.pow(x_coord-center_bowla_blizje_kameri[0],2)+math.pow(y_coord-center_bowla_blizje_kameri[1],2))
                            if razdalja_centr_bowl_pin<radij_bowla and main_thr[y_coord][x_coord]==0:
                                num_crni_piksli_bowl+=1
                                #cv.circle(main_thr, (y_coord, x_coord), 1, (0, 0, 255), 1)
                    stevilo_pinov_v_bowlu=math.floor(num_crni_piksli_bowl/(bowl_factor_osnova+bowl_factor*35))

                    # če jih ne najde 9 ampak se prikaže roka v napoto -> sprememba last action in željenega št pinov
                    if zeljeno_stevilo_bowl_pinov==9 and stevilo_pinov_v_bowlu>10 and action=="vstavljanje":
                        zeljeno_stevilo_bowl_pinov-=1
                        print("roka na startu. Frame:"+str(frame))

                    if stevilo_pinov_v_bowlu<10 and action=="vstavljanje":
                        if zeljeno_stevilo_bowl_pinov==stevilo_pinov_v_bowlu:
                            zeljeno_stevilo_tracker+=1
                            if zeljeno_stevilo_tracker==3:
                                zeljeno_stevilo_tracker=0
                                trenutno_stevilo_pinov_v_bowlu=stevilo_pinov_v_bowlu

                                if action=="vstavljanje" and trenutno_stevilo_pinov_v_bowlu==0:
                                    continue

                                if trenutno_stevilo_pinov_v_bowlu<5 and action=="vstavljanje":
                                    bowl_factor-=1

                                if action=="vstavljanje" and trenutno_stevilo_pinov_v_bowlu>0:
                                    zeljeno_stevilo_bowl_pinov-=1
                                
                                print("V bowlu je: "+str(trenutno_stevilo_pinov_v_bowlu)+" pinov. Zdaj je frame: "+str(frame)+". Action: "+action)
                    if action=="odvzemanje":
                        if stevilo_pinov_v_bowlu>10 and current_state=="ni roka":
                            current_state="roka"
                            if trenutno_stevilo_pinov_v_bowlu<9:
                                trenutno_stevilo_pinov_v_bowlu+=1
                                if trenutno_stevilo_pinov_v_bowlu<5:
                                    bowl_factor+=1
                                zeljeno_stevilo_bowl_pinov+=1
                                print("V bowlu je: "+str(trenutno_stevilo_pinov_v_bowlu)+" pinov. Zdaj je frame: "+str(frame)+". Action: "+action)
                        if stevilo_pinov_v_bowlu<10 and current_state=="roka":
                            current_state="ni roka"
                    elif action=="vstavljanje":
                        if stevilo_pinov_v_bowlu>10 and current_state=="ni roka":
                            current_state="roka"
                        if stevilo_pinov_v_bowlu<10 and current_state=="roka":
                            current_state="ni roka"
                    stevilo_objektov_v_posodici=trenutno_stevilo_pinov_v_bowlu

                    pin_in_bowl_data_flag=True
                    #--------------------------------------------------------------------------------------------------------

                    # v primeru odvzemanja se v vsakem ciklu ponastavi seznam vstavljenih pinov
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
                                
                    # zaznavanje ponovimo za kamero, ki ima 3x3 grid na zgornji polovici slike:
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

                    #primerjamo staro in trenutno stanje zaznanih pinov        
                    sprememba=0
                    for i in range(9):
                        if inserted_pins_spodaj[math.floor(i/3)][math.floor(i%3)]>old_inserted_pins_spodaj[math.floor(i/3)][math.floor(i%3)]:
                            sprememba=1

                    #za štetje pinov ponastavimo število pinov glede na fazo
                    if action=="vstavljanje":
                        pin_count=0
                    elif action=="odvzemanje":
                        pin_count=9
                
                    # preštejemo (oz. odštejemo) trenutno število pinov in shranimo
                    for p in range(9):
                        if inserted_pins_spodaj[math.floor(p/3)][math.floor(p%3)]==1 and action=="vstavljanje":
                            pin_count=pin_count+1  
                        elif inserted_pins_spodaj[math.floor(p/3)][math.floor(p%3)]==0 and action=="odvzemanje":
                            pin_count=pin_count-1
                    num_pins_verified=pin_count
                    
                    #pri odvzemanju spremljamo število pinov (ki ga spremenimo če pogoj drži nekaj frame-ov)
                    if action=="odvzemanje" and pin_count>odvzemanje_tracker_max:
                        odvzemanje_flag_once=odvzemanje_flag_once+1
                        if odvzemanje_flag_once>2:
                            odvzemanje_tracker_max=pin_count
                            odvzemanje_flag_once=0
                    elif pin_count<odvzemanje_tracker_max: # če je manjše število kot program pomni, se je zgodila sprememba in spremenilo največje število pinov
                        sprememba=1
                        odvzemanje_tracker_max=pin_count
                        #old_inserted_pins=np.logical_and(old_inserted_pins,inserted_pins)

                    # če se je zgodila sprememba in smo v fazi vstaljanja v 3x3 grid
                    if sprememba==1 and action=="vstavljanje":

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
                        
                        # če frame-u ni bila dodeljena akcija in to ni prvi pripis dogoka, nazadnje se je pa zgodilo pobiranje (1)
                        if action_assigned==False and len(zgodovina)!=0 and last_action_num==1:
                            last_action="odlaganje_pina"
                            num_pins_hypo+=1

                            action_start_frame=last_action_frame+1
                            action_end_frame=frame-1
                            if action_end_frame<action_start_frame:
                                action_end_frame=action_start_frame

                            zgodovina.append(["prenos_pina",action_start_frame,action_end_frame])
                            last_action_num=2

                            last_action_frame=action_end_frame

                            action_start_frame=last_action_frame+1
                            action_end_frame=frame
                            if action_end_frame<action_start_frame:
                                action_end_frame=action_start_frame

                            zgodovina.append([last_action,action_start_frame,action_end_frame])
                            last_action_num=3

                            last_action_frame=action_end_frame
                            action_assigned=True
                            # akcija je pripisana frameu, označen konec framea in zadnja akcija je (3- odlaganje)


                        if debug_prikaz==1:
                            prikazi_vmesne_korake=1

                        old_inserted_pins_spodaj=np.logical_or(old_inserted_pins_spodaj,inserted_pins_spodaj)
                        old_inserted_pins_zgoraj=np.logical_or(old_inserted_pins_zgoraj,inserted_pins_zgoraj)
                        
                        if debug_outlines==1:
                            cv.imshow("Edges | Frame: "+str(frame) +" - cam1", edges)
                            cv.imshow("Shapes | Frame: "+str(frame)+" - cam1", result)

                            cv.imshow("Edges | Frame: "+str(frame) +" - cam2", edges_2)
                            cv.imshow("Shapes | Frame: "+str(frame)+" - cam2", result_2)

                        sprememba=0
                    
                    # če se je zgodila sprememba in smo v fazi odvzemanja (iz 3x3 grida)
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

                        if action_assigned==False and len(zgodovina)!=0 and last_action_num==3:
                            last_action="prijemanje_pina"
                            num_pins_hypo-=1

                            action_start_frame=last_action_frame+1
                            action_end_frame=frame-1
                            if action_end_frame<action_start_frame:
                                action_end_frame=action_start_frame

                            if pin_count==0 and trenutno_stevilo_pinov_v_bowlu==9:#ce ni pinov vec v 3x3 gridu
                                action_end_frame=actual_dolzina_videja
                                print("Actual dolzina: "+str(actual_dolzina_videja)+". Frame: "+str(frame))

                            zgodovina.append(["prazna_roka",action_start_frame,action_end_frame])
                            last_action_num=4

                            last_action_frame=action_end_frame

                            action_start_frame=last_action_frame+1
                            action_end_frame=frame
                            if action_end_frame<action_start_frame:
                                action_end_frame=action_start_frame
                            
                            if pin_count!=0:
                                zgodovina.append([last_action,action_start_frame,action_end_frame])
                            last_action_num=1

                            last_action_frame=action_end_frame
                            action_assigned=True


                        if debug_outlines==1:
                            cv.imshow("Detected Edges in ROI - cam1", edges)
                            cv.imshow(f"Shape Detection in ROI - cam1 (Area: {min_area}-{max_area})", result)

                            cv.imshow("Detected Edges in ROI - cam2", edges_2)
                            cv.imshow(f"Shape Detection in ROI - cam2 (Area: {min_area}-{max_area})", result_2)

                    sprememba=0


                    if current_state=="roka" and start_frame_pokrivanja_bowla==0:
                        start_frame_pokrivanja_bowla=frame

                    print("V bowlu je: "+str(trenutno_stevilo_pinov_v_bowlu)+" pinov. Najdeno: "+str(stevilo_pinov_v_bowlu)+" pinov. Frame: "+str(frame)+" . Stanje roke: "+str(current_state)+" . Action: "+action+". Bowl factor: "+str(bowl_factor)+" .Start: "+str(start_frame_pokrivanja_bowla)+" , konec: "+str(stop_frame_pokrivanja_bowla))

                    if current_state=="ni roka" and start_frame_pokrivanja_bowla!=0:
                        stop_frame_pokrivanja_bowla=frame
                        if (stop_frame_pokrivanja_bowla-start_frame_pokrivanja_bowla)>1:
                            if debug==1:
                                print("roka pokriva bowl! start frame: "+str(start_frame_pokrivanja_bowla)+", stop frame: "+str(stop_frame_pokrivanja_bowla))
                            print("Gruntam: start frame: "+str(start_frame_pokrivanja_bowla)+", stop frame: "+str(stop_frame_pokrivanja_bowla))
                            # za začetek videja:
                            if len(zgodovina)==0 and action_assigned==False:

                                action_start_frame=1
                                action_end_frame=start_frame_pokrivanja_bowla
                                if action_end_frame<action_start_frame:
                                    action_end_frame=action_start_frame

                                zgodovina.append(["prazna_roka",action_start_frame,action_end_frame])
                                last_action_num=4
                                #print("konc uvoda: "+str(last_action_frame)+"  "+str(start_frame_pokrivanja_bowla))
                                last_action_frame=action_end_frame
                                last_action="prijemanje_pina"

                            elif action=="vstavljanje" and action_assigned==False and len(zgodovina)!=0 and last_action_num==3:

                                action_start_frame=last_action_frame+1
                                action_end_frame=start_frame_pokrivanja_bowla
                                if action_end_frame<action_start_frame:
                                    action_end_frame=action_start_frame

                                zgodovina.append(["prazna_roka",action_start_frame,action_end_frame])
                                last_action_num=4

                                last_action_frame=action_end_frame
                                last_action="prijemanje_pina"
                            
                            elif action=="odvzemanje" and len(zgodovina)!=0 and trenutno_stevilo_pinov_v_bowlu==9:
                                action="konec"
                                zgodovina.append(["prazna_roka",stop_frame_pokrivanja_bowla,actual_dolzina_videja])
                                #print("AAAAAAAAAAAAAAAA")
                                action_assigned=True

                            elif action=="odvzemanje" and action_assigned==False and len(zgodovina)!=0 and last_action_num==1:

                                action_start_frame=last_action_frame+1
                                action_end_frame=start_frame_pokrivanja_bowla#-1
                                if action_end_frame<action_start_frame:
                                    action_end_frame=action_start_frame

                                zgodovina.append(["prenos_pina",action_start_frame,action_end_frame])
                                last_action_num=2

                                last_action_frame=action_end_frame
                                last_action="odlaganje_pina"
                                
                            # "LUT" za akcije
                            if last_action=="prijemanje_pina":
                                last_action_num=1
                            elif last_action=="prenos_pina":
                                last_action_num=2
                            elif last_action=="odlaganje_pina":
                                last_action_num=3
                            elif last_action=="prazna_roka":
                                last_action_num=4
                            
                            if action_assigned==False and len(zgodovina)!=0:

                                action_start_frame=last_action_frame+1
                                action_end_frame=stop_frame_pokrivanja_bowla
                                if action_end_frame<action_start_frame:
                                    action_end_frame=action_start_frame

                                zgodovina.append([last_action,action_start_frame,action_end_frame])

                                last_action_frame=action_end_frame
                                stop_frame_pokrivanja_bowla=0
                                start_frame_pokrivanja_bowla=0
                                action_assigned=True
                        else:
                            stop_frame_pokrivanja_bowla=0
                            start_frame_pokrivanja_bowla=0
                        #posodica_flag=True
                    
                    if pin_count==9 and action=="vstavljanje":
                        if debug_sprotno_stanje==1:
                            print("Vse pini so bili vstavljeni, frame: "+str(frame))
                        action="odvzemanje"
                    elif pin_count==0 and action=="odvzemanje" and len(pin_centers_spodaj)==0:
                        if debug_sprotno_stanje==1:
                            print("Vsi pini so bili pobrani, frame: "+str(frame))
                        #action="konec"
                    elif action=="konec":
                        if debug_sprotno_stanje==1:
                            print("konec, frame: "+str(frame))

            if debug_outlines==1 or debug_pins==1 or debug_sprotno_stanje==1 or debug_prikazi_vsak_frame==1 or debug_bowl==1:
                cv.waitKey(0)
                cv.destroyAllWindows()

        frame=frame+1

        action_assigned=False

    #if debug_sprotno_stanje==1:
    #    print(zgodovina)

    create_annotation_json(ime_videja, zgodovina)
    print("JSON file "+ime_videja+" created successfully!")
    create_annotation_json(ime_videja_2, zgodovina)
    print("JSON file "+ime_videja_2+" created successfully!")

    if debug_false_positive==1:
        primerjava_change_eventov(path_do_videjev+ime_videja+".json",pravilna_anotacija_path+ime_pravilna_anotacija+".json")
        primerjava_change_eventov(path_do_videjev+ime_videja_2+".json",pravilna_anotacija_path+ime_pravilna_anotacija_2+".json")

    if debug_anotacija==1:
        primerjava_anotacij(path_do_videjev+ime_videja+".json",pravilna_anotacija_path+ime_pravilna_anotacija+".json")
        primerjava_anotacij(path_do_videjev+ime_videja_2+".json",pravilna_anotacija_path+ime_pravilna_anotacija_2+".json")
