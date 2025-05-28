import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
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
#       preveri dogajanje v zadnjih framih (NUJNO)

#----------------iz configa--------------
debug_prikaz=config_izziv_main.debug_prikaz           
debug_outlines=config_izziv_main.debug_outlines            
debug_pins=config_izziv_main.debug_pins
debug_sprotno_stanje=config_izziv_main.debug_sprotno_stanje
debug_video=config_izziv_main.debug_video
debug_prikazi_vsak_frame=config_izziv_main.debug_prikazi_vsak_frame
debug_bowl=config_izziv_main.debug_bowl
debug_visualization_text=config_izziv_main.debug_visualization_text
debug_aktivna_roka=config_izziv_main.debug_aktivna_roka
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
    """
    Gets a single frame from a video and passes it on as an image.
    
    Args:
        video_path: the path to the video
        frame_numer: number of the frame
        
    Returns:
        image: image of the chosen frame
        total_frames: the length of the video (in frames)
    """
    
    #video_dir=path_do_videjev

    #video_path = video_dir + video_path + ".mp4"

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
        image=cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return image,total_frames
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
    global prikazi_vmesne_korake

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
def detect_shapes(image, region_of_interest=None, canny_low=100, canny_high=120):
    """
    Zazna oblike na podlagi kontur. Prvotno mišljena za zaznavanje vseh objektov, vendar so se parametri
    izkazali ugodni le za iskanje pinov v 3x3 mreži lukenj.
    
    Args:
        image: slika, na kateri želimo iskati oblike
        region_of_interest: območje na sliki, znotraj katerega želimo iskati oblike
        canny_low: spodnja meja za Canny algoritem za detekcijo robov
        canny_high: zgornja meja za Canny algoritem za detekcijo robov

    Returns:
        pins_centers: Središča najdenih objektov (pinov v 3x3 mreži lukenj)
        shapes: Podrobne informacije (parametri kot so aspect ratio ipd.) najdene oblike
        (x, y, w, h): touple, ki hrani podatke območja, znotraj katerega smo iskali (x in y začetne točke, w=širina,h=višina)
    """

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

    edges = cv.Canny(blurred, canny_low, canny_high,apertureSize=3)
    
    # dodano iz interneta za odebelitev robov/omogočanje lažjega iskanja zaprtih kontur
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    dilated = cv.dilate(edges, kernel)

    # Use RETR_LIST to get all contours, not just external ones
    contours, _ = cv.findContours(dilated, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    #ZADNJI PARAMETER:CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE ,CHAIN_APPROX_TC89_L1 CHAIN_APPROX_TC89_KCOS 
    
    shapes = {
        'pins': []
    }
    
    pins_centers=[]

    for contour in contours:
        area = cv.contourArea(contour)
        #if area < min_area or area > max_area:
        if 90 > area and area> 170:
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
        if (aspect_ratio > 1 and aspect_ratio<3 and circularity>0.3):#90 < area < 170
            shapes['pins'].append(shape_info)
            pins_centers.append([abs_cx,abs_cy])
            if debug_pins:
                print("pin: "+str(aspect_ratio)+"|"+str(area)+"|"+str(circularity)+"|"+str(radius)+"|"+str(perimeter)+"|"+str(color_name))

    if debug_pins:
        print("")

    return pins_centers,shapes, (x, y, w, h) if region_of_interest else None, edges

# funkcija za določanje barve pina:
def classify_color(hsv_values):
    """
    Zazna barvo na podlagi HSV vrednosti.
    
    Args:
        hsv_values: HSV vrednosti pina    
    
    Returns:
        barva v obliki besede (string) 
    """
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



# Funkcija za iskanje pinov v bowlu
def detect_dark_objects(image):
    """
    Pripravi vhodno sliko za zaznavanje pinov v bowlu z upragovljanjem.
    
    Args:
        image: slika, ki jo želimo pripraviti
    
    Returns:
        thresh_og: črno bela slika, ki je rezultat upragovljanja +
    """
    
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
        cv.imshow("Bowl blurred",blurred_og)
    cv.waitKey(0)
    return thresh_og

def zaznaj_roko(base_image, hand_image):

    roi_x = 150
    roi_y = 50
    roi_height=400
    roi_width=200

    base_image= cv.cvtColor(base_image, cv.COLOR_BGR2GRAY)
    base_image=cv.GaussianBlur(base_image, (5, 5), 0)
    _,base_image=cv.threshold(base_image,200,255,cv.THRESH_BINARY)

    hand_image= cv.cvtColor(hand_image, cv.COLOR_BGR2GRAY)
    hand_image=cv.GaussianBlur(hand_image, (5, 5), 0)
    _,hand_image=cv.threshold(hand_image,200,255,cv.THRESH_BINARY)

    razlika=base_image-hand_image
    razlika_copy=razlika.copy()

    #izris rezulatov:

    # cv.rectangle(razlika_copy,(roi_x,roi_y),(roi_x+roi_width,roi_y+roi_height),(255),2)

    # cv.imshow("razlika",razlika_copy)
    # cv.imshow("base",base_image)
    # cv.imshow("hand",hand_image)
    # cv.waitKey(0)

    trenutno_stevilo_pikslov_roke_levo=0
    trenutno_stevilo_pikslov_roke_desno=0

    for h_y in range(roi_y,(roi_y+roi_height)):
        for h_x in range(roi_x,(roi_x+math.floor(roi_width/2))):
            if razlika[h_y][h_x]==1:
                trenutno_stevilo_pikslov_roke_levo+=1

    for h_y in range(roi_y,(roi_y+roi_height)):
        for h_x in range((roi_x+math.floor(roi_width/2)),(roi_x+roi_width)):
            if razlika[h_y][h_x]==1:
                trenutno_stevilo_pikslov_roke_desno+=1

    
    if trenutno_stevilo_pikslov_roke_levo>trenutno_stevilo_pikslov_roke_desno:
        return "leva"
    elif trenutno_stevilo_pikslov_roke_levo<trenutno_stevilo_pikslov_roke_desno:
        return "desna"
    else:
        return ""


# funkcija za pripravo slike za prikaz zaznanih oblik:
def visualize_detection(slika, shapes, roi_rect=None):
    """
    Vizualizira podane oblike.
    
    Args:
        slika:      slika, na kateri želimo prikazati oblike
        shapes:     seznam oblik
        roi_rect:   območje interesa, ki ga lahko prikažemo
                    da vemo znotraj katerega območja smo
                    našli oblike
    
    Returns:
        display:    originalna slika, ki ima čez narisane 
                    zaznane oblike
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
            
            if debug_visualization_text==1:
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
            "event": [event]
        })
    
    # Create the filename with .json extension
    json_filename = f"{filename}.json"
    
    # Write to file (overwrite if exists)
    with open(json_filename, 'w') as f:
        json.dump(data, f, indent=4)
    
    return json_filename

def luknje_v_3x3_gridu_POV(centers_spodaj_temp,centers_zgoraj_temp):
    """
    Funkcija za pravilno razporejanje lukenj za pine, da bodo razporejene
    pravilno (gledano iz posamezne kamere)
    
    Args:
        centers_spodaj_temp:    središča lukenj za pine, gledano iz kamere ki
                                ima 3x3 mrežo lukenj na spodnji polovici slike
        centers_zgoraj_temp:    središča lukenj za pine, gledano iz kamere ki
                                ima 3x3 mrežo lukenj na zgornji polovici slike
    
    Returns:
        centers_spodaj_final:   urejena tabela središč lukenj za pine, 
                                gledano iz kamere ki ima 3x3 mrežo lukenj na 
                                spodnji polovici slike
        centers_zgoraj_final:   urejena tabela središč lukenj za pine, 
                                gledano iz kamere ki ima 3x3 mrežo lukenj na 
                                zgornji polovici slike 
    """

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

def main_optimized(video1,video2,output_json_path):
    global last_action, last_action_num, prikazi_vmesne_korake, razporedi_centre_lukenj, premaknjen_board, num_pins_hypo, debug_aktivna_roka
    
    #-----------------------------
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

    # podatek o radiju sklede s pini:
    radij_bowla=0

    # zastavica za prejem podatka o pinih v bowlu iz funkcije
    pin_in_bowl_data_flag=True

    # podatek o factorju za izračun površine
    bowl_factor_osnova=3500/9

    # faktor ki se množi z osnovo
    bowl_factor=0

    zeljeno_stevilo_bowl_pinov=9        # koliko pinov pričakujemo kot naslednji pravilni odgovor
    zeljeno_stevilo_tracker=0           # spremenljivka za spremljanje koliko frameov smo imeli željeno število pinov v bowlu
    trenutno_stevilo_pinov_v_bowlu=0    # potrjeno število pinov
    current_state="ni roka"             # spremenljiva za stanje, če roka pokriva bowl ("roka") ali ne ("ni roka")    

    # spremenljivka, ki hrani podatek o dolžini v videja (število frame-ov)
    actual_dolzina_videja=0

    action_assigned=False               # zastavica, ki zabeleži da je v anotacijo že bila zapisana akcija v tej iteraciji
    action_start_frame=0                # spremenljivka, ki hrani podatek o začetku anotirane akcije
    action_end_frame=0                  # spremenljivka, ki hrani podatek o koncu anotirane akcije

    # za detekcijo uporabljene roke (leva ali desna)
    base_img=[]                         # spremenljivka za shranjevanje osnovne slike
    hand_img=[]                         # slika, na kateri pričakujemo da bo roka že vidna
    uporabljena_roka=""                 # spremenljivka, ki hrani podatek o roki, ki jo uporabnik uporablja
    base_img_1=[]                       # spremenljivka za shranjevanje prve slike prve kamere
    base_img_2=[]                       # spremenljivka za shranjevanje prve slike druge kamere

    # za detekcijo prehitrega zacetka poskusa:
    base_img_lucke=[]                   # spremenljivka ki hrani sliko iz kamere bližje 3x3 mreži lukenj
    prehiter_poskus_flag=None           # zastavica za beleženje, če je uporabnik prehitro začel s poskusom

    while video!="stop":
    
        #shranimo sliki iz obeh kamer izbranega poskusa:
        #slika,dolzina_videja = get_video_frame(ime_videja, frame+1)
        #slika_2,dolzina_videja_2 = get_video_frame(ime_videja_2, frame+1)
        slika,dolzina_videja = get_video_frame(video1, frame+1)
        slika_2,dolzina_videja_2 = get_video_frame(video2, frame+1)
    

        if dolzina_videja is not None and dolzina_videja_2 is not None:
            if dolzina_videja>dolzina_videja_2:
                actual_dolzina_videja=dolzina_videja_2
            elif dolzina_videja<=dolzina_videja_2:
                actual_dolzina_videja=dolzina_videja
            #print(f"1: {dolzina_videja}, 2: {dolzina_videja_2}")

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
            
            #za zaznavanje roke:
            if len(base_img_1)==0:
                base_img_1=image_bgr.copy()
        
            if len(base_img_2)==0:
                base_img_2=image_bgr_2.copy()

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
                                                    canny_low=canny_low,
                                                    canny_high=canny_high)
                
                pin_centers_2,shapes_2, roi_rect_2, edges_2 = detect_shapes(image_bgr_2, 
                                                    region_of_interest=roi,
                                                    canny_low=canny_low,
                                                    canny_high=canny_high)


                # Ko najdemo središča lukenj, lahko določimo kje ima
                # katera kamera ima 3x3 grid na spodnji/zgornji strani
                if setup==1 or setup==0:

                    #določanje na kateri polovici slike so luknje
                    vsota_y=0
                    vsota_y_2=0
                    for s in centers:
                        vsota_y+=s[1]
                    for s_2 in centers_2:
                        vsota_y_2+=s_2[1]
                    
                    if vsota_y>vsota_y_2:
                        stran_lukenj="spodaj"
                        stran_lukenj_2="zgoraj"
                    else:
                        stran_lukenj="zgoraj"
                        stran_lukenj_2="spodaj"
                    # konec določanja

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
                        
                    if prikazi_vmesne_korake==1:
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
                        main_thr=po_thr_2

                        # za detekcijo uporabljene roke
                        base_img=base_img_1

                        # za detekcijo uporabljene roke
                        if len(base_img_lucke)==0:
                            base_img_lucke=slika.copy()
                        
                        # za detekcijo uporabljene roke
                        if frame>100 and len(hand_img)==0:
                            hand_img=slika.copy()

                    elif stran_lukenj_2=="spodaj":
                        pin_centers_spodaj=pin_centers_2
                        pin_centers_zgoraj=pin_centers
                        centers_spodaj_temp=centers_2
                        centers_zgoraj_temp=centers
                        main_thr=po_thr

                        # za detekcijo uporabljene roke
                        base_img=base_img_2

                        # za detekcijo uporabljene roke
                        if len(base_img_lucke)==0:
                            base_img_lucke=slika_2.copy()

                        # za detekcijo uporabljene roke
                        if frame>100 and len(hand_img)==0:
                            hand_img=slika_2.copy()

                    # za detekcijo uporabljene roke
                    if debug_aktivna_roka==1:
                        if frame>100 and uporabljena_roka=="" and len(base_img)!=0 and len(hand_img)!=0:
                            uporabljena_roka=zaznaj_roko(base_img,hand_img)

                        if uporabljena_roka!="":
                            print(f"Aktivna je {uporabljena_roka} roka")
                            debug_aktivna_roka=0

                    # za detekcijo prehitrega poskusa
                    if prehiter_poskus_flag==None:
                        pass

                    # od tu naprej za določanje stanja v 3x3 gridu
                    if razporedi_centre_lukenj==True:
                        #enkrat razporedimo centre lukenj v 3x3 grid
                        centers_spodaj_final,centers_zgoraj_final=luknje_v_3x3_gridu_POV(centers_spodaj_temp,centers_zgoraj_temp)
                        razporedi_centre_lukenj=False

                    if premaknjen_board==True:
                        #
                        # sem paše transormacija 3x3 grida lukenj s transf matriko


                        premaknjen_board=False

                    # izračun razdalj med luknjami v 3x3 gridu,
                    dx=centers_zgoraj_final[1][0]-centers_zgoraj_final[4][0]
                    dy=centers_zgoraj_final[1][1]-centers_zgoraj_final[4][1]

                    # izračun radija bowla in središča bowla
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
                        cv.imshow("Pini v bowlu",slikaaa)
                    
                    #------------------------število pinov v bowlu glede na število črnih pikslov---------------------------

                    num_crni_piksli_bowl=0
                    for y_coord in range(main_thr.shape[0]):
                        for x_coord in range(main_thr.shape[1]):
                            razdalja_centr_bowl_pin=math.sqrt(math.pow(x_coord-center_bowla_blizje_kameri[0],2)+math.pow(y_coord-center_bowla_blizje_kameri[1],2))
                            if razdalja_centr_bowl_pin<radij_bowla and main_thr[y_coord][x_coord]==0:
                                num_crni_piksli_bowl+=1
                    stevilo_pinov_v_bowlu=math.floor(num_crni_piksli_bowl/(bowl_factor_osnova+bowl_factor*35))

                    # če jih ne najde 9 ampak se prikaže roka v napoto -> sprememba last action in željenega št pinov
                    if zeljeno_stevilo_bowl_pinov==9 and stevilo_pinov_v_bowlu>10 and action=="vstavljanje":
                        zeljeno_stevilo_bowl_pinov-=1

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
                            
                    if action=="odvzemanje":
                        if stevilo_pinov_v_bowlu>10 and current_state=="ni roka":
                            current_state="roka"
                            if trenutno_stevilo_pinov_v_bowlu<9:
                                trenutno_stevilo_pinov_v_bowlu+=1
                                if trenutno_stevilo_pinov_v_bowlu<5:
                                    bowl_factor+=1
                                zeljeno_stevilo_bowl_pinov+=1

                        if stevilo_pinov_v_bowlu<10 and current_state=="roka":
                            current_state="ni roka"
                    elif action=="vstavljanje":
                        if stevilo_pinov_v_bowlu>10 and current_state=="ni roka":
                            current_state="roka"
                        if stevilo_pinov_v_bowlu<10 and current_state=="roka":
                            current_state="ni roka"

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

                    # če se je zgodila sprememba in smo v fazi vstaljanja v 3x3 grid
                    if sprememba==1 and action=="vstavljanje":

                        if debug_sprotno_stanje==1:
                            print("-----------------")
                            #print("Frame:" + str(frame))
                            #print("faza: vstavljanje")
                            #for c in range(3):
                            #    print(inserted_pins_zgoraj[c])
                            #print("~~~~~~~~~~~~~~~")
                            print("Stanje pri frame "+str(frame))
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
                            # izris grida v terminalu 
                            # (gledano iz kamere, ki ima mrežo bližje sebi)
                            print("-----------------")
                            print("Stanje pri frame "+str(frame))
                            for cd in range(3):
                                print(inserted_pins_spodaj[cd])
                            print("-----------------")

                        if action_assigned==False and len(zgodovina)!=0 and last_action_num==3 and trenutno_stevilo_pinov_v_bowlu!=9:
                            last_action="prijemanje_pina"
                            num_pins_hypo-=1

                            action_start_frame=last_action_frame+1
                            action_end_frame=frame-1
                            if action_end_frame<action_start_frame:
                                action_end_frame=action_start_frame

                            if pin_count==0 and trenutno_stevilo_pinov_v_bowlu==9:#ce ni pinov vec v 3x3 gridu
                                action_end_frame=actual_dolzina_videja-1

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

                    if debug_bowl==1:
                        print("V bowlu je: "+str(trenutno_stevilo_pinov_v_bowlu)+" pinov. Najdeno: "+str(stevilo_pinov_v_bowlu)+" pinov. Frame: "+str(frame)+" . Stanje roke: "+str(current_state)+" . Action: "+action+". Last action number: "+str(last_action_num)+". Bowl factor: "+str(bowl_factor)+" .Start: "+str(start_frame_pokrivanja_bowla)+" , konec: "+str(stop_frame_pokrivanja_bowla))

                    if current_state=="ni roka" and start_frame_pokrivanja_bowla!=0:
                        stop_frame_pokrivanja_bowla=frame
                        if (stop_frame_pokrivanja_bowla-start_frame_pokrivanja_bowla)>1:

                            # za začetek videja:
                            if len(zgodovina)==0 and action_assigned==False:

                                action_start_frame=1
                                action_end_frame=start_frame_pokrivanja_bowla
                                if action_end_frame<action_start_frame:
                                    action_end_frame=action_start_frame

                                zgodovina.append(["prazna_roka",action_start_frame,action_end_frame])
                                last_action_num=4

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
                            
                            #elif action=="odvzemanje" and len(zgodovina)!=0 and trenutno_stevilo_pinov_v_bowlu==9 and last_action_num==3:
                            #    action="konec"
                            #    zgodovina.append(["prazna_roka",stop_frame_pokrivanja_bowla,actual_dolzina_videja])

                            #    action_assigned=True        

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
                    
                    if action=="odvzemanje" and len(zgodovina)!=0 and trenutno_stevilo_pinov_v_bowlu==9 and last_action_num==3:
                        action="konec"
                        zgodovina.append(["prazna_roka",action_end_frame+1,actual_dolzina_videja-1])

                        action_assigned=True  
                    
                    if pin_count==9 and action=="vstavljanje":
                        if debug_sprotno_stanje==1:
                            print("Vse pini so bili vstavljeni, frame: "+str(frame))
                        action="odvzemanje"

                    elif pin_count==0 and action=="odvzemanje" and len(pin_centers_spodaj)==0:
                        if debug_sprotno_stanje==1:
                            print("Vsi pini so bili pobrani, frame: "+str(frame))

                    elif action=="konec":
                        if debug_sprotno_stanje==1:
                            print("konec, frame: "+str(frame))

                    

            if debug_outlines==1 or debug_pins==1 or debug_sprotno_stanje==1 or debug_prikazi_vsak_frame==1 or debug_bowl==1:
                cv.waitKey(0)
                cv.destroyAllWindows()

        frame=frame+1

        action_assigned=False

    #create_annotation_json(ime_videja, zgodovina)
    #print("JSON file "+ime_videja+" created successfully!")
    #create_annotation_json(ime_videja_2, zgodovina)
    #print("JSON file "+ime_videja_2+" created successfully!")

    output_json_path=output_json_path[0:-5]#odreže .json
    create_annotation_json(output_json_path, zgodovina)
    print("JSON file "+output_json_path+" created successfully!")


if __name__ == "__main__":
    main_optimized()
