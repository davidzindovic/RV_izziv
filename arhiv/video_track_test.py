import numpy as np
import cv2

cap = cv2.VideoCapture('64210323_video_7.mp4')

# params for corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame,
                        cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None,
                             **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame,
                              cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                           frame_gray,
                                           p0, None,
                                           **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, 
                                       good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a=int(a)
        b=int(b)
        c=int(c)
        d=int(d)
        mask = cv2.line(mask, (a, b), (c, d),
                        color[i].tolist(), 2)
        
        frame = cv2.circle(frame, (a, b), 5,
                           color[i].tolist(), -1)
        
    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)
    
    k = cv2.waitKey(25)
    if k == 27:
        break

    # Updating Previous frame and points 
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
'''
from ipywidgets import interact, IntSlider
from funkcije import *

# nalozi video

#oVideo = loadVideo('64210323_video_1.mp4')
oVideo = loadVideo('video1.avi')

# prikaz
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
img = ax.imshow(oVideo[...,0], cmap='gray')

def update(frame):
    img.set_data(oVideo[...,frame])
    fig.canvas.draw()

#interact(update, frame = IntSlider(min=0, max=oVideo.shape[2] - 1))

def regLucasKanade(iImgFix, iImgMov, iMaxIter, oPar = (0,0), iVerbose=True):
    """Postopek poravnave Lucas-Kanade"""
    # pretvori vhodne slike v numpy polja tipa float
    iImgType = np.asarray(iImgMov).dtype
    iImgFix = np.array(iImgFix, dtype='float')
    iImgMov = np.array(iImgMov, dtype='float')
    # doloci zacetne parametre preslikae
    oPar = np.array(oPar)     
    # izracunaj prva odvoda slike
    Gx, Gy = imageGradient(iImgMov)      
    # v zanki iterativno posodabljaj parametre
    for i in range(iMaxIter):
        # doloci preslikavo pri trenutnih parametrih        
        oMat2D = transAffine2D(iTrans=oPar)        
        # preslikaj premicno sliko in sliki odvodov        
        iImgMov_t = transformImage(iImgMov, oMat2D)
        Gx_t = transformImage(Gx, oMat2D)
        Gy_t = transformImage(Gy, oMat2D)        
        # izracunaj sliko razlike in sistemsko matriko
        I_t = iImgMov_t - iImgFix
        B = np.vstack((Gx_t.flatten(), Gy_t.flatten())).transpose()
        # resi sistem enacb
        invBtB = np.linalg.inv(np.dot(B.transpose(), B))
        dp = np.dot(np.dot(invBtB, B.transpose()), I_t.flatten())        
        # posodobi parametre        
        oPar = oPar + dp.flatten()           
        if iVerbose: print('iter: %d' % i, ', oPar: ', oPar)
    # doloci preslikavo pri koncnih parametrih        
    oMat2D = transAffine2D(iTrans=oPar)        
    # preslikaj premicno sliko        
    oImgReg = transformImage(iImgMov, oMat2D).astype(iImgType)
    # vrni rezultat
    return oPar, oImgReg

from rvlib import showImage
# Preizkus funkcije regLucasKanade

# doloci fiksno in premicno sliko
oPar = [0, 1]
iImgFix = oVideo[:,:,0]
iImgMov = transformImage(iImgFix, transAffine2D(iTrans = oPar)) # sintetično generiran premik
# klici Lucas-Kanade poravnavo slik
import time    
ts = time.time()
oPar, oImgReg = regLucasKanade(iImgFix, iImgMov, 100)
print('parameters: ', oPar)
print('elapsed time: ', 1000.0*(time.time()-ts), ' ms')  
# narisi rezultate
showImage(iImgFix.astype('float') - iImgMov.astype('float'), 'Pred poravnavo')
showImage(iImgFix.astype('float') - oImgReg.astype('float'), 'Po poravnavi')

def regPyramidLK(iImgFix, iImgMov, iMaxIter, iNumScales, iVerbose=True):
    """Piramidna implementacija poravnave Lucas-Kanade"""
    # pretvori vhodne slike v numpy polja tipa float
    iImgFix = np.array(iImgFix, dtype='float')
    iImgMov = np.array(iImgMov, dtype='float')
    # pripravi piramido slik
    iPyramid = [ (iImgFix, iImgMov) ]
    for i in range(1,iNumScales):
        # decimiraj fiksno in premicno sliko za faktor 2
        iImgFix_2 = decimateImage2D(iImgFix, i)
        iImgMov_2 = decimateImage2D(iImgMov, i)
        # dodaj v seznam
        iPyramid.append((iImgFix_2,iImgMov_2))
    # doloci zacetne parametre preslikave
    oPar = np.array((0,0))          
    # izvedi poravnavo od najmanjse do najvecje locljivosti slik
    for i in range(len(iPyramid)-1,-1,-1):
        if iVerbose: 
            print('PORAVNAVA Z DECIMACIJO x{}'.format(2**i))
        # posodobi parametre preslikave
        oPar = oPar * 2.0
        # izvedi poravnavo pri trenutni locljivosti
        oPar, oImgReg = regLucasKanade(iPyramid[i][0], iPyramid[i][1], \
                            iMaxIter, oPar, iVerbose=iVerbose)
    # vrni koncne parametre in poravnano sliko
    return oPar, oImgReg

# Preizkus funkcije regPyramidLK
# doloci fiksno in premicno sliko
oPar = [0, 10]
iImgFix = oVideo[:,:,0]
iImgMov = transformImage(iImgFix, transAffine2D(iTrans = oPar))
# klici Lucas-Kanade poravnavo slik
import time    
ts = time.time()    
oPar, oImgReg = regPyramidLK(iImgFix, iImgMov, 20, 3)
print('parameters: ', oPar)
print('elapsed time: ', 1000.0*(time.time()-ts), ' ms')  
# narisi rezultate
showImage(iImgFix.astype('float') - iImgMov.astype('float'), 'Pred poravnavo')
showImage(iImgFix.astype('float') - oImgReg.astype('float'), 'Po poravnavi')

def trackTargetLK(iVideoMat, iCenterXY, iFrameXY, iVerbose=True):
    """Postopek sledenja Lucas-Kanade"""
    # pretvori vhodni video v numpy polje
    iVideoMat = np.asarray(iVideoMat)
    iCenterXY = np.array(iCenterXY)
    # definiraj izhodno spremenljivko
    oPathXY = np.array(iCenterXY.flatten()).reshape((1,2))
    # definiraj koordinate v tarci
    gx, gy = np.meshgrid(range(iFrameXY[0]), range(iFrameXY[1]))
    gx = gx - float(iFrameXY[0]-1)/2.0
    gy = gy - float(iFrameXY[1]-1)/2.0
    # zazeni LK preko vseh zaporednih okvirjev
    for i in range(1,iVideoMat.shape[-1]):
        # vzorcni tarco v dveh zaporednih okvirjih        
        iImgFix = interpolate1Image2D(iVideoMat[...,i-1], \
                    gx+oPathXY[-1,0], gy+oPathXY[-1,1])
        iImgMov = interpolate1Image2D(iVideoMat[...,i], \
                    gx+oPathXY[-1,0], gy+oPathXY[-1,1])
        # zazeni piramidno LK poravnavo
        oPar, oImgReg = regPyramidLK(iImgFix, iImgMov, 30, 3, iVerbose=False)
        # shrani koordinate
        oPathXY = np.vstack((oPathXY, oPathXY[-1,:] + oPar.flatten()))     
        print('\rkoordinate tarce: ', oPathXY[-1,:], end="")
    # vrni spremenljivko
    return oPathXY

# Preizkus funkcije trackTargetLK
# klici Lucas-Kanade sledenje tarci
import time    
ts = time.time()    
oPathXY = trackTargetLK(oVideo[...,:], (250,350), (40,40))
print('\nelapsed time: ', 1000.0*(time.time()-ts), ' ms')

# prikaz
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

def update(frame):
    #img.set_data(oVideo[...,frame])
    ax.clear()
    ax.imshow(oVideo[...,frame], cmap='gray')
    ax.plot(oPathXY[frame, 0], oPathXY[frame, 1], 'xr', markersize=9)
    fig.canvas.draw()

for a in range(100):
    update(a+1)
    print(a)
    #time.sleep(10)

#interact(update, frame = IntSlider(min=0, max=oVideo.shape[2] - 1))
'''
"""
import cv2
import numpy as np

ix, iy, k = 200,200,1
def onMouse(event, x, y, flag, param):
	global ix,iy,k
	if event == cv2.EVENT_LBUTTONDOWN:
		ix,iy = x,y 
		k = -1

cv2.namedWindow("window")
cv2.setMouseCallback("window", onMouse)

cap = cv2.VideoCapture("64210323_video_7.mp4")

while True:
    _, frm = cap.read()
    #print("a")
    if frm is not None:
        cv2.imshow("window", frm)

        if cv2.waitKey(1) == 27 or k == -1:
            old_gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

            cv2.destroyAllWindows()
            break

old_pts = np.array([[ix,iy]], dtype="float32").reshape(-1,1,2)
mask = np.zeros_like(frm)

while True:
    #print("b")
    _, frame2 = cap.read()
    #if frame2 is not None:
    new_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    new_pts,status,err = cv2.calcOpticalFlowPyrLK(old_gray, 
                        new_gray, 
                        old_pts, 
                        None, maxLevel=1,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                        15, 0.08))
    #print(type(new_pts))
    cv2.circle(mask, (int(new_pts.ravel()[0]) ,int(new_pts.ravel()[1])), 2, (0,255,0), 2)
    combined = cv2.addWeighted(frame2, 0.7, mask, 0.3, 0.1)

    cv2.imshow("new win", mask)
    cv2.imshow("wind", combined)

    old_gray = new_gray.copy()
    old_pts = new_pts.copy()

    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break 
"""
"""
import cv2
import numpy as np
cap = cv2.VideoCapture("64210323_video_1.mp4")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    ret, frame2 = cap.read()

    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

cap.release()
cv2.destroyAllWindows()
"""