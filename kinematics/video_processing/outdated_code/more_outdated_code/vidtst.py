import numpy as np
import cv2
from imutils.video import FPS

# read video
infile = r'C:/Users/Dalton/Downloads/cam_vids/2021_01_03_foraging_session1_event024_cam1.avi'
cap = cv2.VideoCapture(infile)
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
filename = r'C:/Users/Dalton/Downloads/cam_vids/2021_01_03_foraging_session1_event024_cam1_filtered.avi' #121620C2.avi'

print(cap.isOpened())
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(filename, fourcc, int(fps), (int(width), int(height)))
ct = 1
fps = FPS().start()

while(cap.isOpened()):
    # get frame
    ret, frame = cap.read()
    if ret==False:
        print('frame not loaded...')
        break

    print(ct)
    # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    yframe = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    #    yframe[:,:,0] = cv2.equalizeHist(yframe[:,:,0])
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
    yframe[:,:,0] = clahe.apply(yframe[:,:,0])
    frame = cv2.cvtColor(yframe, cv2.COLOR_YUV2BGR)
    #    wb = cv2.xphoto.createSimpleWB()
    #    frame = wb.balanceWhite(frame)

    out.write(frame)    
    # cv2.imshow('frame', frame)
    ct+=1
    if cv2.waitKey(1) == ord('q'):
        break

    fps.update()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))    

cap.release()
out.release()
cv2.destroyAllWindows()
    
# rotate image
#sample = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
# convert image to YUV space
#Ysample = cv2.cvtColor(sample, cv2.COLOR_BGR2YUV)
# equalize the Y channel
#eqy = cv2.equalizeHist(Ysample[:,:,0])
# put contrast correction back into image
#Ysample[:,:,1] = eqy
# convert corrected image to RGB     
#Rsample = cv2.cvtColor(Ysample, cv2.COLOR_YUV2RGB)
