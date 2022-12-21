from imutils.video import FileVideoStream
from imutils.video import FPS
import multiprocessing as mp
import argparse
from cv2 import cv2
import glob
import os

def filterFrame(frame):
    if frame is not None:
        yframe = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
        yframe[:,:,0] = clahe.apply(yframe[:,:,0])
        frame = cv2.cvtColor(yframe, cv2.COLOR_YUV2BGR)
    return frame

def processVideo(vidPath):
    #print("[INFO] starting video file thread - " + vidPath)
    #fvs = FileVideoStream(vidPath, filterFrame).start()
    
    # read video
    cap = cv2.VideoCapture(vidPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    intermediate_path, infile = os.path.split(vidPath)
    outfile = os.path.join(os.path.split(intermediate_path)[0], 'bright_uncompressed_avi_videos', infile)
    cap.release()
    if os.path.exists(outfile):
        print(outfile + ' already exists - skipping this video')
        return
    else:
        print("[INFO] starting video file thread - " + vidPath)
        fvs = FileVideoStream(vidPath, filterFrame).start()
    
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        outvid = cv2.VideoWriter(outfile, fourcc, int(fps), (int(width), int(height)))
        fps = FPS().start()
        while fvs.running():
            try:
                frame = fvs.read()
                if frame is None:
                    break  
            
                outvid.write(frame)    
                fps.update()      
            except KeyboardInterrupt:
                break 
    
        fps.stop()
        print("[INFO - " + vidPath + "] approx. FPS: {:.2f}".format(fps.fps()))
    
        outvid.release()
        cv2.destroyAllWindows()
        fvs.stop()
        return

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True,
     	help="path to directory for task and marmoset pair. E.g. /project/nicho/data/marmosets/kinematics_videos/crickets/TYJL/")
    ap.add_argument("-n", "--nDates", default=1,
     	help="how many dates to process data for, starting from the most recent dataset. If the nDates is 0, all dates will be processed")
    ap.add_argument("-p", "--nProc", default=14,
     	help="how many processes to use. Match this with nodes*ntasks-per-node from the .sbatch job script")
    args = vars(ap.parse_args())
    
    if int(args['nDates']) == 0:
        date_dirs = sorted(glob.glob(os.path.join(args['dir'], '*')))
    else:
        date_dirs = sorted(glob.glob(os.path.join(args['dir'], '*')))[-int(args['nDates']):]

    videos = []
    for ddir in date_dirs:
        os.makedirs(os.path.join(ddir, 'bright_uncompressed_avi_videos'), exist_ok=True)       
        videos.extend(sorted(glob.glob(os.path.join(ddir, 'unfiltered_videos', '*.avi'))))    
    while len(videos) > 0: 
        nProc = min(int(args['nProc']), len(videos))
        procs = []
        for i_proc in range(nProc):
            vidPath = videos[i_proc]
            procs.append(mp.Process(target=processVideo, args=(vidPath,)))
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        videos[:nProc] = []
