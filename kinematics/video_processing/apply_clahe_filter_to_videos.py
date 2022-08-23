from imutils.video import FileVideoStream
from imutils.video import FPS
import argparse
import cv2
import glob
import os
import time
import numpy as np

def filterFrame(frame):
    if frame is not None:
        yframe = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
        yframe[:,:,0] = clahe.apply(yframe[:,:,0])
        frame = cv2.cvtColor(yframe, cv2.COLOR_YUV2BGR)
    return frame

def processVideo(vidPath):
    # read video
    cap = cv2.VideoCapture(vidPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    intermediate_path, infile = os.path.split(vidPath)
    outfile = os.path.join(os.path.split(intermediate_path)[0], 'bright_uncompressed_avi_videos', infile)
    cap.release()
    if os.path.exists(outfile):
        # print(os.path.basename(outfile) + ' already exists - skipping this video', flush=True)
        return
    else:
        print("Applying CLAHE filter and storing at " + outfile, flush=True)
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
        # print("[INFO - " + vidPath + "] approx. FPS: {:.2f}".format(fps.fps()))
    
        outvid.release()
        # cv2.destroyAllWindows()
        fvs.stop()
        
        return

if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dir", required=True, type=str,
     	help="path to directory for task and marmoset pair. E.g. /project/nicho/data/marmosets/kinematics_videos/crickets/TYJL/")
    ap.add_argument("-d", "--dates", nargs='+', required=True, type=str,
        help="date(s) of recording (can have multiple entries separated by spaces)")

    args = vars(ap.parse_args())
    
    print('\n\n Beginning CLAHE filter code at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)
    
    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    n_tasks = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))

    videos = []
    bright_dirs = []
    for date in args['dates']:
        ddir = os.path.join(args['input_dir'], date)
        os.makedirs(os.path.join(ddir, 'bright_uncompressed_avi_videos'), exist_ok=True)       
        videos.extend(sorted(glob.glob(os.path.join(ddir, 'unfiltered_videos', '*.avi'))))    

    bright_videos = [os.path.join(v.split('/unfiltered_videos')[0], 'bright_uncompressed_avi_videos', os.path.basename(v)) for v in videos]
    
    task_idx_cutoffs = np.floor(np.linspace(0, len(videos), n_tasks+1))   
    task_idx_cutoffs = [int(cut) for cut in task_idx_cutoffs]
    task_videos        = videos       [task_idx_cutoffs[task_id] : task_idx_cutoffs[task_id+1]]    
    task_bright_videos = bright_videos[task_idx_cutoffs[task_id] : task_idx_cutoffs[task_id+1]]
    
    prev_sum_video_sizes = 0
    updated_sum = 10
    while updated_sum > prev_sum_video_sizes or any(np.array([os.path.getsize(f) for f in bright_videos if os.path.exists(f)]) < 10000):
        prev_sum_video_sizes = sum(os.path.getsize(f) for f in bright_videos if os.path.exists(f))
        
        for vidPath, newVidPath in zip(task_videos, task_bright_videos):                            
            if os.path.exists(newVidPath):
                print(newVidPath + ' already exists', flush=True)
                continue
            else:
                processVideo(vidPath)

        time.sleep(10)

        updated_sum = sum(os.path.getsize(f) for f in bright_videos if os.path.exists(f))

    # for vidPath in videos:
    #     try:
    #         os.remove(vidPath)
    #     except OSError:
    #         pass
    # os.rmdir(os.path.dirname(videos[0]))