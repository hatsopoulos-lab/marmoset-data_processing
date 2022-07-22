import argparse
import glob
import os
import subprocess
import time
import numpy as np

def recompressVideo(vidPath):
    outVidPath, infile = os.path.split(vidPath)
    #print((outVidPath, infile))
    outVidPath = os.path.join(os.path.split(outVidPath)[0], 'avi_videos', infile)
    #print(outVidPath)
    if not os.path.exists(outVidPath):
        subprocess.call(['ffmpeg', '-i', vidPath, '-vcodec', 'libx264', '-crf', '15', outVidPath])
    try:
        os.remove(vidPath)
    except OSError:
        pass
    
if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dir", required=True,
        help="path to directory for task and marmoset pair. E.g. /project/nicho/data/marmosets/kinematics_videos/crickets/TYJL/")
    ap.add_argument("-d", "--dates", nargs='+', required=True,
        help="date(s) of recording (can have multiple entries separated by spaces)")
    args=vars(ap.parse_args())
   
    jobid = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    time.sleep(jobid)

    videos = []
    for date in args['dates']:        
        ddir = os.path.join(args['input_dir'], date)
        os.makedirs(os.path.join(ddir, 'avi_videos'), exist_ok=True)
        videos.extend(sorted(glob.glob(os.path.join(ddir, 'bright_uncompressed_avi_videos', '*.avi'))))

    final_videos = [os.path.join(v.split('/bright_uncompressed_avi_videos')[0], 'avi_videos', os.path.basename(v)) for v in videos]
    prev_sum_video_sizes = 0
    updated_sum = 10
    while updated_sum  > prev_sum_video_sizes or any(np.array([os.path.getsize(f) for f in final_videos if os.path.exists(f)]) < 10000):
        prev_sum_video_sizes = sum(os.path.getsize(f) for f in final_videos if os.path.exists(f))

        for vidPath, newVidPath in zip(videos, final_videos):
            if os.path.exists(newVidPath):
                print(newVidPath + ' already exists', flush=True)
                continue
            else:
                recompressVideo(vidPath)

        time.sleep(5)

        updated_sum = sum(os.path.getsize(f) for f in final_videos if os.path.exists(f))

