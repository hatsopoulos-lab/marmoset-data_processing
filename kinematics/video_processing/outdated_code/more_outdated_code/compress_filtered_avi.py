import multiprocessing as mp
import argparse
import glob
import os
import subprocess

def recompressVideo(vidPath):
    outVidPath = vidPath.split('_filtered')[0] + '_bright.avi'
    subprocess.call(['ffmpeg', '-i', vidPath, '-vcodec', 'libx264', '-crf', '20', outVidPath])
    os.remove(vidPath)
    
if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True,
     	help="path to directory for task and marmoset pair. E.g. /gpfs/data/nicho-lab/marmosets/kinematics_videos/foraging/TYJL/")
    ap.add_argument("-n", "--nDates", default=1,
     	help="how many dates to process data for, starting from the most recent dataset. If the nDates is 0, all dates will be processed")
    args = vars(ap.parse_args())
    
    if int(args['nDates']) == 0:
        date_dirs = sorted(glob.glob(os.path.join(args['dir'], '*')))
    else:
        date_dirs = sorted(glob.glob(os.path.join(args['dir'], '*')))[-int(args['nDates']):]
    
    videos = []
    for ddir in date_dirs:        
        videos.extend(sorted(glob.glob(os.path.join(ddir, 'filtered_avi_videos', '*_filtered.avi'))))
        
    while len(videos) > 0: 
        nProc = min(14, len(videos))
        procs = []
        for i_proc in range(nProc):
            vidPath = videos[i_proc]
            procs.append(mp.Process(target=recompressVideo, args=(vidPath,)))
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        videos[:nProc] = []