import argparse
import glob
import os
import subprocess
import time
import numpy as np

def recompressVideo(vidPath):
    outVidPath, infile = os.path.split(vidPath)
    outVidPath = os.path.join(os.path.split(outVidPath)[0], 'avi_videos', infile)
    #print(outVidPath)
    print("Recompressing video and storing at " + outVidPath, flush=True)
    if not os.path.exists(outVidPath):
        subprocess.call(['ffmpeg', '-i', vidPath, '-vcodec', 'libx264', '-crf', '15', outVidPath])
    
if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_dir", required=True, type=str,
        help="path to directory for task and marmoset pair. E.g. /project/nicho/data/marmosets/kinematics_videos/crickets/TYJL/")
    ap.add_argument("-d", "--dates", nargs='+', required=True, type=str,
        help="date(s) of recording (can have multiple entries separated by spaces)")
    args=vars(ap.parse_args())

    print('\n\n Beginning recompression code at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)
    
    try:
        task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        n_tasks = int(os.getenv('SLURM_ARRAY_TASK_COUNT'))
    except:
        task_id = 0
        n_tasks = 1

    videos = []
    for date in args['dates']:        
        ddir = os.path.join(args['input_dir'], date)
        os.makedirs(os.path.join(ddir, 'avi_videos'), exist_ok=True)
        videos.extend(sorted(glob.glob(os.path.join(ddir, 'bright_uncompressed_avi_videos', '*.avi'))))

    final_videos = [os.path.join(v.split('/bright_uncompressed_avi_videos')[0], 'avi_videos', os.path.basename(v)) for v in videos]
    
    task_idx_cutoffs = np.floor(np.linspace(0, len(videos), n_tasks+1))   
    task_idx_cutoffs = [int(cut) for cut in task_idx_cutoffs]
    task_videos       = videos      [task_idx_cutoffs[task_id] : task_idx_cutoffs[task_id+1]]    
    task_final_videos = final_videos[task_idx_cutoffs[task_id] : task_idx_cutoffs[task_id+1]]
    
    print(task_idx_cutoffs)
    print(task_idx_cutoffs[task_id])
    print('creating avi videos %s thru %s' % (os.path.basename(task_final_videos[0]), 
                                              os.path.basename(task_final_videos[-1])))
        
    # prev_sum_video_sizes = 0
    # updated_sum = 10
    # while updated_sum  > prev_sum_video_sizes or any(np.array([os.path.getsize(f) for f in final_videos if os.path.exists(f)]) < 10000):
        # prev_sum_video_sizes = sum(os.path.getsize(f) for f in final_videos if os.path.exists(f))

    for vidPath, newVidPath in zip(task_videos, task_final_videos):
        if os.path.exists(newVidPath):
            print(newVidPath + ' already exists', flush=True)
            continue
        else:
            recompressVideo(vidPath)

        # time.sleep(10)

        # updated_sum = sum(os.path.getsize(f) for f in final_videos if os.path.exists(f))

    
    # for vidPath in videos:
    #     try:
    #         os.remove(vidPath)
    #     except OSError:
    #         pass

    # try:
    #     os.rmdir(os.rmdir(os.path.dirname(videos[0])))
    # except:
    #     pass

    print('\n\n Ending recompression code at %s\n\n' % time.strftime('%c', time.localtime()), flush=True)

