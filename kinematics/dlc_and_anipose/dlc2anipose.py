import multiprocessing as mp
import argparse
import glob
import os
import subprocess

def calibration_jpg2avi(camFold, calibPath, date, label, fps, video_format):
    camNum = os.path.split(camFold)[1]
    outVidPath = os.path.join(calibPath, date + '_' + label + '_' + camNum + '.' + video_format)
    subprocess.call(['ffmpeg', '-r', fps, '-f', 'image2', '-s', '1440x1080', 
                     '-pattern_type', 'glob', '-i', os.path.join(camFold, '*.jpg'), '-vcodec', 'libx264', '-crf', '15', outVidPath])
    
if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imagePath", required=True,
     	help="path to directory holding calibration images. E.g. /home/marms/Documents/bci02_dlc/additional_videos/2021_08_31/calib")
    ap.add_argument("-a", "--aniposePath",
     	help="path to anipose project folder. E.g. /path/to/aniposeProj")
    ap.add_argument("-f", "--fps",
        help="frame rate of calibration video")
    ap.add_argument("-v", "--video_format",
        help="format of video to be created")    
    args = vars(ap.parse_args())
    
    date, label = os.path.split(args['imagePath'])
    date = os.path.split(date)[1]
    calibPath = os.path.join(args['aniposePath'], date, 'calibration')
    os.makedirs(calibPath, exist_ok=True)
    cam_folders = glob.glob(os.path.join(args['imagePath'], '*'))
    procs = []
    for camFold in cam_folders:   
        procs.append(mp.Process(target=calibration_jpg2avi, args=(camFold, calibPath, date, label, args['fps'], args['video_format'])))
    for p in procs:
        p.start()
    for p in procs:
        p.join()