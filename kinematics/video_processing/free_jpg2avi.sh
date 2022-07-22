#!/bin/bash

# An automated processing script for converting jpg files into videos.
# An automated processing script for converting jpg files into videos.
                                                                     
# Example: sudo Documents/camera_control_code/jpg2avi_apparatus.sh 'TYJL' '2021_01_08' 'foraging' '1' '150'

#                                                                marmCode    date      expName session  framerate


############ IMPORTANT #############################
# From here on, please review all commented-out lines and change to reflect where you want long-term storage of videos. 
####################################################

if [ ! -d "/media/CRI/marmosets" ]; then
    echo ''
    echo "CRI is not mounted. Please copy the following code into the terminal, enter your username after 'user= ', and press Enter. This will mount CRI, where we will save the avi files. After doing doing so, re-run the script. Below is the code: "
    echo ''
    echo "sudo mount -v -t cifs -o user=daltonm,domain=BSDAD //prfs.cri.uchicago.edu/nicho-lab /media/CRI"
    echo ''
    exit
fi

cams='1 2 3 4'

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR # cd to directory that holds this file

marms=$1
date=$2
exp=$3
sNums=$4
fps=$5

for sNum in $sNums; do
    path=/project/nicho/data/marmosets/kinematics_videos/$exp/$marms/$date/session$sNum #+++
    video_path=/project/nicho/data/marmosets/kinematics_videos/$exp/$marms/$date/unfiltered_videos #+++
        
    # This chunck pulls out the last event number so a for loop can run to convert videos in order with conistent naming conventions.
    fileName=$(ls -1 $path/jpg_cam1/ | tail -n 1)
    eventChunk="$( echo "$fileName" | grep -o "event.*_" )"
    eventChunk=$(echo $eventChunk | cut -c -12)
    lastEvent="$( echo "$eventChunk" | grep -o "[0-9][0-9][0-9]" )"
    firstDig="$( echo $lastEvent | cut -c 1 )"
    twoDig="$( echo $lastEvent | cut -c -2 )"
    if [ $twoDig == 00 ]; then
        lastEvent="$( echo "$lastEvent" | cut -c 3 )"
    elif [ $firstDig == 0 ]; then
        lastEvent="$( echo "$lastEvent" | cut -c 2- )"
    fi
    
    for cam in $cams; do
    
        echo ''
        echo 'creating avi videos for cam' $cam
        echo ''
        # If you want to change filename conventions or video params, this is where to do so (in the ffmpeg line). 
        # -s flag adjusts resolution. 
        # -pattern_type glob grabs all filenames that start as written and end in .jpg, and turns these into a video. 
        # -crf flag changes the compression ratio, with higher numbers = more compression. 
        # -vcodec changes the video codec used, i wouldn't touch this unless you get an error. 
        # The last argument is the filepath for the video to be stored.     
        for (( event=1; event <= lastEvent; event++ )); do
            eventNum=$(printf "%03d" $event)
    
            currentFile=$(find $path/jpg_cam$cam/ -name ${marms}_${date}_${exp}_session_${sNum}_cam${cam}_event_${eventNum}_frame_'0000001*.jpg')
            len=$(expr length $currentFile)
            time=$(echo $currentFile | cut -c $(($len-10))-$(($len-4)))
                
            python $DIR/fix_drops_during_vid_conversion.py "$path/jpg_cam$cam/${marms}_${date}_${exp}_session_${sNum}_cam${cam}_event_${eventNum}_frame_*" "$fps" 
            ffmpeg -r $fps -f image2 -s 1440x1080 -pattern_type glob -i $path/jpg_cam$cam/${marms}_${date}_${exp}_session_${sNum}_cam${cam}_event_${eventNum}_frame_'*.jpg' -vf "transpose=2" -crf 15 -vcodec libx265 $video_path/${marms}_${date}_${time}_${exp}_s${sNum}_e${eventNum}_cam$cam.avi        
        done
    done
done