# Package Management.

### General rules of thumb for package management

1. Before making any changes, I suggest cloning the environment to store a backup version before updating significant packages. Make sure to clean up the deprecated environments as you go. Note: you can also restore old versions from snapshots stored on beagle3/project.
2. In `/beagle3/nicho/environments/nwb_and_neuroconv`, you should update neuroconv and pynwb packages before you start a new experiment and begin making new NWB files. This will ensure those fiels are compatible for upload to Dandi when you get to the publishing stage.
3. In `/beagle3/nicho/environments/anipose-dlc-pytorch`, you should update deeplabcut before doing pose estimation if you haven't updated in a while.

## Anipose
1. Anipose is installed in two environments: `/beagle3/nicho/environments/anipose-dlc-pytorch` and `/project/nicho/environments/dlc`. The former contains the deeplabut V3 with pytorch, but doesn't anipose breaks in this environment for most steps. Because of this, the first step in pose estimation should be done using `anipose-dlc-pytorch`, but then you ahve to switch to `dlc` to do other anipose steps. This is okay because anipose is not actively maintained and hasn't changed in multiple years.
2. If you do need to make a new environment and re-install anipose, you need to track this one change over. In the code located at `/beagle3/nicho/environments/anipose-dlc-pytorch/lib/python3.11/site-packages/anipose/pose-videos.py`, I added lines to save scorer_info.txt within the function below. If a fresh version of anipose is installed, you'll want to copy over that portion of the function. 
```
def rename_dlc_files(folder, base, save_scorer=False):
    files = glob(os.path.join(folder, base+'*'))
    print('num_files = %d' % len(files), flush=True)
    print(files, flush=True)
    for fname in files:
        basename = os.path.basename(fname)
        _, ext = os.path.splitext(basename)
        os.rename(os.path.join(folder, basename),
                  os.path.join(folder, base + ext))

    if save_scorer:
        print('first_file is %s' % files[0], flush=True)
        print(files[0].split('DLC'), flush=True)
        scorer, _ = os.path.splitext(files[0].split('DLC')[1])
        scorer = f'DLC{scorer}'
        with open(os.path.join(os.path.split(os.path.split(files[0])[0])[0], 'scorer_info.txt'), 'w') as f:
            f.write(scorer)
```

## DeepLabCut
The environment stored at `/beagle3/nicho/environments/anipose-dlc-pytorch` currently has DLCv3.0.0rc2 installed. There have likely been updates to this, and you may want to upgrade. The only major change I would expect is that you won't need the pytorch config template currently stored at `/project/nicho/projects/marmosets/dlc_project_files/full_marmoset_model-Dalton-2022-07-26/pytorch_config_template.yaml`. In this version of DLC, the pytorch config was not located correctly and had to be manually added to dlc-models-pytorch (see train_dlc.py). I would expect this to be fixed in future versions. 

## Neuroconv

When you update neuroconv, this may also update the spikeinterface package. At the time of writing this guide, I had inserted a line in the code located at `/beagle3/nicho/environments/nwb_and_neuroconv/lib/python3.11/site-packages/spikeinterface/extractors/neoextractors/neobaseextractor.py` to extract the attribute "unit_name" within the sortingextractor. You may need to port this line into the upgraded version (you may also want to initiate a pull request to prevent this necessity in the future). Info below for modification:

```
class NeoBaseSortingExtractor(_NeoBaseExtractor, BaseRecording):
      below line 374:        BaseSorting.__init__(self, sampling_frequency, unit_ids)
      insert at line 376:    self.set_property("unit_name", spike_channels["name"])
```  

# TO-DO List

1. Some old datasets from TY still don't have acquisition NWB files because of analog/video timestamps alignment problems. Full notes for this can be found at `/cds3/nicho/data/marmosets/electrophys_archive/TY_array02/nwb_conversion_notes.txt`.
2. For the JL dataset, all the calibrations were done with double-sided calibration boards. Because of this, ~1/2 of the frames capture the board in the reverse orientation resulting in a terrible calibration. Currently, this can be adjusted manually for a single session using [TEMPLATE_check_for_bad_calibration_images.py](subject_specific_scripts/TEMPLATES/python/TEMPLATE_check_for_bad_calibration_images.py), but this takes multiple hours per session. A better way to do this would be train a lightweight object detection model that would automatically detect the board's orientation and perform the correct keypoints adjustment (to replace the manual flip/rotation options in the code. The very early attempt at this is located at `/project/nicho/projects/marmosets/detect_checkerboard_orientation_models/`, but it is incomplete.