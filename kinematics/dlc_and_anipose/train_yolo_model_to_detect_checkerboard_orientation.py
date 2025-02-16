#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 08:46:54 2024

@author: daltonm
"""

from autodistill_grounded_sam import GroundedSAM
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8
import supervision as sv
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import cv2

class image_dataset():
    def __init__(self, data_dir: Path) -> None:
        self.original_images_path = data_dir / 'images'
        self.dset_path            = data_dir / 'dataset'
        self.dataset              = None 
    
    def plot_sample_unlabeled_images(self) -> None:
        sample_size = 16
        image_paths = sv.list_files_with_extensions(directory=self.original_images_path, 
                                                    extensions=["png", "jpg", "jpg"])    
        rng = np.random.default_rng()
        samples = rng.choice(np.array(image_paths), sample_size, replace=False)
        titles = [ipath.stem for ipath in samples]
        images = [cv2.imread(str(ipath)) for ipath in samples]
        sv.plot_images_grid(images=images, titles=titles, 
                            grid_size=(int(np.sqrt(sample_size)), int(np.sqrt(sample_size))), 
                            size=(sample_size, sample_size))
    
    def load_or_autolabel_dataset(self, ontology_dict: dict) -> None:

        try:        
            dataset = sv.DetectionDataset.from_yolo(
                images_directory_path      = str(self.dset_path / 'train' / 'images'),
                annotations_directory_path = str(self.dset_path / 'train' / 'labels'),
                data_yaml_path             = str(self.dset_path / 'data.yaml'),
                )
        except:
            dataset = []
        
        if len(dataset) > 0:
            self.dataset = dataset
        
        else:
            ontology=CaptionOntology(ontology_dict)
            base_model = GroundedSAM(ontology=ontology)
            
            dataset = base_model.label(
                input_folder=str(self.original_images_path),
                extension=".png",
                output_folder=str(self.dset_path))
            
            self.dataset = dataset
                
    def plot_labeled_images(self):
        image_names = list(self.dataset.images.keys())
        image_titles = [Path(img_name).stem for img_name in image_names]
        
        box_annotator   = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        images = []
        for image_name in image_names:
            image = self.dataset.images[image_name]
            annotations = self.dataset.annotations[image_name]
            labels = [self.dataset.classes[class_id] for class_id in annotations.class_id]
            annotated_image = box_annotator.annotate(scene      = image.copy(),
                                                     detections = annotations,)
            # annotated_image = label_annotator.annotate(scene    = annotated_image,
            #                                            detections = annotations,
            #                                            labels   = labels,)
            images.append(annotated_image)
        
        for start, stop in zip(range(0, len(image_names), 16), range(16, len(image_names)+16, 16)):
            sv.plot_images_grid(
                images=images[start:stop+1],
                titles=image_titles[start:stop+1],
                grid_size=(4, 4),
                size=(10, 8))
    
    def train_target_model(self):
        target_model = YOLOv8(str(self.dset_path / "yolov8n.pt"))
        target_model.train(str(self.dset_path / 'data.yaml'), epochs=100)
        
if __name__ == '__main__':
    
    cb_dset = image_dataset(data_dir=Path('/project/nicho/projects/marmosets/detect_checkerboard_orientation_models'))
    cb_dset.plot_sample_unlabeled_images()
    
    # ontology_dict = dict(origin='small dark circle or ellipse on a white background where two dark perpindicular lines intersect',
    #                      axes='dark line on a white background parallel to an edge of the checkerboard pattern')
    
    ontology_dict = {'small dark circle on light background' : 'origin'}
    
    cb_dset.load_or_autolabel_dataset(ontology_dict=ontology_dict,)
    
    cb_dset.plot_labeled_images()
    
    
    
    