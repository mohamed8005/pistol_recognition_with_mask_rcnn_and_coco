"""
Mask R-CNN


------------------------------------------------------------


    
    #    access venv :
    myvenv/Script/activate
    #    install requirements:
    pip install -r requirements_mouad2.txt

    #  test model on an image with latest trained model:
    python custom2.py splash --weights=last --image=".\\dataset\\train\\pistol (24).jpg"
        

    # Train a new model starting from pre-trained COCO weights
    python3 custom2.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 custom2.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 custom2.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 custom2.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 custom2.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import numpy as np
import skimage.draw
import argparse
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn.visualize import display_instances
import tensorflow as tf
import skimage.io
import matplotlib.pyplot as plt

# Check if a GPU is available in TensorFlow 1.x
if tf.test.is_gpu_available():
    print("GPU is available")
else:
    print("GPU is not available, using CPU")

# Root directory of the project
ROOT_DIR = os.path.abspath("..\\..\\")

# Path to COCO trained weights
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class CustomConfig(Config):
    NAME = "pistol"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + pistol
    
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

class CustomDataset(utils.Dataset):
    def load_custom(self, dataset_dir, subset):
        self.add_class("pistol", 1, "pistol")
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations = json.load(open(os.path.join(dataset_dir, "pistolet_json.json")))

        for image_id, info in annotations.items():
            polygons = [region['shape_attributes'] for region in info['regions']]
            objects = [region['region_attributes']['names'] for region in info['regions']]
            num_ids = [1 if name == "pistol" else 0 for name in objects]
            image_path = os.path.join(dataset_dir, info['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            self.add_image(
                "pistol",
                image_id=info['filename'],
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                num_ids=num_ids)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "pistol":
            return super(self.__class__, self).load_mask(image_id)
        
        num_ids = image_info['num_ids']
        mask = np.zeros([image_info["height"], image_info["width"], len(image_info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(image_info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        return self.image_info[image_id]["path"]

def train(model, dataset_dir, epochs=20):
    dataset_train = CustomDataset()
    dataset_train.load_custom(dataset_dir, "train")
    dataset_train.prepare()

    dataset_val = CustomDataset()
    dataset_val.load_custom(dataset_dir, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,
                layers='heads')

def apply_splash(model, image_path=None, video_path=None):
    if image_path:
        image = skimage.io.imread(image_path)
        result = model.detect([image], verbose=1)[0]

        # Display results
        display_instances(image, result['rois'], result['masks'], result['class_ids'], 
                          ["BG", "pistol"], result['scores'])
        plt.show()

    elif video_path:
        
        pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or apply color splash with Mask R-CNN.")
    parser.add_argument("command", metavar="<command>", help="'train' or 'splash'")
    parser.add_argument("--dataset", required=False, metavar="/path/to/dataset/", help="Directory of the dataset")
    parser.add_argument("--weights", required=True, metavar="/path/to/weights.h5", help="Path to weights file")
    parser.add_argument("--image", required=False, metavar="path or URL to image", help="Image to apply the effect on")
    args = parser.parse_args()

    config = CustomConfig()
    model_dir = DEFAULT_LOGS_DIR
    model = modellib.MaskRCNN(mode="training" if args.command == "train" else "inference", config=config, model_dir=model_dir)
    
    # Determine weights path based on input
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
    elif args.weights.lower() == "last":
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights and exclude class-specific layers for compatibility
    if args.weights.lower() == "coco":
        model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Execute command
    if args.command == "train":
        train(model, args.dataset, epochs=20)
    elif args.command == "splash":
        assert args.image, "Please provide --image to apply splash"
        apply_splash(model, image_path=args.image)
