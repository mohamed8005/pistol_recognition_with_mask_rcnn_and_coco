"""
    loss log (val and train)
    python .\custom4.py loss --log=.\training_log2.txt --dataset=.\dataset --weights=last

    confusion matrix :
    python .\custom4.py evaluate --dataset=.\dataset --weights=last
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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import re

# Configure TensorFlow 1.15 for compatibility
tf.compat.v1.disable_eager_execution()

# GPU configuration and memory growth
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# Check for GPU availability in TensorFlow 1.15
if tf.test.is_gpu_available():
    print("GPU is available")
else:
    print("GPU is not available, using CPU")

# Root directory of the project
ROOT_DIR = os.path.abspath("..\\..\\")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Custom configuration for the model
class CustomConfig(Config):
    NAME = "pistol"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + pistol
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

# Custom dataset loader
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

# Plot losses function
def plot_losses(history):
    if history:
        plt.plot(history.get('loss', []), label='train loss')
        plt.plot(history.get('val_loss', []), label='validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    else:
        print("No training history found.")
def plot_losses_log_file(log_file_path):
    training_losses = []
    validation_losses = []

    # Patterns for extracting training and validation loss
    train_loss_pattern = re.compile(r'/step - loss: ([\d.]+) - rpn_class_loss:')
    val_loss_pattern = re.compile(r' - val_loss: ([\d.]+) - val_rpn_class_loss:')
    # Read log file and extract loss values
    with open(log_file_path, 'r') as f:
        for line in f:
            # print(line)
            train_loss_match = train_loss_pattern.search(line)
            if train_loss_match:
                training_losses.append(float(train_loss_match.group(1)))
                # print("aloalo " +str(training_losses))

            val_loss_match = val_loss_pattern.search(line)
            if val_loss_match:
                validation_losses.append(float(val_loss_match.group(1)))
                # print("aloalo " +str(validation_losses))

    # Plot the extracted losses
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss from Log File")
    plt.legend()
    plt.grid(True)
    
    # Save the plot as a PNG file
    plt.savefig("loss_plot.png")
    plt.show()
# def plot_losses_log_file(log_file_path):
#     training_losses = []
#     validation_losses = []

#     # Read log file and extract loss values
#     with open(log_file_path, 'r') as f:
#         for line in f:
#             # Split the line by spaces
#             parts = line.split()

#             # Check if 'loss:' and 'val_loss:' are in the line
#             if 'loss:' in parts and 'val_loss:' in parts:
#                 # Find indices of 'loss:' and 'val_loss:'
#                 try:
#                     # Training loss is located immediately after 'loss:'
#                     train_loss_index = parts.index('loss:') + 1
#                     training_losses.append(float(parts[train_loss_index]))
#                     print("Training Loss Found:", training_losses[-1])  # Debug print

#                     # Validation loss is located immediately after 'val_loss:'
#                     val_loss_index = parts.index('val_loss:') + 1
#                     validation_losses.append(float(parts[val_loss_index]))
#                     print("Validation Loss Found:", validation_losses[-1])  # Debug print
#                 except (ValueError, IndexError) as e:
#                     print(f"Error extracting losses: {e}")

#     # Plot the extracted losses
#     plt.figure(figsize=(10, 6))
#     plt.plot(training_losses, label="Training Loss")
#     plt.plot(validation_losses, label="Validation Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Training and Validation Loss from Log File")
#     plt.legend()
#     plt.grid(True)
    
#     # Save the plot as a PNG file
#     plt.savefig("loss_plot.png")
#     plt.show()
# Confusion matrix calculation
def calculate_confusion_matrix(model, dataset_val):
    actuals, predictions = [], []
    
    for image_id in dataset_val.image_ids:
        image = dataset_val.load_image(image_id)
        mask, class_ids = dataset_val.load_mask(image_id)
        results = model.detect([image], verbose=1)
        pred_class_ids = results[0]['class_ids']

        # Check for mismatched lengths between actuals and predictions
        if len(pred_class_ids) > len(class_ids):
            # If more predictions than actuals, add actuals and pad with background class (0)
            actuals.extend(class_ids.flatten())
            predictions.extend(pred_class_ids[:len(class_ids)].flatten())
            actuals.extend([0] * (len(pred_class_ids) - len(class_ids)))  # Padding with background
            predictions.extend(pred_class_ids[len(class_ids):].flatten())
        elif len(class_ids) > len(pred_class_ids):
            # If more actuals than predictions, add predictions and pad with background class (0)
            actuals.extend(class_ids[:len(pred_class_ids)].flatten())
            predictions.extend(pred_class_ids.flatten())
            predictions.extend([0] * (len(class_ids) - len(pred_class_ids)))  # Padding with background
        else:
            # If lengths match, simply extend both lists
            actuals.extend(class_ids.flatten())
            predictions.extend(pred_class_ids.flatten())

    # Calculate and display the confusion matrix
    cm = confusion_matrix(actuals, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Training function
def train(model, dataset_dir, epochs=20, learning_rate=0.001):
    dataset_train = CustomDataset()
    dataset_train.load_custom(dataset_dir, "train")
    dataset_train.prepare()

    dataset_val = CustomDataset()
    dataset_val.load_custom(dataset_dir, "val")
    dataset_val.prepare()

    print("Training network heads")
    history = model.train(
        dataset_train, dataset_val,
        learning_rate=learning_rate,
        epochs=epochs,
        layers='heads'
    )

    # Save training history and plot losses
    plot_losses(history.history)

# Independent evaluation function
def evaluate_model(model, dataset_dir):
    # Load validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(dataset_dir, "val")
    dataset_val.prepare()

    # Display confusion matrix
    calculate_confusion_matrix(model, dataset_val)

    # Assuming previous training losses are available to plot
    # Plot losses if history is available
    # You can load previous training history if saved as a file
    # plot_losses(previous_history)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or evaluate Mask R-CNN model.")
    parser.add_argument("command", metavar="<command>", help="'train', 'evaluate', or 'loss'")
    parser.add_argument("--dataset", required=False, metavar="/path/to/dataset/", help="Directory of the dataset")
    parser.add_argument("--weights", required=True, metavar="/path/to/weights.h5", help="Path to weights file")
    parser.add_argument("--epochs", required=False, default=20, type=int, help="Number of epochs")
    parser.add_argument("--learning_rate", required=False, default=0.001, type=float, help="Learning rate")
    parser.add_argument("--log", required=False, default=0.001, metavar="/path/to/log_file.txt", help="log_file")
    args = parser.parse_args()

    config = CustomConfig()
    model_dir = DEFAULT_LOGS_DIR
    model = modellib.MaskRCNN(mode="training" if args.command == "train" else "inference", config=config, model_dir=model_dir)

    # Load weights
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_mask"])
    elif args.weights.lower() == "last":
        weights_path = model.find_last()
        model.load_weights(weights_path, by_name=True)
    else:
        weights_path = args.weights
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate based on command
    if args.command == "train":
        train(model, args.dataset, epochs=args.epochs, learning_rate=args.learning_rate)
    elif args.command == "evaluate":
        evaluate_model(model, args.dataset)
    elif args.command == "loss":
        if args.log:
            plot_losses_log_file(args.log)
        else:
            print("Please provide a path to the log file using '--log' argument.")