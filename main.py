#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ultralytics import YOLO
import os
import cv2
from PIL import Image
import shutil
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
import pandas as pd
import re
import torch; print(torch.cuda.is_available())


# In[2]:


images_dir = "C:/Users/YusufEmirComert/Desktop/VsCode/CV/Ass3/cars_dataset/Images"
ann_dir = "C:/Users/YusufEmirComert/Desktop/VsCode/CV/Ass3/cars_dataset/Annotations"
imagesets_dir = "C:/Users/YusufEmirComert/Desktop/VsCode/CV/Ass3/cars_dataset/ImageSets"


# In[3]:


# Create the output directory if it doesn't exist
output_dir = "C:/Users/YusufEmirComert/Desktop/VsCode/CV/Ass3/cars_dataset_yolo"
os.makedirs(output_dir, exist_ok=True)


# In[4]:


# create the images and labels directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)


# In[5]:


# convert the annotations to YOLO format
def convert_to_yolo(x1, y1, x2, y2, img_w, img_h):
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height


# In[6]:


# Iterate through the splits and copy images and annotations
for split in ['train', 'val', 'test']:
    with open(os.path.join(imagesets_dir, f"{split}.txt")) as f:
        lines = f.read().splitlines()
    
    for name in lines:
        # Copy the image
        src_img = os.path.join(images_dir, f"{name}.png")
        dst_img = os.path.join(output_dir, 'images', split, f"{name}.png")
        shutil.copy2(src_img, dst_img)

        # Open the image to get its dimensions
        img = Image.open(src_img)
        w, h = img.size

        # Copy the annotation file
        ann_file = os.path.join(ann_dir, f"{name}.txt")
        dst_label = os.path.join(output_dir, 'labels', split, f"{name}.txt")
        with open(ann_file, 'r') as af, open(dst_label, 'w') as yf:
            for line in af:
                x1, y1, x2, y2, cls = map(int, line.strip().split())
                x_c, y_c, width, height = convert_to_yolo(x1, y1, x2, y2, w, h)
                yf.write(f"0 {x_c:.6f} {y_c:.6f} {width:.6f} {height:.6f}\n")


# In[7]:


# create the data.yaml file
yaml_content = """
path: C:/Users/YusufEmirComert/Desktop/VsCode/CV/Ass3/cars_dataset_yolo
train: images/train
val: images/val
test: images/test

names:
  0: car
"""
# write the yaml content to a file
with open("data.yaml", "w") as f:
    f.write(yaml_content)


# In[8]:


# pre-trained model
model = YOLO('yolov8n.pt')  


# In[40]:


def evaluate_model(model_path):
    model = YOLO(model_path)
    
    # === YOLO Metrics ===
    metrics = model.val(data='data.yaml', split='test')
    print("\n YOLO Val Metrics:")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")

    # Calculating Exact Match Accuracy and Normal Accuracy
    print("\n Calculating Counting Accuracy Metrics...")
    test_dir = 'cars_dataset_yolo/images/test'
    label_dir = 'cars_dataset_yolo/labels/test'
    test_images = [f for f in os.listdir(test_dir) if f.endswith('.png')]

    # Initialize variables
    exact_matches = 0
    total_preds = 0
    total_gts = 0 # ground truth (GT)
    sum_correct = 0  
    squared_errors = []

    for img_name in tqdm(test_images):
        img_path = os.path.join(test_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace('.png', '.txt'))

        gt_count = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                gt_count = len(f.readlines())

        preds = model.predict(img_path, verbose=False)
        pred_count = len(preds[0].boxes)

        if pred_count == gt_count:
            exact_matches += 1

        total_preds += pred_count
        total_gts += gt_count

        # Calculate the correct counts
        correct_count = min(pred_count, gt_count)
        sum_correct += correct_count

        squared_errors.append((pred_count - gt_count) ** 2)

    exact_acc = exact_matches / len(test_images) * 100 if len(test_images) > 0 else 0
    normal_acc_v2 = (sum_correct / total_gts * 100) if total_gts > 0 else 0
    mse = np.mean(squared_errors) if squared_errors else 0

    print(f"\n Exact Match Accuracy: {exact_acc:.2f}%")
    print(f" Normal Accuracy (Correct counts / GT counts): {normal_acc_v2:.2f}%")
    print(f" Mean Squared Error (MSE): {mse:.4f}")


# In[10]:


# Visualize predictions
# Save predictions to a directory
# and visualize them

def visualize_predictions(model_path, save_dir='predictions'):
    model = YOLO(model_path)
    os.makedirs(save_dir, exist_ok=True)
    test_dir = 'cars_dataset_yolo/images/test'
    test_images = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    for img_name in tqdm(test_images[:10], desc="Saving predictions"):
        img_path = os.path.join(test_dir, img_name)
        model(img_path, save=True, project=save_dir, name='examples', exist_ok=True)


# General Main Function to Train the Model

# In[11]:


# Train model 
def run_experiment(freeze_value, data_yaml='data.yaml', epochs=50, batch=16, imgsz=640):
    print(f"\n Training with freeze={freeze_value} blocks...")

    model = YOLO('yolov8n.pt')
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        freeze=freeze_value,
        imgsz=imgsz,
        name=f"freeze_{freeze_value}",  # folders to save the results
        verbose=False
    )

    # find the last directory
    run_dir = f"C:/Users/YusufEmirComert/runs/detect/freeze_{freeze_value}"
    if not os.path.exists(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # # Model Path
    best_model_path = os.path.join(run_dir, 'weights', 'best.pt')
    visualize_predictions(best_model_path)


# TRAINING THE MODEL WITH 4 DIFFERENT FREEZE VALUES

# In[12]:


# Freeze blocks and train
for freeze_val in [5, 10, 21, 0]:
    run_experiment(freeze_val)


# Evaluating Results.

# In[13]:


# Evaluate the models for each freeze value
freeze_vals = [5, 10, 21, 0]

for freeze in freeze_vals:
    print(f"\n Analyzing freeze={freeze} results...\n")
    run_dir = f"C:/Users/YusufEmirComert/runs/detect/freeze_{freeze}"


    # Evaluations
    best_model_path = os.path.join(run_dir, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        evaluate_model(best_model_path)
    else:
        print(f" best.pt has not found: {best_model_path}")


# In[14]:


# Function to plot loss curves
def plot_loss_curves(run_dir, save_root):
    csv_path = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(csv_path):
        print(f"File NOT FOUND: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss (Train)')
    plt.plot(df['epoch'], df['train/cls_loss'], label='Class Loss (Train)')
    plt.plot(df['epoch'], df['val/box_loss'], label='Box Loss (Val)', linestyle='--')
    plt.plot(df['epoch'], df['val/cls_loss'], label='Class Loss (Val)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve: {os.path.basename(run_dir)}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # If not exist, create the directory
    os.makedirs(save_root, exist_ok=True)

    # Save the plot
    freeze_val = os.path.basename(run_dir).split('_')[-1]  # freeze_5 -> 5
    save_path = os.path.join(save_root, f'loss_curve_freeze_{freeze_val}.png')

    plt.savefig(save_path)
    print(f"Loss curve saved to: {save_path}")

    plt.show()


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Run the loss curve plotting for each freeze value
freeze_vals = [5, 10, 21, 0]
save_root = r"C:\Users\YusufEmirComert\Desktop\VsCode\CV\Ass3\results"

for freeze in freeze_vals:
    run_dir = f"C:/Users/YusufEmirComert/runs/detect/freeze_{freeze}"
    print(f"Loss curve for freeze={freeze}:")
    plot_loss_curves(run_dir, save_root)


# In[43]:


def run_experiment_custom(freeze_value, optimizer, lr, batch, imgsz, mosaic, epochs, data_yaml):
    run_name = f"freeze{freeze_value}_opt{optimizer}_lr{lr}_batch{batch}_mosaic{mosaic}_imgsz{imgsz}"
    project_dir = "runs/detect/"
    run_dir = os.path.join(project_dir, run_name)

    print(f"\nRunning: {run_name}")
    model = YOLO("yolov8n.pt")

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr,
        optimizer=optimizer,
        project=project_dir,
        name=run_name,
        freeze=freeze_value,
        mosaic=mosaic,
        device='cuda',
        workers=2,
        save=True,
        save_period=-1,
        verbose=True
    )

    best_model_path = os.path.join(run_dir, 'weights', 'best.pt')
    if not os.path.exists(best_model_path):
        print(f"!!Model not saved: {best_model_path}!!")
    else:
        print(f"Model saved at: {best_model_path}")


# In[44]:


def run_general_hyperparam_tests():
    freeze = 10
    epochs = 25
    data_yaml = 'data.yaml'

    optimizers = ['AdamW', 'SGD']
    lrs = [0.005, 0.01]
    batches = [16, 32]
    imgsz = 640
    mosaic = False

    print("Running General Hyperparameter Tests...")
    for opt in optimizers:
        for lr in lrs:
            for batch in batches:
                run_experiment_custom(freeze_value=freeze, optimizer=opt, lr=lr, batch=batch, imgsz=imgsz, mosaic=mosaic, epochs=epochs, data_yaml=data_yaml)



# In[45]:


def run_yolo_specific_hyperparam_tests():
    freeze = 10
    epochs = 25
    data_yaml = 'data.yaml'

    optimizer = 'AdamW'
    lr = 0.01
    batch = 32

    mosaics = [True, False]
    imgsizes = [640, 320]

    print("Running YOLO-specific Hyperparameter Tests...")
    for mosaic_flag in mosaics:
        for size in imgsizes:
            run_experiment_custom(freeze_value=freeze, optimizer=optimizer, lr=lr, batch=batch, imgsz=size, mosaic=mosaic_flag, epochs=epochs, data_yaml=data_yaml)


# In[46]:


run_general_hyperparam_tests()


# In[47]:


run_yolo_specific_hyperparam_tests()


# In[48]:


def parse_hyperparams_from_name(name):
    try:
        parts = name.split('_')
        freeze = int(parts[0].replace('freeze', ''))
        opt = parts[1].replace('opt', '')
        lr = float(parts[2].replace('lr', ''))
        batch = int(parts[3].replace('batch', ''))
        mosaic = parts[4].replace('mosaic', '') == 'True'
        imgsz = int(parts[5].replace('imgsz', ''))
        return {
            'freeze': freeze,
            'optimizer': opt,
            'lr': lr,
            'batch': batch,
            'mosaic': mosaic,
            'imgsz': imgsz
        }
    except Exception as e:
        print(f"   [!] Parsing failed for '{name}': {e}")
        return None


# In[49]:


# === Main Evaluation Loop ===
root_dir = "C:/Users/YusufEmirComert/Desktop/VsCode/CV/Ass3/runs/detect/"

print("\n===== Evaluating General Hyperparameter Runs =====\n")

for folder_name in os.listdir(root_dir):
    full_path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(full_path):
        continue
    print(f"Checking folder: {folder_name}")
    if not folder_name.startswith("freeze") or "opt" not in folder_name:
        continue

    params = parse_hyperparams_from_name(folder_name)
    if not params:
        print(f"   Folder name skipped (unmatched): {folder_name}")
        continue

    print(f"\n>>> Analyzing: {folder_name}")
    print(f"   Params: {params}")

    best_model_path = os.path.join(full_path, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        print(f"   ✓ Evaluating model: {best_model_path}")
        evaluate_model(best_model_path)
    else:
        print(f"   best.pt not found in: {best_model_path}")


# <div style="font-family: Arial; font-size: 15px; line-height: 1.6; max-width: 900px;">
# 
# ###  <b>Assignment 3: YOLOv8-Based Object Detection Report</b>
# 
# ---
# 
# #### <b>1. Step-by-Step Explanation</b>
# 
# Workflow of the assignment.
# 
# 1. **Setting Up Paths and Directories:**  
#    The original dataset paths for `Images`, `Annotations`, and `ImageSets` were defined. A new output directory (`cars_dataset_yolo`) was created to store the converted dataset in YOLO format.
# 
# 2. **Creating Folder Structure:**  
#    Separate folders for `train`, `val`, and `test` splits were created under both `images` and `labels` directories inside the output directory. This organization matches YOLO’s expected dataset structure.
# 
# 3. **Converting Annotations to YOLO Format:**  
#    The original bounding box coordinates in VOC format (`x1, y1, x2, y2`) were converted to YOLO format — normalized center coordinates (`x_center`, `y_center`) and normalized width and height.
# 
# 4. **Copying Images and Converting Annotations:**  
#    For each data split (`train`, `val`, `test`), image filenames were read from the corresponding text files. Images were copied to the new folder structure, and annotation files were read, converted to YOLO format, and saved in the corresponding labels folder.
# 
# 5. **Creating the Data Configuration File (`data.yaml`):**  
#    A YAML configuration file was created to specify the dataset paths and class names. This file is used by YOLOv8 during training and evaluation.
# 
# 6. **Loading the Pre-trained Model and Training:**  
#    The pretrained `yolov8n.pt` model was loaded. Training was performed multiple times with different numbers of frozen layers (`freeze=5, 10, 21, 0`), each for 50 epochs, to compare performance.
# 
# 7. **Model Evaluation:**  
#    The trained models were evaluated on the test set using YOLO metrics such as Precision, Recall, mAP50, and mAP50-95. Additionally, counting-based accuracy metrics were calculated: Exact Match Accuracy (predicted count equals ground truth), Normal Accuracy (correct counts over ground truth counts), and Mean Squared Error (MSE).
# 
# 8. **Visualizing Predictions:**  
#    Predictions on test images were saved and visualized to qualitatively assess model performance.
# 
# 9. **Comparing Results Across Experiments:**  
#    Performance metrics from different freezing configurations were compared to identify the optimal number of frozen layers, where `freeze=10` gave the best results in this case.
# 
# 
# 
# ---
# 
# #### <b>2. Loss Curves (Train & Validation)</b>
# Plots were generated from `results.csv` files and display both training and validation losses:
# - Box Loss (train/val)
# - Class Loss (train/val)
# 
# *Example:*
# ![Loss Curve](./results/loss_curve_freeze_0.png)
# 
# *Example:*
# ![Loss Curve](./results/loss_curve_freeze_5.png)
# 
# *Example:*
# ![Loss Curve](./results/loss_curve_freeze_10.png)
# 
# *Example:*
# ![Loss Curve](./results/loss_curve_freeze_21.png)
# 
# 
# 
# ---
# 
# #### <b>3. General Learning Hyperparameters</b>
# - Optimizer: AdamW (auto-selected)
# - Batch size: 16
# - Learning rate: 0.01 (overridden by auto optimizer)
# - Image size: 640 (stabled the image size for faster training)
# - Best performing setup: `freeze=10` with smooth convergence, highest exact accuracy and better result
# 
# ---
# 
# #### <b>4. YOLO-Specific Hyperparameters</b>
# - `freeze`: Key hyperparameter in this study (5, 10, 21, 0)
# - `mosaic`, `hsv_*`, `erasing`: Used default values
# - Larger freeze values tend to help with generalization; `freeze=10` performed best.
# 
# ---
# 
# #### **5. Exploring the Effects of Hyperparameter Changes**
# 
# In this section, we systematically explore the impact of various general and YOLO-specific hyperparameters on the model's performance. The aim is to evaluate how each setting affects precision, recall, mAP, exact match accuracy, and MSE.
# 
# ##### **5.1 General Learning Hyperparameter Tests**
# 
# We conducted a grid search over the following configurations:
# 
# - **Optimizers**: `AdamW`, `SGD`  
# - **Learning Rates**: `0.005`, `0.01`  
# - **Batch Sizes**: `16`, `32`  
# - **Image Size**: Fixed at `640`  
# - **Mosaic Augmentation**: Disabled (`False`)  
# - **Frozen Layers**: First 10 layers (`freeze=10`)  
# - **Epochs**: `25`
# 
# Each combination was tested using the `run_experiment_custom` function, resulting in a total of 8 experiments (2 optimizers × 2 learning rates × 2 batch sizes). These tests provided insight into the model's sensitivity to optimizer choice and training parameters.
# 
# ##### **5.2 YOLO-specific Hyperparameter Tests**
# 
# In a separate set of experiments, I focused on hyperparameters that are particularly relevant to the YOLOv8 architecture:
# 
# - **Optimizer**: `AdamW`  
# - **Learning Rate**: `0.01`  
# - **Batch Size**: `32`  
# - **Frozen Layers**: `10`  
# - **Epochs**: `25`  
# - **Mosaic Augmentation**: `True` or `False`  
# - **Image Size**: `640` or `320`  
# 
# This test matrix evaluates how the YOLOv8-specific features such as input resolution and mosaic augmentation affect training dynamics and final performance. A total of 4 experiments were run for all mosaic–image size combinations.
# 
# > Detailed results and analysis of these experiments are provided in Section 7: *Comparative Evaluation*.
# 
# ---
# 
# #### <b>6. Prediction Visualization</b>
# Bounding boxes were visualized for test images.  
# Example predictions saved in: `predictions/examples`
# 
# ✅ Correct Detections  
# ❌ Missed small or overlapping cars in some images
# 
# *Example:✅*
# ![Example 1](predictions/examples/20160331_NTU_00008.jpg)
# 
# *Example:❌*
# ![Example 2](predictions/examples/20160331_NTU_00055.jpg)
# 
# As you can see model could not find the car with the red marker due to from lack of visibility.
# 
# *Example:✅*
# ![Example 3](predictions/examples/20160331_NTU_00078.jpg)
# 
# *Example:❌*
# ![Example 4](predictions/examples/20160331_NTU_00033.jpg)
# As you can see model could not find the car with the red marker due to vehicle is halfly showen.
# 
# ---
# 
# #### <b>7. Comparative Evaluation</b>
# 
# This table shows the outputs of default project output(not the experiments)
# 
# | Freeze | Exact Match (%) | MSE      | Precision | Recall  |
# |--------|-----------------|----------|-----------|---------|
# | 5      | 39.50           | 3.3850   | 0.9919    | 0.9814  |
# | 10     | 43.50           | 4.8850   | 0.9914    | 0.9784  |
# | 21     | 4.00            | 111.7850 | 0.9745    | 0.9363  |
# | 0      | 41.50           | 2.3050   | 0.9921    | 0.9863  |
# 
# 
# **Observation:**  
# Among the different freezing configurations, **freeze=10** achieved the best balance of accuracy and stability in detection performance, with the highest Exact Match accuracy at 43.5%. Although its MSE (4.8850) is slightly higher than freeze=0 (2.3050), the precision and recall remain very high (0.9914 and 0.9784), indicating strong detection consistency.
# 
# The **freeze=5** setup shows competitive precision and recall but slightly lower exact match accuracy and a moderate MSE, reflecting somewhat less stable learning.
# 
# The **freeze=0** (no freezing) scenario yields the lowest MSE and high recall but slightly lower exact match accuracy compared to freeze=10. This indicates the model may overfit slightly less but at the cost of exact match stability.
# 
# The **freeze=21** case exhibits a drastic drop in exact match accuracy (only 4%), with extremely high MSE (111.7850), and noticeably lower precision and recall compared to other setups. This indicates the model failed to effectively learn due to freezing too many layers, severely limiting the model’s adaptability to the new dataset. Such excessive freezing restricts weight updates mostly to the final layers, preventing the network from capturing domain-specific features, leading to poor detection results.
# 
# Overall, **freeze=10** stands out as the optimal setting, balancing effective transfer learning from pretrained weights with the ability to fine-tune sufficiently for the task, yielding the best overall detection and counting performance.
# 
# 
# This is the general and yolo specific experiment results:
# 
# **General Hyperparameter Comparison**
# (freeze = 10, mosaic = false, Image size = 640 fixed)
# 
# | # | Optimizer | Learning Rate  | Batch Size  | Exact Match (%) | MSE     | Precision | Recall  |
# |---|-----------|----------------|-------------|-----------------|---------|-----------|---------|
# | 1 | SGD       | 0.005          | 16          | 25.50           | 14.33   | 0.9879    | 0.9609  |
# | 2 | SGD       | 0.005          | 32          | 27.50           | 17.10   | 0.9891    | 0.9593  |
# | 3 | SGD       | 0.01           | 16          | 31.00           | 11.57   | 0.9895    | 0.9684  |
# | 4 | SGD       | 0.01           | 32          | 31.00           | 11.96   | 0.9898    | 0.9678  |
# | 5 | Adam      | 0.005          | 16          | 31.00           | 09.02   | 0.9904    | 0.9735  |
# | 6 | Adam      | 0.005          | 32          | 39.00           | 06.22   | 0.9898    | 0.9737  |
# | 7 | Adam      | 0.01           | 16          | 39.00           | 06.52   | 0.9922    | 0.9751  |
# | 8 | Adam      | 0.01           | 32          | 36.00           | 06.04   | 0.9894    | 0.9740  |
# 
# - As you can see 6 and 7'th experiment gave the highest exact accuracy which is 39.00%. 
# 
# **Yolo-Specific Hyperparameter Comparison**
# Experiment Results (Freeze=10, Optimizer=AdamW, LR=0.01, Batch=32 fixed)
# 
# | Mosaic | Image Size | Precision | Recall | Exact Match Acc. | MSE     |
# |--------|------------|-----------|--------|------------------|---------|
# | False  | 640        | 0.9894    | 0.9740 | 36.00%           | 6.0400  |
# | False  | 320        | 0.9594    | 0.9304 | 11.00%           | 332.9100|
# | True   | 320        | 0.9671    | 0.9311 | 16.50%           | 195.2150|
# | True   | 640        | 0.9865    | 0.9763 | 42.00%           | 7.0600  |
# 
# - In this part, we can not directly say that mosaic differs but we can clearly say that Image size has a lot of affect to the detecting the objects. in the low image size(320), the models gave the worst results.
# 
# ---
# 
# #### <b>8. Results</b>
# 
# - The model's performance is visualized using Precision-Recall (PR) curves and confusion matrices to better understand detection quality and counting accuracy.
# 
# - **Precision-Recall Curve:**  
#   ![PR Curve](results\PR_curve.png)
# 
# - **Confusion Matrix:**  
#   ![Confusion Matrix](results\confusion_matrix.png)
# 
# - These plots show the trade-off between precision and recall at different confidence thresholds and highlight where the model tends to confuse detections.
# - As you can see, the results are almost perfect which indicates that model is good.
# 
# 
# ---
# #### <b>9. Alternative Methods That Might Improve The Model</b>
# - **Regression CNNs:**  
#   These models take an image and directly predict the total number of objects without detecting each one separately. They learn to estimate the count from the overall features of the image. This method is simple and fast but may not work well when objects overlap or are very close to each other.
# 
# - **Density Map Models:**  
#   Instead of predicting the count directly, these models create a density map where each pixel shows how likely it is to contain part of an object. The sum of all pixel values in the map gives the total count. This approach is useful for crowded scenes where objects are densely packed, such as counting people in a crowd.
# 
# - **Transformers:**  
#   Transformer models use attention mechanisms to understand the relationships between objects in the image. They can detect objects and count them by considering the whole image context. DETR is an example that combines detection and counting with good accuracy, especially in complex scenes.
# 
# - **Instance Segmentation:**  
#   These methods find the exact shape of each object in the image, not just bounding boxes. Mask R-CNN is a popular model that does this. It works well when objects overlap or are very close, providing precise counting and location information. However, it can be slower and requires more computation.
# 
# ---
# 
# <b>✅ Conclusion:</b>  
# Using YOLOv8 with partial layer freezing is effective for object counting, balancing speed and accuracy. The method shows high precision and low error rates. In the future, combining YOLOv8 with density map models or transformers could improve counting in very crowded or difficult scenes. Data augmentation and model fine-tuning can also help increase robustness.
# 
# ---
# 
# 
# </div>
# 
