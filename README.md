# Weighted Boxes Fusion Hypertuned Explainable YOLOv8

This repository presents the combination of methods for obtaining an optimized you only look once (YOLO) model.

---

## To perform the hypetuning of YOLO the **Optuna framework** is applied.

There are two ways to compute the Optuna, the first is locally on a cluster or your PC and the second is using Google Colab.

> If you decide to use Colab, please go ahead and try it on using this [notebook](https://github.com/SFStefenon/WBF-HE-YOLO/blob/main/Hypertuning_Optuna/Google_Colab_Computing/YOLOv8_Optuna.ipynb)! You will be asked to confirm your access to the drive where the data will be saved.

> If you want to use a local machine, you can follow this Python-based [algorithm](https://github.com/SFStefenon/WBF-HE-YOLO/blob/main/Hypertuning_Optuna/Cluster_Computing/yolov8_insulator_exp1.py). Using a Cluster the study is gonna be saved and you can evaluate latter using [Colab](https://github.com/SFStefenon/WBF-HE-YOLO/blob/main/Hypertuning_Optuna/Cluster_Computing/Experiment_Results/Optuna_Results.ipynb).


OBS: Since the analysis is using a deep learning based model, depending on your dataset a high processing time will be required for the model to perform the defined epochs. The file that calls the dataset must be in the same main folder of the model.

---

## For architecture optimization the **weighted box fusion (WBF)** is used.

---

## For interpretability the **xxx** explainable AI (XAI) is presented.

---

## Compute YOLOv8 in your machine

The first step is to download the YOLOv8. I recommend doing that from the official developer [Ultralytics](https://github.com/ultralytics/ultralytics).
This version is based on PyTorch, being available for your machine or Google Colab.

The second step to compute YOLO in your machine is to create the environment:

```
# Enter in the folder of your project
cd ~~~ 

# Check the environments available
conda env list

# Create a new environment
conda create --name yolo-env python=3.9

# Activate the environment to install packges
conda activate yolo-env

# Install the requirements
pip install -r requirements.txt

# If something goes wrong and you need to remove the environmen
conda env remove -n yolo-env
```

The third step is to organize your dataset.

```
# Example of how to upload your dataset in a Cluster
scp -r C:/Users/user_name/Desktop/dataset/ cluster:/home/user_name/dataset/
```
There are tree ways to organize your personalized dataset.
The first way is using a different path for training and validation (works for YOLOv5 and YOLOv7).

### Organize Your Dataset

There are tree ways to organize your personalized dataset.

The first way is using a different path for training and validation (works for YOLOv5 and YOLOv7).
```    
dataset/train/images
dataset/train/labels
dataset/valid/images
dataset/valid/labels
```

The second way is using a different path for training and validation (works specially for YOLOv6).
```    
dataset/images/train
dataset/images/val
dataset/labels/train
dataset/labels/val
```

The third way is to place everything in the same place and call it by a `.txt` file (works only for YOLOv5).
```
dataset/train.txt
dataset/valid.txt
```

**I highly recommend that you use the second way, as it's going to be easier to change what you are using for training and testing.**

In the `train.txt` you will have the path of all your pictures, one by one like:
```
diclub:/home/sfrizzostefenon/dataset/RFI_640_110c10_0.jpg
```

When you have decided how to organize your data, you will need to change how the model loads it.

This is going to be in the file that you use to call the script that loads your data.

In the `train.py` there is (where you define how the data will be loaded):
```
parser.add_argument('--data', type=str, default=ROOT / 'data/mydata.yaml', help='dataset.yaml path') 
```

This file is going to be in the data folder inside the YOLO model.

Depending on how you decided to organize the data the `mydata.yaml` will look like this:
```
path: ../dataset
train: train/images
val: valid/images
```
or
```
path: ../dataset
train: train.txt
val: val.txt
```

**If you use this structure the model will load the labels automatically based on their names.**

OBS: Here the test is optional because it will be performed after training.
```
# test images (optional)
```


---

### Used Database

The model presented in this repository was evaluated using the dataset realeased by Dexter Lewis and Pratik Kulkarni, which can be found at [competition-insulator-defect-detection](https://dx.doi.org/10.21227/vkdw-x769) (accessed on March 25, 2023).

**We encourage you to perform comparative analyzes with your dataset!**

---

### Create Your Custom Dataset

To create a custom dataset with the goal of object detection it is necessary to use an image labeling algorithm or software.

I recommend to use the [labelImg](https://github.com/heartexlabs/labelImg), it's based on Python, so it's light and easy to use.
LabelImg is a graphical image annotation tool written in Python.

After you download the algorithm you can run the `labelImg.py`

In the `data/predefined_classes.txt` you can define the classes that you are going to use. 
This will allocate the spaces in the memory, therefore the number in the annotation will follow this order.

Later on, the classes that you have used have to match with `mydata.yaml`, that you previously created as this example:

```
nc: 4 
names: ['broken_shell', 'flashover_damage', 'good_insulator', 'insulator_string']
```

---

Wrote by Dr. **Stefano Frizzo Stefenon** and Dr. **Laio Oriel Seman**.

Trento, Italy, March 10, 2023.
