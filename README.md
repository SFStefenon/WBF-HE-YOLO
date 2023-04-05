# Weighted Boxes Fusion Hypertuned Explainable YOLOv8

This repository presents the combination of methods for obtaining an optimized you only look once (YOLO) model.

---

### To perform the hypetuning of YOLO the **Optuna framework** is applied.

There are two ways to compute the Optuna, the first is locally on a cluster or your PC and the second is using Google Colab.

> If you decide to use Colab, please go ahead and try it on using this [notebook](https://github.com/SFStefenon/WBF-HE-YOLO/blob/main/Hypertuning_Optuna/Google_Colab_Computing/YOLOv8_Optuna.ipynb)! You will be asked to confirm your access to the drive where the data will be saved.

> If you want to use a local machine, you can follow this Python-based [algorithm](https://github.com/SFStefenon/WBF-HE-YOLO/blob/main/Hypertuning_Optuna/Cluster_Computing/yolov8_insulator_exp1.py). Using a Cluster the study is gonna be saved and you can evaluate latter using [Colab](https://github.com/SFStefenon/WBF-HE-YOLO/blob/main/Hypertuning_Optuna/Cluster_Computing/Experiment_Results/Optuna_Results.ipynb).


OBS: Since the analysis is using a deep learning based model, depending on your dataset a high processing time will be required for the model to perform the defined epochs. The file that calls the dataset must be in the same main folder of the model.

---

### For architecture optimization the **weighted box fusion (WBF)** is used.

---

### For interpretability the **xxx** explainable AI (XAI) is presented.

---

The model presented in this repository was evaluated using the dataset realeased by Dexter Lewis and Pratik Kulkarni, which can be found at [competition-insulator-defect-detection](https://dx.doi.org/10.21227/vkdw-x769) (accessed on March 25, 2023).

**We encourage you to perform comparative analyzes with your dataset!**

---

### Compute YOLOv8 in your machine

The first step is to download the YOLOv8. I recommend doing that from the official developer [Ultralytics](https://github.com/ultralytics/ultralytics) repository.
This version is based on PyTorch, beeing available for your machine or Google Colab.



The first step to compute YOLO in your PC or in a cluster is to create the environment:

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



---


Wrote by Dr. **Stefano Frizzo Stefenon** and Dr. **Laio Oriel Seman**.

Trento, Italy, March 10, 2023.
