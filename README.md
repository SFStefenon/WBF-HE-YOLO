# Weighted Boxes Fusion Hypertuned Explainable YOLOv8

This repository presents the combination of methods for obtaining an optimized you only look once (YOLO) model.

---

### To perform the hypetuning of YOLO the **Optuna framework** is applied.

There are two ways to compute the Optuna, the first is locally on a cluster or your PC and the second is using Google Colab.

> If you decide to use Colab, please go ahead and try it on [here](https://github.com/SFStefenon/WBF-HE-YOLO/blob/main/Hypertuning_Optuna/Google_Colab_Computing/YOLOv8_Optuna.ipynb)!

> If you want to use a local machine, you can follow this Python-based [algorithm](https://github.com/SFStefenon/WBF-HE-YOLO/blob/main/Hypertuning_Optuna/Cluster_Computing/yolov8_insulator_exp1.py). Using a Cluster the study is gonna be saved and you can evaluate latter using [Colab](https://github.com/SFStefenon/WBF-HE-YOLO/blob/main/Hypertuning_Optuna/Cluster_Computing/Experiment_Results/Optuna_Results.ipynb).


OBS: Since the analysis is using a deep learning based model, depending on your dataset a high processing time will be required for the model to perform the defined epochs.

---

### For architecture optimization the **weighted box fusion (WBF)** is used.

---

### For interpretability the **xxx** explainable AI (XAI) is presented.

---

The model presented in this repository was evaluated using the dataset realeased by Dexter Lewis and Pratik Kulkarni, which can be found at [competition-insulator-defect-detection](https://dx.doi.org/10.21227/vkdw-x769) (accessed on March 25, 2023).

**We encourage you to perform comparative analyzes with your dataset!**

---


Wrote by Dr. **Stefano Frizzo Stefenon** and Dr. **Laio Oriel Seman**.

Trento, Italy, March 10, 2023.
