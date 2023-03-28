import optuna
import re
import numpy as np
from ultralytics import YOLO
import time

start = time.time()

def attention(model_opt, dnn, aug):
        model = YOLO(model_opt)
        model.train(data="insulator.yaml", epochs=10, dnn=dnn, augment=aug, workers=16, batch=8, dropout=0.25453413174523265)
        metrics = model.val()
        return metrics.mean_results()[-1]

def objective(trial):
        model_opt = trial.suggest_categorical("Model", ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'])
        dnn = trial.suggest_categorical("DNN", [True, False])
        aug = trial.suggest_categorical("Augment Data", [True, False])
        error = attention(model_opt, dnn, aug)
        return error 

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
study.best_params

import joblib
joblib.dump(study, "study_exp_2.pkl")

end = time.time()
time_s = end - start
print(f'Time (s): {time_s} seconds')

############################################################################################################################################
# OBS: If you are running the experiments on your desktop, you can plot straightforward
############################################################################################################################################

#importances = optuna.visualization.plot_param_importances(study)
#importances.write_image("importances.pdf")
#contour = optuna.visualization.matplotlib.plot_contour(study, params=['Model', 'DNN', 'Augment Data'])
#contour.write_image("contour.pdf")
#parallel = optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=['Model', 'DNN', 'Augment Data'])
#parallel.write_image("parallel.pdf")

'''
OBS: IF you are interested here there are other hyperparameters that can be evaluated
 fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5) 
 hsv_h: 0.015  # image HSV-Hue augmentation (fraction) 
 hsv_s: 0.7  # image HSV-Saturation augmentation (fraction) 
 hsv_v: 0.4  # image HSV-Value augmentation (fraction) 
 degrees: 0.0  # image rotation (+/- deg) 
 translate: 0.1  # image translation (+/- fraction) 
 scale: 0.5  # image scale (+/- gain) 
 shear: 0.0  # image shear (+/- deg) 
 perspective: 0.0  # image perspective (+/- fraction), range 0-0.001 
 flipud: 0.0  # image flip up-down (probability) 
 fliplr: 0.5  # image flip left-right (probability) 
 mosaic: 1.0  # image mosaic (probability) 
 mixup: 0.0  # image mixup (probability) 
'''
