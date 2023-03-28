import optuna
import re
import numpy as np
from ultralytics import YOLO
import time

start = time.time()

def attention(opt, momen, lr):
        model = YOLO('yolov8x.pt')
        model.train(data="insulator.yaml", epochs=10, optimizer=opt, momentum=momen, lr0=lr, workers=16, batch=8, dropout=0.25453413174523265)
        metrics = model.val()
        return metrics.mean_results()[-1]

def objective(trial):
        opt = trial.suggest_categorical("Optimizer", ['SGD', 'Adam', 'AdamW'])
        momen = trial.suggest_float('Momentum', 0.85, 0.95)
        lr = trial.suggest_float('Learning Rate', 0.001, 0.1)
        error = attention(opt, momen, lr)
        return error

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
study.best_params

import joblib
joblib.dump(study, "study_exp_3.pkl")

end = time.time()
time_s = end - start
print(f'Time (s): {time_s} seconds')

############################################################################################################################################
# OBS: If you are running the experiments on your desktop, you can plot straightforward
############################################################################################################################################

#importances = optuna.visualization.plot_param_importances(study)
#importances.write_image("importances.pdf")
#contour = optuna.visualization.matplotlib.plot_contour(study, params=['Optimize', 'Momentum', 'Learning Rate'])
#contour.write_image("contour.pdf")
#parallel = optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=['Optimize', 'Momentum', 'Learning Rate'])
#parallel.write_image("parallel.pdf")