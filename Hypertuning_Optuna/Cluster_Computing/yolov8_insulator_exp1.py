import optuna
import re
import numpy as np
from ultralytics import YOLO
import time

start = time.time()

def attention(work, bat, drop):
        model = YOLO('yolov8n.pt')
        model.train(data="insulator.yaml", epochs=10, workers=work, batch=bat, dropout=drop)
        metrics = model.val()
        return metrics.mean_results()[-1]

def objective(trial):
        work = trial.suggest_categorical('Workers', [2, 4, 8, 16, 32])
        bat = trial.suggest_categorical('Batch Size', [2, 4, 8, 16, 32])
        drop = trial.suggest_float("Dropout", 0, 0.3)
        error = attention(work, bat, drop)
        return error

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
study.best_params

import joblib
joblib.dump(study, "study_exp_1.pkl")

end = time.time()
time_s = end - start
print(f'Time (s): {time_s} seconds')

############################################################################################################################################
# OBS: If you are running the experiments on your desktop, you can plot straightforward
############################################################################################################################################

#importances = optuna.visualization.plot_param_importances(study)
#importances.write_image("importances.pdf")
#contour = optuna.visualization.matplotlib.plot_contour(study, params=['Workers', 'Batch Size', 'Dropout'])
#contour.write_image("contour.pdf")
#parallel = optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=['Workers', 'Batch Size', 'Dropout'])
#parallel.write_image("parallel.pdf")