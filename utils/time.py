import os
import time


class TimeEstimator:
    def __init__(self, predict_mode='mean', sample_size=60):
        self.pred_mode = predict_mode
        self.t = None
        self.t_1 = None

    def stamp(self):
        if self.t is None:
            self.t = time.time_ns()


a = time.time_ns()
time.sleep(1)
diff = time.time_ns() - a
print(diff / 1e9)
