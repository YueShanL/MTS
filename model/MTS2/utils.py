import time

import torch


class TimeProfiler:
    def __init__(self):
        self.times = {}
        self.start_times = {}

    def start(self, key):
        torch.cuda.synchronize()
        self.start_times[key] = time.time()

    def stop(self, key):
        torch.cuda.synchronize()
        t = time.time() - self.start_times[key]
        self.times[key] = self.times.get(key, 0) + t

    def report(self, step=None):
        msg = f"[PROFILE step={step}] "
        for k, v in self.times.items():
            msg += f"{k}: {v:.3f}s | "
        print(msg)
        self.times = {}