"""This file is to put any helping functions"""

import time
import os


def timeit(f):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()

        print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
        return result

    return timed


def create_dirs(experiment_dir):
    monitor_dir = os.path.join(experiment_dir, "monitor/")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints/")
    summary_dir = os.path.join(experiment_dir, "summaries/")
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_dir):
        os.makedirs(monitor_dir)
