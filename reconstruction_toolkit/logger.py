
# logger.py
# ---------------------------------------------
# Logging utility for writing time-stamped logs 
# to both terminal and file.
# 
# Functions:
# - log_step(step_name): Context manager for logging process durations.
# - write_log(message): Write plain log messages with timestamp.
# 
# Author: Kubra Kumrular (RKK)
# ---------------------------------------------

import datetime, os, time

LOG_FILE_PATH = None  # Will be set after user_inputs

os.makedirs("output", exist_ok=True)

def log_step(step_name):
    """
    Log a step: prints to terminal and writes to log file.
    Usage:
        with log_step("Loading projections"):
            # do stuff
    """
    class LogContext:
        def __enter__(self):
            self.start_time = time.time()
            self._log(f"{step_name} started...")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            elapsed = time.time() - self.start_time
            self._log(f"{step_name} completed in {elapsed:.2f} sec")

        def _log(self, message):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{timestamp}] {message}"
            print(line)
            with open(LOG_FILE_PATH, "a") as f:
                f.write(line + "\n")

    return LogContext()

def write_log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    if LOG_FILE_PATH:
        with open(LOG_FILE_PATH, "a") as f:
            f.write(line + "\n")


def set_log_path(path):
    global LOG_FILE_PATH
    LOG_FILE_PATH = path

