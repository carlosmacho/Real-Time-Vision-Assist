import subprocess
import threading


def say_predictions(voice_lock: threading.Lock, predict_diff: list[str]):
    voice_lock.acquire()
    for new_predict in predict_diff:
        subprocess.Popen(f"say {new_predict}", shell=True).wait()
    voice_lock.release()
