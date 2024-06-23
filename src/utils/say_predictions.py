import subprocess
import threading

# Define the function to say the predictions using the macOS 'say' command and a voice lock to prevent overlapping voices in the output audio stream (e.g., when multiple predictions are made simultaneously).
def say_predictions(voice_lock: threading.Lock, predict_diff: list[str]):
    voice_lock.acquire()
    for new_predict in predict_diff:
        subprocess.Popen(f"say {new_predict}", shell=True).wait()
    voice_lock.release()
