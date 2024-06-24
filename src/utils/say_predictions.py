import os
import subprocess
import threading
from gtts import gTTS
from playsound import playsound


# Define the function to say the predictions using the macOS 'say' command and a voice lock to prevent overlapping voices in the output audio stream (e.g., when multiple predictions are made simultaneously).
def say_predictions(voice_lock: threading.Lock, predict_diff: list[str]):
    voice_lock.acquire()
    for new_predict in predict_diff:
            #subprocess.Popen(f"say {new_predict}", shell=True).wait() # Uncomment this line to use the 'say' command on macOS
            tts = gTTS(text=new_predict, lang='en')
            filename = "prediction.mp3"
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
    voice_lock.release()
