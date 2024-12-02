"""Live captions from microphone using Moonshine and SileroVAD ONNX models with GUI."""

import argparse
import os
import time
import threading
from queue import Queue, Empty
import argostranslate.package
import argostranslate.translate

####### TRANSLATION CONFIG ##########
from_code = "en"
to_code = "it"

# Download and install Argos Translate package
print("Downloading Argos Translate...")
argostranslate.package.update_package_index()
available_packages = argostranslate.package.get_available_packages()
package_to_install = next(
    filter(
        lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
    )
)
argostranslate.package.install_from_path(package_to_install.download())
#####################################

import numpy as np
from silero_vad import VADIterator, load_silero_vad
from sounddevice import InputStream

from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

import tkinter as tk

SAMPLING_RATE = 16000

CHUNK_SIZE = 512  # Silero VAD requirement with sampling rate 16000.
LOOKBACK_CHUNKS = 5
MAX_LINE_LENGTH = 80

# These affect live caption updating - adjust for your platform speed and model.
MAX_SPEECH_SECS = 15
MIN_REFRESH_SECS = 0.2


class Transcriber(object):
    def __init__(self, model_name, rate=16000):
        if rate != 16000:
            raise ValueError("Moonshine supports sampling rate 16000 Hz.")
        self.model = MoonshineOnnxModel(model_name=model_name)
        self.rate = rate
        self.tokenizer = load_tokenizer()

        self.inference_secs = 0
        self.number_inferences = 0
        self.speech_secs = 0
        self.__call__(np.zeros(int(rate), dtype=np.float32))  # Warmup.

    def __call__(self, speech):
        """Returns string containing Moonshine transcription of speech."""
        self.number_inferences += 1
        self.speech_secs += len(speech) / self.rate
        start_time = time.time()

        tokens = self.model.generate(speech[np.newaxis, :].astype(np.float32))
        text = self.tokenizer.decode_batch(tokens)[0]

        # translation
        text = argostranslate.translate.translate(text, from_code, to_code)

        self.inference_secs += time.time() - start_time

        return text


def create_input_callback(q):
    """Callback method for sounddevice InputStream."""

    def input_callback(data, frames, time, status):
        if status:
            print(status)
        q.put((data.copy().flatten(), status))

    return input_callback


def end_recording(speech, caption_cache, do_print=True):
    """Transcribes, prints and caches the caption then clears speech buffer."""
    text = transcribe(speech)
    if do_print:
        print_captions(text, caption_cache)
    caption_cache.append(text)
    speech *= 0.0


def print_captions(text, caption_cache):
    """Updates the GUI with the new captions."""
    # Combine cached captions and current text
    full_text = " ".join(caption_cache + [text])
    gui_queue.put(full_text)


def soft_reset(vad_iterator):
    """Soft resets Silero VADIterator without affecting VAD model state."""
    vad_iterator.triggered = False
    vad_iterator.temp_end = 0
    vad_iterator.current_sample = 0


class CaptionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Captions")

        # Configure font size here
        font_settings = ("Helvetica", 64)  # Font name and size (increase as needed)

        # Add a scrollbar for the text widget
        self.text_frame = tk.Frame(self.root)
        self.text_frame.pack(fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.text_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_widget = tk.Text(
            self.text_frame,
            wrap="word",
            height=20,
            width=80,
            font=font_settings,
            yscrollcommand=self.scrollbar.set,
        )
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.text_widget.yview)

        self.update_interval = 100  # milliseconds
        self.last_text = ""

    def update_gui(self):
        try:
            while True:
                text = gui_queue.get_nowait()
                if text != self.last_text:
                    self.text_widget.delete("1.0", tk.END)
                    self.text_widget.insert(tk.END, text)
                    self.text_widget.see(tk.END)  # Automatically scroll to the bottom
                    self.last_text = text
        except Empty:
            pass
        self.root.after(self.update_interval, self.update_gui)


def audio_processing():
    global transcribe
    parser = argparse.ArgumentParser(
        prog="live_captions",
        description="Live captioning demo of Moonshine models",
    )
    parser.add_argument(
        "--model_name",
        help="Model to run the demo with",
        default="moonshine/base",
        choices=["moonshine/base", "moonshine/tiny"],
    )
    args = parser.parse_args()
    model_name = args.model_name
    print(f"Loading Moonshine model '{model_name}' (using ONNX runtime) ...")
    transcribe = Transcriber(model_name=model_name, rate=SAMPLING_RATE)

    vad_model = load_silero_vad(onnx=True)
    vad_iterator = VADIterator(
        model=vad_model,
        sampling_rate=SAMPLING_RATE,
        threshold=0.5,
        min_silence_duration_ms=300,
    )

    q = Queue()
    stream = InputStream(
        samplerate=SAMPLING_RATE,
        channels=1,
        blocksize=CHUNK_SIZE,
        dtype=np.float32,
        callback=create_input_callback(q),
    )
    stream.start()

    caption_cache = []
    lookback_size = LOOKBACK_CHUNKS * CHUNK_SIZE
    speech = np.empty(0, dtype=np.float32)

    recording = False

    print("Press Ctrl+C to quit live captions.\n")

    with stream:
        gui_queue.put("Ready...")
        try:
            while True:
                chunk, status = q.get()
                if status:
                    print(status)

                speech = np.concatenate((speech, chunk))
                if not recording:
                    speech = speech[-lookback_size:]

                speech_dict = vad_iterator(chunk)
                if speech_dict:
                    if "start" in speech_dict and not recording:
                        recording = True
                        start_time = time.time()

                    if "end" in speech_dict and recording:
                        recording = False
                        end_recording(speech, caption_cache)
                elif recording:
                    # Possible speech truncation can cause hallucination.

                    if (len(speech) / SAMPLING_RATE) > MAX_SPEECH_SECS:
                        recording = False
                        end_recording(speech, caption_cache)
                        soft_reset(vad_iterator)

                    if (time.time() - start_time) > MIN_REFRESH_SECS:
                        text = transcribe(speech)
                        print_captions(text, caption_cache)
                        start_time = time.time()
        except KeyboardInterrupt:
            stream.close()

            if recording:
                while not q.empty():
                    chunk, _ = q.get()
                    speech = np.concatenate((speech, chunk))
                end_recording(speech, caption_cache, do_print=False)

            print(
                f"""

                 model_name :  {model_name}
           MIN_REFRESH_SECS :  {MIN_REFRESH_SECS}s

          number inferences :  {transcribe.number_inferences}
        mean inference time :  {(transcribe.inference_secs / transcribe.number_inferences):.2f}s
      model realtime factor :  {(transcribe.speech_secs / transcribe.inference_secs):0.2f}x
    """
            )
            if caption_cache:
                print(f"Cached captions.\n{' '.join(caption_cache)}")


if __name__ == "__main__":
    gui_queue = Queue()
    # Start the audio processing thread
    audio_thread = threading.Thread(target=audio_processing)
    audio_thread.start()

    # Create GUI
    root = tk.Tk()
    gui = CaptionGUI(root)
    root.after(100, gui.update_gui)  # Start the GUI update loop
    root.mainloop()
