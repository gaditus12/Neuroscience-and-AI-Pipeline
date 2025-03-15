import copy
import os
import tkinter as tk
from tkinter import messagebox
import threading
import csv
import time
import random
import asyncio
import tempfile
from playsound import playsound
import explorepy
from explorepy.stream_processor import TOPICS
from matplotlib import pyplot as plt
import beepy

# ---------------------------
# Text-to-Speech Functionality
# ---------------------------
chosen_voice="en-US-GuyNeural"
def speak_text(text, voice=chosen_voice, rate="-10%"):
    """
    Uses edge-tts to synthesize speech from text, with a slower speaking rate,
    saves the output to a temporary MP3 file, and plays it using pydub and simpleaudio.
    If this fails, it falls back to a simple beep.

    Requires: edge-tts, pydub, simpleaudio, and ffmpeg installed.
    """
    try:
        import edge_tts
        from pydub import AudioSegment
        import simpleaudio as sa
    except ImportError as e:
        print("Please install edge-tts, pydub, and simpleaudio (and ensure ffmpeg is installed).")
        return

    async def run_tts():
        # Pass the slower rate parameter.
        communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
        temp_mp3 = os.path.join(tempfile.gettempdir(), "temp_tts.mp3")
        await communicate.save(temp_mp3)
        return temp_mp3

    try:
        temp_mp3 = asyncio.run(run_tts())
        audio = AudioSegment.from_file(temp_mp3, format="mp3")
        playback = sa.play_buffer(
            audio.raw_data,
            num_channels=audio.channels,
            bytes_per_sample=audio.sample_width,
            sample_rate=audio.frame_rate
        )
        playback.wait_done()
    except Exception as e:
        print("TTS error:", e)
        # TODO beep sound is not working correctly, maybe we don't even want beep sound
        # TODO: arrange the audio, perhaps we just do, imagination instruction: red car little, red car little etc. sort of command
        # Fallback: use winsound to produce a beep.
        try:
            import winsound
            winsound.Beep(440, 1000)  # 440Hz for 1 second
        except Exception as e2:
            print("Fallback beep error:", e2)


# ---------------------------
# Settings / Configuration
# ---------------------------
class Settings:
    def __init__(self):
        self.capture_dir = os.path.join("data", "captures")
        os.makedirs(self.capture_dir, exist_ok=True)


# ---------------------------
# Explore Device Handler
# ---------------------------
class ExploreHandler:
    def __init__(self, device_name="Explore_849D", settings=None):
        self.device_name = device_name
        self.explorer = explorepy.Explore()
        self.settings = settings if settings else Settings()
        self.recording_data = []  # Holds incoming packets (timestamp, data_matrix)
        self.recording_active = False
        self.view_active = False
        # Assume each packet contains 16 samples per channel.
        self.packet_samples = 16
        self.sampling_rate = 250.0  # Hz
        self.sample_interval = 1.0 / self.sampling_rate
        self.view_buffer = {}  # For real-time graphing
        # Attributes for saving recordings:
        self.current_segment_folder = None
        self.current_segment_label = "session"

    def connect(self):
        try:
            self.explorer.connect(device_name=self.device_name)
        except Exception as e:
            messagebox.showerror("Connection Error", f"Failed to connect: {e}")

    def start_acquisition(self):
        def acquisition_thread():
            time.sleep(2)  # Allow connection to stabilize.
            try:
                self.explorer.acquire()  # Start streaming.
            except Exception as e:
                print(f"Acquisition error: {e}")

        threading.Thread(target=acquisition_thread, daemon=True).start()

    # --- Callbacks for streaming ---
    def record_callback(self, packet):
        try:
            t_vector, exg_data = packet.get_data()
            if isinstance(t_vector, (list, tuple)):
                timestamp = t_vector[0] if t_vector else time.time()
            else:
                timestamp = t_vector
            if hasattr(exg_data, "tolist"):
                data_list = exg_data.tolist()
            else:
                data_list = exg_data
            self.recording_data.append((timestamp, data_list))
        except Exception as e:
            print("Error in record_callback:", e)

    def view_callback(self, packet):
        try:
            t_vector, exg_data = packet.get_data()
            if isinstance(t_vector, (list, tuple)):
                timestamp = t_vector[0] if t_vector else time.time()
            else:
                timestamp = t_vector
            sample_values = []
            for ch in range(8):
                try:
                    sample_values.append(exg_data[ch][0])
                except Exception:
                    sample_values.append(None)
            return f"Time: {timestamp:.3f}, Data: {sample_values}"
        except Exception as e:
            print("Error in view_callback:", e)
            return "Error in view_callback"

    def view_graph_callback(self, packet):
        try:
            t_vector, exg_data = packet.get_data()
            if isinstance(t_vector, (list, tuple)):
                timestamp = t_vector[0] if t_vector else time.time()
            else:
                timestamp = t_vector
            for ch in range(8):
                try:
                    sample_value = exg_data[ch][0]
                except Exception:
                    sample_value = None
                if sample_value is not None:
                    self.view_buffer.setdefault(ch, []).append((timestamp, sample_value))
                    if len(self.view_buffer[ch]) > 200:
                        self.view_buffer[ch] = self.view_buffer[ch][-200:]
        except Exception as e:
            print("Error in view_graph_callback:", e)

    # --- Recording functions ---
    def start_recording(self):
        if not self.recording_active:
            self.recording_data = []  # Clear previous data
            self.explorer.stream_processor.subscribe(
                callback=self.record_callback,
                topic=TOPICS.raw_ExG
            )
            self.recording_active = True
        else:
            self.explorer.stream_processor.unsubscribe(
                callback=self.record_callback,
                topic=TOPICS.raw_ExG
            )
            self.recording_active = False
            self.save_recording_to_csv()

    def save_recording_to_csv(self):
        """
        Save recorded data by flattening each 8x16 packet into 16 rows.
        Each row: Timestamp, Channel1, Channel2, ..., Channel8.
        If a current_segment_folder is defined, use that as the target folder.
        """
        target_folder = self.current_segment_folder if self.current_segment_folder else self.settings.capture_dir
        os.makedirs(target_folder, exist_ok=True)
        filename = os.path.join(target_folder, f"{self.current_segment_label}_{int(time.time())}.csv")
        header = ["Timestamp"] + [f"Channel{i + 1}" for i in range(8)]
        try:
            with open(filename, mode="w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                for packet_timestamp, data in self.recording_data:
                    if not (isinstance(data, list) and len(data) == 8 and
                            all(isinstance(ch, list) and len(ch) == self.packet_samples for ch in data)):
                        print("Warning: Unexpected data shape, skipping packet.")
                        continue
                    for i in range(self.packet_samples):
                        row_timestamp = packet_timestamp + i * self.sample_interval
                        row = [row_timestamp] + [data[ch][i] for ch in range(8)]
                        writer.writerow(row)
            # For smooth experimental flow, message boxes are commented out.
            # messagebox.showinfo("Recording Saved", f"Data saved to {filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save CSV: {e}")

    # --- Real-Time View functions (Graphing) ---
    def subscribe_view(self):
        self.explorer.stream_processor.subscribe(
            callback=self.view_graph_callback,
            topic=TOPICS.raw_ExG
        )
        self.view_active = True

    def unsubscribe_view(self):
        self.explorer.stream_processor.unsubscribe(
            callback=self.view_graph_callback,
            topic=TOPICS.raw_ExG
        )
        self.view_active = False

    # --- Electrode Settings ---
    def set_channels(self, active_list):
        """
        Set active electrodes based on a list of booleans for channels 1 to 8.
        The active_list is reversed to form the binary string (LSB is channel 1).
        Example: all True -> "11111111"; if channel 2 is off: [True, False, True, ...] -> "11111101".
        """
        try:
            mask_str = "".join("1" if active else "0" for active in reversed(active_list))
            self.explorer.set_channels(channel_mask=mask_str)
            messagebox.showinfo("Electrode Settings", f"Channels updated (mask: {mask_str}).")
        except Exception as e:
            messagebox.showerror("Electrode Settings", f"Failed to set channels: {e}")


# ---------------------------
# Real-Time Graph Class
# ---------------------------
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class RealTimeGraph:
    def __init__(self, parent, handler: ExploreHandler, buffer_length=200, update_interval=100):
        self.parent = parent
        self.handler = handler
        self.buffer_length = buffer_length
        self.update_interval = update_interval  # milliseconds
        self.fig, self.axes = plt.subplots(8, 1, figsize=(8, 10), sharex=True)
        self.lines = []
        for ax in self.axes:
            line, = ax.plot([], [], lw=1)
            self.lines.append(line)
            ax.set_ylabel("µV")
        self.axes[-1].set_xlabel("Time (s)")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.get_tk_widget().pack()
        self.running = False
    def start(self):
        self.running = True
        self.update_plot()

    def stop(self):
        self.running = False

    def update_plot(self):
        for ch in range(8):
            data = self.handler.view_buffer.get(ch, [])
            if data:
                t_vals, y_vals = zip(*data)
                self.lines[ch].set_data(t_vals, y_vals)
                self.axes[ch].relim()
                self.axes[ch].autoscale_view()
        self.canvas.draw()
        if self.running:
            self.parent.after(self.update_interval, self.update_plot)


# ---------------------------
# Experiment Session Class
# ---------------------------
class ExperimentSession:
    """
    This class defines the experimental structure.
    The session folder is created under data/captures.
    The introduction (baseline-pre) is recorded once and serves as the initial baseline.
    Then the experiment loops indefinitely.
    Each imagery task is chosen randomly from a lookup dictionary.
    For each imagery task, a folder is created (if it does not exist) under the session folder.
    Within that imagery folder, three recordings are saved:
      - instruction: the task instruction recording.
      - imagery_task: the actual imagery recording.
      - baseline_post: the post-task baseline recording.
    Files are named with the segment label and a timestamp.
    """

    def __init__(self, handler: ExploreHandler):
        self.handler = handler
        self.session_folder = os.path.join(self.handler.settings.capture_dir, f"session_{int(time.time())}")
        os.makedirs(self.session_folder, exist_ok=True)
        self.imagery_lookup = {
            #"a red car, little": "o_rc_l",
            "the Eiffel Tower, small": "m_et_s",
            # "a green circular start sign, small": "p_sf_s",
            # "a green circular start sign , big": "p_sf_b",
            # "an angry red stop sign, small": "p_af_s",
            # "an angry red stop sign, big": "p_af_b",
            # "the pyramids, small": "m_gp_s",
            # "your cat in bed, big": "a_c_b", #animal
            # "your motorcycle in garage, big": "b_m_b", #belonging
            # "square, small": "o_sq_s", #object square small
            "number three, small": "n_3_s",
            # "spiky glass, big": "i_sg_b", #imaginary object
            # "doing pull-ups": "a_pu_b", #activity, big refers to complete imagery (no size limitation)

            #"Imagine a moving waterfall": "n_mw_l",
            #"Imagine a dark forest": "s_df_m",
            #"Imagine a bright, abstract pattern": "a_ab_s"
        }
        self.imagery_lookup_exhaust=copy.deepcopy(self.imagery_lookup)

        self.baseline_text="Task completed, please relax for a few seconds."
        #TODO  make sure no occurance of more than twice one after another
        self.intro_duration = 2  # Introduction baseline (baseline-pre)
        self.instruction_duration = 5  # Imagery task instruction
        self.imagery_duration = 20  # Imagery task recording
        self.baseline_post_duration = 10  # Baseline post recording
        self.attention_beep='sound/attention.wav'
        self.relax_beep='sound/relax.wav'
        # some settings
        self.sess_count=3
        self.short_intro= True

    def play_introduction(self):
        if not self.short_intro:
            introduction_text_first = "Please listen to the instructions. This serves as the baseline pre-recording and instruction segment, during the experiment, you will be given a set of imagery tasks. You should start imagining after you hear a task followed by the attention beep. Listen to the attention beep:"
            speak_text(introduction_text_first)
            playsound(self.attention_beep)
            introduction_text_after_attention_beep ='This is your imagination attention beep. After hearing this beep you are expected to start visual imagery, you may repeat the label in your head. After your imagery task you will hear a relaxation sound, which indicates the end of the specific imagination task. Listen to the relaxation sound:'
            speak_text(introduction_text_after_attention_beep)
            playsound(self.relax_beep)
            introduction_text_after_relax_beep = "In a few seconds, you will be given your first imagery task."
            speak_text(introduction_text_after_relax_beep)
        else:
            introduction_text='The experiment will start in a few seconds, please relax and get ready.'
            speak_text(introduction_text)
    def finish_experiment(self):
        exit_text='The experiment is complete, you may open your eyes.'
        speak_text(exit_text)
        exit(0)
    def run(self):
        print("Starting experimental session...")
        # --- Introduction Segment (baseline-pre) ---
        intro_folder = os.path.join(self.session_folder, "introduction")
        os.makedirs(intro_folder, exist_ok=True)
        print("\n--- Introduction (Baseline Pre) ---")
        print(f"Recording Introduction")
        self.handler.current_segment_folder = intro_folder
        self.handler.current_segment_label = "introduction"
        self.handler.start_recording()
        self.play_introduction()
        self.handler.start_recording()
        print("Introduction segment completed and saved.")

        # --- Main Loop: Imagery Task Segments ---
        try:
            while True:
                if len(self.imagery_lookup_exhaust)!=0:
                    chosen_task = random.choice(list(self.imagery_lookup_exhaust.keys()))
                    imagery_code = self.imagery_lookup_exhaust[chosen_task]
                    self.imagery_lookup_exhaust.pop(chosen_task)
                else:
                    if (self.sess_count==0):
                        self.finish_experiment()
                    else:
                        self.sess_count=self.sess_count-1
                        self.imagery_lookup_exhaust=copy.deepcopy(self.imagery_lookup)
                        chosen_task = random.choice(list(self.imagery_lookup_exhaust.keys()))
                        imagery_code = self.imagery_lookup_exhaust[chosen_task]
                        self.imagery_lookup_exhaust.pop(chosen_task)
                imagery_folder = os.path.join(self.session_folder, imagery_code)
                os.makedirs(imagery_folder, exist_ok=True)

                # --- Instruction Segment for this Imagery Task ---
                print(f"\n--- {imagery_code} Instruction ---")
                instruction_text = f"Your imagery task is: {chosen_task}, again, {chosen_task}" # removed 'start imagining after the beep'
                self.handler.current_segment_folder = imagery_folder
                self.handler.current_segment_label = f"{imagery_code}_instruction"
                print(f"Recording Instruction for {self.instruction_duration} seconds...")
                self.handler.start_recording()
                print(instruction_text)
                speak_text(instruction_text)
                playsound(self.attention_beep)
                self.handler.start_recording()
                print("Instruction segment completed and saved.")

                # --- Imagery Task Segment ---
                print(f"\n--- {imagery_code} Imagery Task ---")
                # input("Press Enter to start the imagery task recording...")
                self.handler.current_segment_folder = imagery_folder
                self.handler.current_segment_label = f"{imagery_code}_imagery_task"
                self.handler.start_recording()
                print(f"Recording Imagery Task for {self.imagery_duration} seconds...")
                time.sleep(self.imagery_duration)
                self.handler.start_recording()
                print("Imagery Task segment completed and saved.")



                # --- Baseline Post Segment (belongs to the imagery task) ---
                playsound(self.relax_beep)
                speak_text(self.baseline_text)
                print(f"\n--- {imagery_code} Baseline Post ---")
                # input("Press Enter to start the baseline post recording...")
                self.handler.current_segment_folder = imagery_folder
                self.handler.current_segment_label = f"{imagery_code}_baseline_post"
                self.handler.start_recording()
                print(f"Recording Baseline Post for {self.baseline_post_duration} seconds...")
                time.sleep(self.baseline_post_duration)
                self.handler.start_recording()
                print("Baseline Post segment completed and saved.")

                # Optionally: Continue indefinitely.
                # cont = input("Press Enter to continue to the next task or type 'exit' to finish: ")
                # if cont.lower().strip() == "exit":
                #     break
        except KeyboardInterrupt:
            print("Experiment interrupted by user.")
        print("Experimental session completed.")


# ---------------------------
# Minimal GUI for Session Control
# ---------------------------
def session_controller_gui(handler: ExploreHandler):
    root = tk.Tk()
    root.title("Experiment Controller")
    label = tk.Label(root, text="Connecting to device and starting acquisition...\nCheck console for updates.")
    label.pack(pady=20)

    def start_session():
        root.destroy()  # Close the controller window.
        session = ExperimentSession(handler)
        session.run()

    start_button = tk.Button(root, text="Start Experimental Session", command=start_session)
    start_button.pack(pady=10)
    root.mainloop()


# ---------------------------
# Main Application
# ---------------------------
def main():
    settings = Settings()
    handler = ExploreHandler(device_name="Explore_849D", settings=settings)
    threading.Thread(target=handler.connect, daemon=True).start()
    handler.start_acquisition()
    session_controller_gui(handler)


if __name__ == "__main__":
    main()
