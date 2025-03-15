import os
import tkinter as tk
from tkinter import messagebox
import threading
import csv
import time
import subprocess

import explorepy
from explorepy.stream_processor import TOPICS

# For embedding matplotlib in Tkinter:
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


# ---------------------------
# Settings / Configuration
# ---------------------------
class Settings:
    def __init__(self):
        self.capture_dir = os.path.join("data", "captures", "free_recordings")
        os.makedirs(self.capture_dir, exist_ok=True)


# ---------------------------
# Explore Device Handler
# ---------------------------
class ExploreHandler:
    def __init__(self, device_name="Explore_849D", settings=None):
        self.device_name = device_name
        self.explorer = explorepy.Explore()
        self.settings = settings if settings else Settings()
        self.recording_data = []         # Holds incoming packets (timestamp, data_matrix)
        self.recording_active = False
        self.view_active = False
        # Assume each packet contains 16 samples per channel.
        self.packet_samples = 16
        self.sampling_rate = 250.0  # Hz
        self.sample_interval = 1.0 / self.sampling_rate
        # Buffer for real-time graphing (each channel: list of (timestamp, sample_value))
        self.view_buffer = {}

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
        """
        For real-time graphing: extract the first sample from each channel and append to a buffer.
        """
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
        The timestamp for each sample is computed by adding an offset.
        """
        filename = os.path.join(self.settings.capture_dir, f"recording_{int(time.time())}.csv")
        header = ["Timestamp"] + [f"Channel{i+1}" for i in range(8)]
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
            messagebox.showinfo("Recording Saved", f"Data saved to {filename}")
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
        The active_list is reversed to form the binary string, as LSB corresponds to channel 1.
        For example, if all channels are active: [True]*8 -> "11111111".
        If channel 2 is off: [True, False, True, True, True, True, True, True] ->
        reversed becomes "11111101".
        """
        try:
            mask_str = "".join("1" if active else "0" for active in reversed(active_list))
            # Call the Explore API using the proper parameter name.
            self.explorer.set_channels(channel_mask=mask_str)
            messagebox.showinfo("Electrode Settings", f"Channels updated (mask: {mask_str}).")
        except Exception as e:
            messagebox.showerror("Electrode Settings", f"Failed to set channels: {e}")


# ---------------------------
# Real-Time Graph Class
# ---------------------------
class RealTimeGraph:
    def __init__(self, parent, handler: ExploreHandler, buffer_length=200, update_interval=100):
        self.parent = parent
        self.handler = handler
        self.buffer_length = buffer_length
        self.update_interval = update_interval  # in milliseconds
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
# GUI Class
# ---------------------------
class MentalabGUI:
    def __init__(self, root, handler: ExploreHandler):
        self.root = root
        self.handler = handler
        self.real_time_graph = None
        self.build_gui()

    def build_gui(self):
        self.root.title("Mentalab Paradigm GUI")
        # Recording button.
        self.record_button = tk.Button(self.root, text="Start Recording", width=20,
                                       command=self.toggle_recording)
        self.record_button.pack(pady=10)
        # Real-Time View button.
        self.view_button = tk.Button(self.root, text="Start Real-Time View", width=20,
                                     command=self.toggle_view)
        self.view_button.pack(pady=10)
        # Electrode Settings button.
        self.electrode_button = tk.Button(self.root, text="Electrode Settings", width=20,
                                          command=self.open_electrode_settings)
        self.electrode_button.pack(pady=10)
        # Frame for embedding the real-time graph.
        self.graph_frame = tk.Frame(self.root)
        self.graph_frame.pack(pady=10)

    def toggle_recording(self):
        self.handler.start_recording()
        if self.handler.recording_active:
            self.record_button.config(text="Stop Recording")
        else:
            self.record_button.config(text="Start Recording")

    def toggle_view(self):
        if not self.handler.view_active:
            self.handler.subscribe_view()
            if not self.real_time_graph:
                self.real_time_graph = RealTimeGraph(self.graph_frame, self.handler,
                                                     buffer_length=200, update_interval=100)
            self.real_time_graph.start()
            self.view_button.config(text="Stop Real-Time View")
        else:
            self.handler.unsubscribe_view()
            if self.real_time_graph:
                self.real_time_graph.stop()
            self.view_button.config(text="Start Real-Time View")

    def open_electrode_settings(self):
        window = tk.Toplevel(self.root)
        window.title("Electrode Settings")
        instructions = tk.Label(window, text="Select the active electrodes:")
        instructions.pack(pady=5)
        self.channel_vars = []
        for i in range(8):
            var = tk.BooleanVar(value=True)
            chk = tk.Checkbutton(window, text=f"Channel {i+1}", variable=var)
            chk.pack(anchor='w')
            self.channel_vars.append(var)
        def apply_settings():
            active_list = [var.get() for var in self.channel_vars]
            self.handler.set_channels(active_list)
            window.destroy()
        apply_btn = tk.Button(window, text="Apply", command=apply_settings)
        apply_btn.pack(pady=10)


# ---------------------------
# Main Application
# ---------------------------
def main():
    root = tk.Tk()
    settings = Settings()
    handler = ExploreHandler(device_name="Explore_849D", settings=settings)
    threading.Thread(target=handler.connect, daemon=True).start()
    handler.start_acquisition()
    gui = MentalabGUI(root, handler)
    root.mainloop()


if __name__ == "__main__":
    main()
