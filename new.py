# -*- coding: utf-8 -*-
"""
Final Version v7.1: Real-time Graphing Dashboard (Ticker Fix)
Author: Gemini
Description: Fixes the 'tkinter.ticker' AttributeError by correctly importing and
             using the ticker module from matplotlib.
"""
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, PanedWindow, Frame
from ultralytics import YOLO
import threading
import queue
from PIL import Image, ImageTk
import csv
from datetime import datetime
import time
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import matplotlib.ticker as mticker # <-- FIX 1: IMPORT DITAMBAHKAN

# ... (Panel Konfigurasi, Fungsi ROI, dan Fungsi CSV tetap sama persis) ...
# ==================================================================================
# === PANEL KONFIGURASI UTAMA ===
# ==================================================================================
CONFIG = {
    "DISPLAY_WIDTH": 960, "DISPLAY_HEIGHT": 540,
    "YOLO_IMG_SIZE": 640, "CONFIDENCE_THRESHOLD": 0.4, "FRAME_SKIP": 1,
    "STOP_THRESHOLD_PIXELS": 3.0, "STOP_CONFIRMATION_FRAMES": 15,
    "MOVE_CONFIRMATION_FRAMES": 8, "MIN_STOP_DURATION_SECONDS": 2,
    
    "MAX_ROAD_CAPACITY": 50,
    "GRAPH_UPDATE_INTERVAL_MS": 5000
}
# ==================================================================================

roi_points = []
def on_mouse_roi_selection(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN: roi_points.append((x, y))
def select_roi_on_frame(frame):
    global roi_points
    roi_points = []
    clone = frame.copy()
    window_name = "LANGKAH 1: PILIH AREA DETEKSI KENDARAAN (ROI)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_roi_selection)
    print("\n" + "="*50 + "\nPanduan Memilih Area (ROI):\n- Klik kiri untuk menandai titik sudut.\n- Tekan 'ENTER' untuk selesai.\n- Tekan 'c' untuk mengulang.\n" + "="*50 + "\n")
    while True:
        temp_frame = clone.copy()
        if len(roi_points) > 0: cv2.polylines(temp_frame, [np.array(roi_points)], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.putText(temp_frame, "Klik: Tambah Titik | ENTER: Selesai | C: Reset", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            if len(roi_points) > 2: break
            else: print("Error: Area membutuhkan minimal 3 titik.")
        elif key == ord('c'): roi_points = []
    cv2.destroyWindow(window_name)
    return np.array(roi_points) if len(roi_points) > 2 else None
CSV_FILENAME = "log_volume_kendaraan.csv"
def setup_csv(class_names):
    try:
        with open(CSV_FILENAME, 'x', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            headers = ["Timestamp Selesai Berhenti", "Durasi Berhenti (detik)"] + [name.title() for name in class_names]
            writer.writerow(headers)
    except FileExistsError: pass
def log_to_csv(timestamp, duration, counts, class_names_ordered):
    with open(CSV_FILENAME, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        row = [timestamp, f"{duration:.2f}"] + [counts.get(name, 0) for name in class_names_ordered]
        writer.writerow(row)

class VehicleDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Deteksi Kendaraan v7.1 (Dashboard Real-time)")
        self.root.geometry("1600x900")
        self.model_path, self.video_path = "", ""
        self.processing_thread, self.reader_thread = None, None
        self.stop_event = threading.Event()
        self.raw_frame_queue = queue.Queue(maxsize=30)
        self.processed_frame_queue = queue.Queue(maxsize=30)
        self.setup_gui()
        self.setup_graph()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        control_frame = Frame(self.root)
        control_frame.pack(pady=10, padx=10, fill=tk.X)
        tk.Label(control_frame, text="Detektor Kendaraan YOLO", font=("Helvetica", 18, "bold")).pack()
        file_frame = Frame(control_frame); file_frame.pack(pady=10)
        self.btn_model = tk.Button(file_frame, text="Pilih Model (.pt)", command=self.load_model, width=20); self.btn_model.grid(row=0, column=0, padx=5, pady=5)
        self.lbl_model_path = tk.Label(file_frame, text="Model belum dipilih", fg="red", anchor="w"); self.lbl_model_path.grid(row=0, column=1, padx=5, sticky="w")
        self.btn_video = tk.Button(file_frame, text="Pilih Video", command=self.load_video, width=20); self.btn_video.grid(row=1, column=0, padx=5, pady=5)
        self.lbl_video_path = tk.Label(file_frame, text="Video belum dipilih", fg="red", anchor="w"); self.lbl_video_path.grid(row=1, column=1, padx=5, sticky="w")
        action_frame = Frame(control_frame); action_frame.pack(pady=10)
        self.btn_start = tk.Button(action_frame, text="Mulai Deteksi", command=self.start_detection, font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", width=15); self.btn_start.pack(side=tk.LEFT, padx=10)
        self.btn_stop = tk.Button(action_frame, text="Hentikan", command=self.stop_detection, font=("Helvetica", 12), bg="#F44336", fg="white", state=tk.DISABLED, width=15); self.btn_stop.pack(side=tk.LEFT, padx=10)
        main_pane = PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED); main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        video_frame = Frame(main_pane, bg="black")
        self.video_canvas = tk.Label(video_frame, bg="black"); self.video_canvas.pack(fill=tk.BOTH, expand=True)
        main_pane.add(video_frame, stretch="always")
        self.graph_frame = Frame(main_pane, bg="white"); main_pane.add(self.graph_frame, stretch="always")
        self.lbl_status = tk.Label(self.root, text="Status: Siap", bd=1, relief=tk.SUNKEN, anchor="w"); self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_graph(self):
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Grafik Kepadatan Lalu Lintas"); self.ax.set_xlabel("Waktu"); self.ax.set_ylabel("Kepadatan (%)"); self.ax.grid(True)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame); self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
    def update_graph(self):
        if self.stop_event.is_set(): return
        try:
            df = pd.read_csv(CSV_FILENAME)
            if df.empty: raise FileNotFoundError
            df = df.tail(50)
            df['Timestamp'] = pd.to_datetime(df['Timestamp Selesai Berhenti'])
            vehicle_columns = df.columns.drop(['Timestamp Selesai Berhenti', 'Durasi Berhenti (detik)', 'Timestamp'])
            df['Total Kendaraan'] = df[vehicle_columns].sum(axis=1)
            df['Kepadatan'] = (df['Total Kendaraan'] / CONFIG["MAX_ROAD_CAPACITY"]) * 100
            self.ax.clear()
            colors = ['red' if d > 15 else 'orange' if d >= 10 else 'green' for d in df['Kepadatan']]
            self.ax.scatter(df['Timestamp'], df['Kepadatan'], c=colors)
            self.ax.set_title("Grafik Kepadatan Lalu Lintas"); self.ax.set_xlabel("Waktu"); self.ax.set_ylabel("Kepadatan (%)")
            self.ax.grid(True, linestyle='--', alpha=0.6)
            
            # FIX 2: Menggunakan mticker dari matplotlib, bukan tk.ticker
            self.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
            
            self.fig.autofmt_xdate(); self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            self.canvas.draw()
        except FileNotFoundError:
            self.ax.clear()
            self.ax.text(0.5, 0.5, 'Menunggu data dari CSV...', ha='center', va='center', transform=self.ax.transAxes)
            self.ax.set_title("Grafik Kepadatan Lalu Lintas"); self.ax.set_xlabel("Waktu"); self.ax.set_ylabel("Kepadatan (%)")
            self.canvas.draw()
        except Exception as e:
            print(f"Error saat update grafik: {e}")
        self.root.after(CONFIG["GRAPH_UPDATE_INTERVAL_MS"], self.update_graph)

    def start_detection(self):
        if not self.model_path or not self.video_path: messagebox.showerror("Error", "Silakan pilih file model dan video terlebih dahulu!"); return
        if (self.processing_thread and self.processing_thread.is_alive()): messagebox.showinfo("Info", "Proses deteksi sudah berjalan."); return
        self.btn_start.config(state=tk.DISABLED); self.btn_stop.config(state=tk.NORMAL)
        self.stop_event.clear()
        self.reader_thread = threading.Thread(target=self.video_reader_worker); self.reader_thread.daemon = True; self.reader_thread.start()
        self.processing_thread = threading.Thread(target=self.run_detection_worker); self.processing_thread.daemon = True; self.processing_thread.start()
        self.update_video_canvas()
        self.update_graph()
    
    # ... (Sisa kode seperti load_model, stop_detection, dll. sama persis) ...
    def load_model(self):
        path = filedialog.askopenfilename(title="Pilih file model YOLO", filetypes=[("YOLO Model", "*.pt")])
        if path: self.model_path = path; self.lbl_model_path.config(text=f"Model: ...{path[-30:]}", fg="green")
    def load_video(self):
        path = filedialog.askopenfilename(title="Pilih file video", filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
        if path: self.video_path = path; self.lbl_video_path.config(text=f"Video: ...{path[-30:]}", fg="green")
    def stop_detection(self):
        self.lbl_status.config(text="Status: Menghentikan..."); self.stop_event.set()
        time.sleep(1); self.btn_start.config(state=tk.NORMAL); self.btn_stop.config(state=tk.DISABLED)
    def on_closing(self):
        self.stop_event.set(); self.root.destroy()
    def update_video_canvas(self):
        frame = None
        try:
            while not self.processed_frame_queue.empty(): frame = self.processed_frame_queue.get_nowait()
        except queue.Empty: pass
        if frame is not None:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); img_tk = ImageTk.PhotoImage(image=img)
            self.video_canvas.img_tk = img_tk; self.video_canvas.config(image=img_tk)
        if not self.stop_event.is_set(): self.root.after(20, self.update_video_canvas)
    def video_reader_worker(self):
        cap = cv2.VideoCapture(self.video_path)
        while not self.stop_event.is_set():
            if self.raw_frame_queue.full(): time.sleep(0.01); continue
            ret, frame = cap.read()
            if not ret: self.raw_frame_queue.put(None); break
            self.raw_frame_queue.put(frame)
        cap.release(); print("Thread pembaca video berhenti.")
    def run_detection_worker(self):
        try:
            model = YOLO(self.model_path)
            first_frame = self.raw_frame_queue.get();
            if first_frame is None: return
            roi_display_frame = cv2.resize(first_frame, (CONFIG["DISPLAY_WIDTH"], CONFIG["DISPLAY_HEIGHT"]))
            vehicle_roi = select_roi_on_frame(roi_display_frame)
            if vehicle_roi is None: self.stop_event.set(); return
            original_h, original_w = first_frame.shape[:2]
            scale_w, scale_h = original_w / CONFIG["DISPLAY_WIDTH"], original_h / CONFIG["DISPLAY_HEIGHT"]
            vehicle_roi = (vehicle_roi * [scale_w, scale_h]).astype(int)
            class_names_ordered = list(model.names.values()); setup_csv(class_names_ordered)
            traffic_state, stop_start_time, track_history = 'BERGERAK', None, {}
            max_cycle_counts = {name: 0 for name in class_names_ordered}
            stop_counter, move_counter, frame_processed_count = 0, 0, 0
            start_time = datetime.now()
            self.lbl_status.config(text="Status: Deteksi sedang berjalan...")
            while not self.stop_event.is_set():
                frame = self.raw_frame_queue.get()
                if frame is None: self.processed_frame_queue.put(None); break
                frame_processed_count += 1
                displacements, current_track_ids = [], set()
                current_frame_counts = {name: 0 for name in class_names_ordered}
                results = model.track(frame, imgsz=CONFIG["YOLO_IMG_SIZE"], conf=CONFIG["CONFIDENCE_THRESHOLD"], persist=True, verbose=False, device=0)
                if results[0].boxes.id is not None:
                    boxes, track_ids, class_ids = results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.id.cpu().numpy().astype(int), results[0].boxes.cls.cpu().numpy().astype(int)
                    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                        center_point = (int((box[0] + box[2]) / 2), int(box[3]))
                        if cv2.pointPolygonTest(vehicle_roi, center_point, False) >= 0:
                            current_track_ids.add(track_id); current_frame_counts[model.names[class_id]] += 1
                            if track_id in track_history:
                                distance = np.linalg.norm(np.array(center_point) - np.array(track_history[track_id]))
                                displacements.append(distance)
                            track_history[track_id] = center_point
                            x1, y1, x2, y2 = map(int, box); label = model.names[class_id]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                avg_speed = np.mean(displacements) if displacements else 100
                is_stopped_now = avg_speed < CONFIG["STOP_THRESHOLD_PIXELS"]
                if is_stopped_now: move_counter, stop_counter = 0, stop_counter + 1
                else: stop_counter, move_counter = 0, move_counter + 1
                if stop_counter > CONFIG["STOP_CONFIRMATION_FRAMES"] and traffic_state == 'BERGERAK':
                    traffic_state, stop_start_time = 'BERHENTI', datetime.now()
                    max_cycle_counts = current_frame_counts.copy(); print(f"Status TERKONFIRMASI: BERHENTI (Kecepatan Rata-rata: {avg_speed:.2f})")
                    stop_counter = 0
                elif move_counter > CONFIG["MOVE_CONFIRMATION_FRAMES"] and traffic_state == 'BERHENTI':
                    traffic_state, stop_end_time = 'BERGERAK', datetime.now()
                    duration = (stop_end_time - stop_start_time).total_seconds()
                    if duration > CONFIG["MIN_STOP_DURATION_SECONDS"]:
                        log_to_csv(stop_end_time.strftime("%Y-%m-%d %H:%M:%S"), duration, max_cycle_counts, class_names_ordered)
                        print(f"Status TERKONFIRMASI: BERGERAK (Kecepatan Rata-rata: {avg_speed:.2f}). Data tersimpan.")
                    stop_start_time, move_counter = None, 0
                if traffic_state == 'BERHENTI' and sum(current_frame_counts.values()) > sum(max_cycle_counts.values()):
                    max_cycle_counts = current_frame_counts.copy()
                track_history = {tid: pos for tid, pos in track_history.items() if tid in current_track_ids}
                duration = (datetime.now() - start_time).total_seconds()
                fps = frame_processed_count / duration if duration > 0 else 0
                state_color = (0, 0, 255) if traffic_state == 'BERHENTI' else (0, 255, 0)
                cv2.polylines(frame, [vehicle_roi], isClosed=True, color=(255, 0, 0), thickness=3)
                cv2.putText(frame, f"STATUS: {traffic_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3); cv2.putText(frame, f"STATUS: {traffic_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)
                cv2.putText(frame, f"Processing FPS: {fps:.2f}", (frame.shape[1] - 350, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                display_counts = max_cycle_counts if traffic_state == 'BERHENTI' else {name: 0 for name in class_names_ordered}
                y_pos = 80
                for class_name, count in display_counts.items(): cv2.putText(frame, f"{class_name.title()}: {count}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2); y_pos += 35
                if not self.processed_frame_queue.full():
                    display_frame = cv2.resize(frame, (CONFIG["DISPLAY_WIDTH"], CONFIG["DISPLAY_HEIGHT"]))
                    self.processed_frame_queue.put(display_frame)
        except Exception as e: messagebox.showerror("Error pada Worker Thread", f"Terjadi kesalahan: {e}")
        finally:
            self.stop_event.set(); print("Thread pemroses YOLO berhenti.")
            self.root.after(0, self.stop_detection_from_thread)
    def stop_detection_from_thread(self):
        if self.btn_start['state'] == tk.DISABLED: self.stop_detection()

if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleDetectorApp(root)
    root.mainloop()