# -*- coding: utf-8 -*-
"""
Final Perfected Version: Stable & Smooth Vehicle Detector
Author: Gemini
Description: This script uses Vehicle Motion Analysis with a confirmation buffer for stable state
             detection and an intelligent frame display mechanism for a smooth GUI experience.
"""
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import threading
import queue
from PIL import Image, ImageTk
import csv
from datetime import datetime

# --- Variabel Global untuk Menggambar ROI ---
roi_points = []

def on_mouse_roi_selection(event, x, y, flags, param):
    """Callback function to handle mouse events for ROI selection."""
    global roi_points
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))

def select_roi_on_frame(frame):
    """Displays a window for the user to draw a polygonal Region of Interest (ROI)."""
    global roi_points
    roi_points = []
    clone = frame.copy()
    window_name = "LANGKAH 1: PILIH AREA DETEKSI KENDARAAN (ROI)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_roi_selection)

    print("\n" + "="*50)
    print("Panduan Memilih Area (ROI):")
    print("- Klik kiri untuk menandai titik sudut area jalan.")
    print("- Tekan tombol 'ENTER' untuk menyelesaikan.")
    print("- Tekan tombol 'c' untuk menghapus & mengulang.")
    print("="*50 + "\n")

    while True:
        temp_frame = clone.copy()
        if len(roi_points) > 0:
            cv2.polylines(temp_frame, [np.array(roi_points)], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.putText(temp_frame, "Klik: Tambah Titik | ENTER: Selesai | C: Reset", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            if len(roi_points) > 2: break
            else: print("Error: Area membutuhkan minimal 3 titik.")
        elif key == ord('c'): roi_points = []
    cv2.destroyWindow(window_name)
    return np.array(roi_points) if len(roi_points) > 2 else None

# --- Fungsi untuk Menangani File CSV ---
CSV_FILENAME = "log_volume_kendaraan.csv"

def setup_csv(class_names):
    """Creates CSV file and writes the header if it doesn't exist."""
    try:
        with open(CSV_FILENAME, 'x', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            headers = ["Timestamp Selesai Berhenti", "Durasi Berhenti (detik)"] + [name.title() for name in class_names]
            writer.writerow(headers)
    except FileExistsError: pass

def log_to_csv(timestamp, duration, counts, class_names_ordered):
    """Appends a single row of data to the CSV file."""
    with open(CSV_FILENAME, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        row = [timestamp, f"{duration:.2f}"] + [counts.get(name, 0) for name in class_names_ordered]
        writer.writerow(row)

class VehicleDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Deteksi Kendaraan v3.0 (Stabil & Mulus)")
        self.root.geometry("1000x800")
        self.model_path, self.video_path = "", ""
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=10)
        self.setup_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        """Initializes all GUI components."""
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10, padx=10, fill=tk.X)
        tk.Label(control_frame, text="Detektor Kendaraan YOLO", font=("Helvetica", 18, "bold")).pack()
        file_frame = tk.Frame(control_frame)
        file_frame.pack(pady=10)
        self.btn_model = tk.Button(file_frame, text="Pilih Model (.pt)", command=self.load_model, width=20)
        self.btn_model.grid(row=0, column=0, padx=5, pady=5)
        self.lbl_model_path = tk.Label(file_frame, text="Model belum dipilih", fg="red", anchor="w")
        self.lbl_model_path.grid(row=0, column=1, padx=5, sticky="w")
        self.btn_video = tk.Button(file_frame, text="Pilih Video", command=self.load_video, width=20)
        self.btn_video.grid(row=1, column=0, padx=5, pady=5)
        self.lbl_video_path = tk.Label(file_frame, text="Video belum dipilih", fg="red", anchor="w")
        self.lbl_video_path.grid(row=1, column=1, padx=5, sticky="w")
        action_frame = tk.Frame(control_frame)
        action_frame.pack(pady=10)
        self.btn_start = tk.Button(action_frame, text="Mulai Deteksi", command=self.start_detection, font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white", width=15)
        self.btn_start.pack(side=tk.LEFT, padx=10)
        self.btn_stop = tk.Button(action_frame, text="Hentikan", command=self.stop_detection, font=("Helvetica", 12), bg="#F44336", fg="white", state=tk.DISABLED, width=15)
        self.btn_stop.pack(side=tk.LEFT, padx=10)
        self.video_canvas = tk.Label(self.root, bg="black")
        self.video_canvas.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)
        self.lbl_status = tk.Label(self.root, text="Status: Siap", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

    def load_model(self):
        path = filedialog.askopenfilename(title="Pilih file model YOLO", filetypes=[("YOLO Model", "*.pt")])
        if path: self.model_path = path; self.lbl_model_path.config(text=f"Model: ...{path[-30:]}", fg="green")

    def load_video(self):
        path = filedialog.askopenfilename(title="Pilih file video", filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
        if path: self.video_path = path; self.lbl_video_path.config(text=f"Video: ...{path[-30:]}", fg="green")

    def start_detection(self):
        if not self.model_path or not self.video_path: messagebox.showerror("Error", "Silakan pilih file model dan video terlebih dahulu!"); return
        if self.processing_thread and self.processing_thread.is_alive(): messagebox.showinfo("Info", "Proses deteksi sudah berjalan."); return
        self.btn_start.config(state=tk.DISABLED); self.btn_stop.config(state=tk.NORMAL)
        self.stop_event.clear()
        self.processing_thread = threading.Thread(target=self.run_detection_worker); self.processing_thread.daemon = True
        self.processing_thread.start()
        self.update_video_canvas()

    def stop_detection(self):
        self.lbl_status.config(text="Status: Menghentikan..."); self.stop_event.set()
        self.btn_start.config(state=tk.NORMAL); self.btn_stop.config(state=tk.DISABLED)

    def on_closing(self):
        self.stop_event.set(); self.root.destroy()

    def update_video_canvas(self):
        """Menarik frame TERBARU dari antrean untuk tampilan paling mulus."""
        frame = None
        try:
            # Tampilan Cerdas: kosongkan antrean dan ambil frame terakhir
            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
        except queue.Empty:
            pass

        if frame is not None:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img)
            self.video_canvas.img_tk = img_tk
            self.video_canvas.config(image=img_tk)
            
        if not self.stop_event.is_set():
            # Kecepatan refresh tampilan sekitar 40 FPS
            self.root.after(25, self.update_video_canvas)

    def run_detection_worker(self):
        try:
            DISPLAY_WIDTH, DISPLAY_HEIGHT = 1280, 720
            model = YOLO(self.model_path)
            cap = cv2.VideoCapture(self.video_path)
            ret, first_frame = cap.read()
            if not ret: raise ValueError("Gagal membaca frame pertama dari video.")

            roi_display_frame = cv2.resize(first_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            vehicle_roi = select_roi_on_frame(roi_display_frame)
            if vehicle_roi is None: self.stop_detection(); self.lbl_status.config(text="Status: Dibatalkan, ROI tidak dipilih."); return

            original_h, original_w = first_frame.shape[:2]
            scale_w, scale_h = original_w / DISPLAY_WIDTH, original_h / DISPLAY_HEIGHT
            vehicle_roi = (vehicle_roi * [scale_w, scale_h]).astype(int)

            class_names_ordered = list(model.names.values())
            setup_csv(class_names_ordered)
            
            traffic_state = 'BERGERAK'
            stop_start_time = None
            track_history = {}
            max_cycle_counts = {name: 0 for name in class_names_ordered}
            
            # --- Variabel Kunci untuk Stabilitas ---
            # 'Kenop' yang bisa Anda sesuaikan untuk stabilitas
            STOP_CONFIRMATION_FRAMES = 15  # Harus berhenti selama 15 frame untuk valid
            MOVE_CONFIRMATION_FRAMES = 8   # Harus bergerak selama 8 frame untuk valid
            stop_counter, move_counter = 0, 0
            
            self.lbl_status.config(text="Status: Deteksi sedang berjalan...")
            
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret: break
                
                displacements, current_track_ids = [], set()
                current_frame_counts = {name: 0 for name in class_names_ordered}
                
                results = model.track(frame, persist=True, verbose=False, device=0)

                if results[0].boxes.id is not None:
                    boxes, track_ids, class_ids = results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.id.cpu().numpy().astype(int), results[0].boxes.cls.cpu().numpy().astype(int)
                    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                        center_point = (int((box[0] + box[2]) / 2), int(box[3]))
                        if cv2.pointPolygonTest(vehicle_roi, center_point, False) >= 0:
                            current_track_ids.add(track_id)
                            current_frame_counts[model.names[class_id]] += 1
                            if track_id in track_history:
                                distance = np.linalg.norm(np.array(center_point) - np.array(track_history[track_id]))
                                displacements.append(distance)
                            track_history[track_id] = center_point
                            
                            x1, y1, x2, y2 = map(int, box)
                            label = model.names[class_id]
                            box_color = (0, 255, 0) # Hijau
                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                
                avg_speed = np.mean(displacements) if displacements else 100
                STOP_THRESHOLD = 3.0
                is_stopped_now = avg_speed < STOP_THRESHOLD

                # --- Logika State Machine BARU dengan Buffer Konfirmasi ---
                if is_stopped_now: move_counter, stop_counter = 0, stop_counter + 1
                else: stop_counter, move_counter = 0, move_counter + 1

                if stop_counter > STOP_CONFIRMATION_FRAMES and traffic_state == 'BERGERAK':
                    traffic_state, stop_start_time = 'BERHENTI', datetime.now()
                    max_cycle_counts = current_frame_counts.copy()
                    print(f"Status TERKONFIRMASI: BERHENTI (Kecepatan Rata-rata: {avg_speed:.2f})")
                    stop_counter = 0

                elif move_counter > MOVE_CONFIRMATION_FRAMES and traffic_state == 'BERHENTI':
                    traffic_state, stop_end_time = 'BERGERAK', datetime.now()
                    duration = (stop_end_time - stop_start_time).total_seconds()
                    if duration > 2:
                        log_to_csv(stop_end_time.strftime("%Y-%m-%d %H:%M:%S"), duration, max_cycle_counts, class_names_ordered)
                        print(f"Status TERKONFIRMASI: BERGERAK (Kecepatan Rata-rata: {avg_speed:.2f}). Data tersimpan.")
                    stop_start_time, move_counter = None, 0

                if traffic_state == 'BERHENTI' and sum(current_frame_counts.values()) > sum(max_cycle_counts.values()):
                    max_cycle_counts = current_frame_counts.copy()

                track_history = {tid: pos for tid, pos in track_history.items() if tid in current_track_ids}
                
                # --- Visualisasi ---
                cv2.polylines(frame, [vehicle_roi], isClosed=True, color=(255, 0, 0), thickness=3)
                state_color = (0, 0, 255) if traffic_state == 'BERHENTI' else (0, 255, 0)
                cv2.putText(frame, f"STATUS: {traffic_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3); cv2.putText(frame, f"STATUS: {traffic_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)
                display_counts = max_cycle_counts if traffic_state == 'BERHENTI' else {name: 0 for name in class_names_ordered}
                y_pos = 80
                for class_name, count in display_counts.items(): cv2.putText(frame, f"{class_name.title()}: {count}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2); y_pos += 35
                
                display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                if not self.frame_queue.full(): self.frame_queue.put(display_frame)

        except Exception as e: messagebox.showerror("Error pada Worker Thread", f"Terjadi kesalahan: {e}"); self.lbl_status.config(text=f"Status: Error - {e}")
        finally:
            if 'cap' in locals() and cap.isOpened(): cap.release()
            self.stop_event.set(); self.root.after(0, self.stop_detection)
            self.lbl_status.config(text="Status: Selesai."); print("Worker thread finished.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleDetectorApp(root)
    root.mainloop()