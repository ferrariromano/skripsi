import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import threading

# --- Variabel Global untuk Menggambar ROI ---
roi_points = []
drawing_complete = False

def on_mouse(event, x, y, flags, param):
    """
    Callback function untuk menangani event mouse saat menggambar ROI.
    """
    global roi_points, drawing_complete

    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing_complete:
            roi_points.append((x, y))
    
    # Event untuk mengakhiri gambar (opsional, bisa diganti dengan menekan tombol)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(roi_points) > 2:
            drawing_complete = True


def select_roi(frame):
    """
    Fungsi untuk menampilkan frame pertama dan memungkinkan pengguna menggambar ROI.
    """
    global roi_points, drawing_complete
    roi_points = []
    drawing_complete = False
    
    clone = frame.copy()
    window_name = "Pilih Area Deteksi (ROI)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    print("==========================================================")
    print("Panduan Memilih Area (ROI):")
    print("- Klik kiri untuk menambahkan titik sudut area.")
    print("- Buat minimal 3 titik untuk membentuk poligon.")
    print("- Tekan tombol 'ENTER' jika sudah selesai menggambar.")
    print("- Tekan tombol 'C' untuk menghapus dan mengulang.")
    print("==========================================================")

    while True:
        # Tampilkan frame saat ini
        temp_frame = clone.copy()
        
        # Gambar titik dan garis yang sudah dibuat
        if len(roi_points) > 0:
            for point in roi_points:
                cv2.circle(temp_frame, point, 5, (0, 255, 0), -1)
            
            if len(roi_points) > 1:
                cv2.polylines(temp_frame, [np.array(roi_points)], isClosed=False, color=(0, 255, 0), thickness=2)

        cv2.putText(temp_frame, "Klik kiri: Tambah titik. ENTER: Selesai. C: Reset", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(window_name, temp_frame)

        key = cv2.waitKey(1) & 0xFF
        
        # Tombol ENTER untuk menyelesaikan
        if key == 13: # 13 adalah kode ASCII untuk Enter
            if len(roi_points) > 2:
                drawing_complete = True
                break
            else:
                print("Error: Anda butuh minimal 3 titik untuk membuat area.")

        # Tombol C untuk clear/reset
        elif key == ord('c'):
            roi_points = []
            drawing_complete = False

    cv2.destroyWindow(window_name)
    return np.array(roi_points) if drawing_complete else None


class VehicleDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detektor Kendaraan Persimpangan")
        self.root.geometry("500x300")

        self.model_path = ""
        self.video_path = ""
        self.processing_thread = None

        # --- Tampilan GUI ---
        # Label
        tk.Label(root, text="Program Deteksi Kendaraan YOLO", font=("Helvetica", 16, "bold")).pack(pady=10)

        # Frame untuk tombol
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        # Tombol
        self.btn_model = tk.Button(btn_frame, text="1. Pilih Model (.pt)", command=self.load_model, width=20)
        self.btn_model.grid(row=0, column=0, padx=5, pady=5)

        self.btn_video = tk.Button(btn_frame, text="2. Pilih Video", command=self.load_video, width=20)
        self.btn_video.grid(row=1, column=0, padx=5, pady=5)

        self.btn_start = tk.Button(root, text="3. Mulai Deteksi", command=self.start_detection_thread, font=("Helvetica", 12, "bold"), bg="green", fg="white")
        self.btn_start.pack(pady=20)

        # Label untuk menampilkan path file
        self.lbl_model_path = tk.Label(root, text="Model belum dipilih", fg="red")
        self.lbl_model_path.pack()

        self.lbl_video_path = tk.Label(root, text="Video belum dipilih", fg="red")
        self.lbl_video_path.pack()

        # Label untuk status
        self.lbl_status = tk.Label(root, text="", font=("Helvetica", 10))
        self.lbl_status.pack(pady=10)


    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("YOLO Model", "*.pt")])
        if path:
            self.model_path = path
            self.lbl_model_path.config(text=f"Model: {path.split('/')[-1]}", fg="green")
            print(f"Model dipilih: {self.model_path}")

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
        if path:
            self.video_path = path
            self.lbl_video_path.config(text=f"Video: {path.split('/')[-1]}", fg="green")
            print(f"Video dipilih: {self.video_path}")

    def start_detection_thread(self):
        if not self.model_path or not self.video_path:
            messagebox.showerror("Error", "Silakan pilih file model dan video terlebih dahulu!")
            return

        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showinfo("Info", "Proses deteksi sudah berjalan.")
            return

        # Jalankan proses deteksi di thread terpisah agar GUI tidak macet
        self.processing_thread = threading.Thread(target=self.run_detection)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.lbl_status.config(text="Status: Memulai proses deteksi...")

    def run_detection(self):
        try:
            # 1. Muat Model YOLO
            model = YOLO(self.model_path)
            class_names = model.names
            print("Nama kelas pada model:", class_names)

            # 2. Buka Video
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                messagebox.showerror("Error", "Gagal membuka file video.")
                self.lbl_status.config(text="Status: Gagal membuka video.")
                return
            
            # 3. Ambil frame pertama untuk memilih ROI
            ret, first_frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Gagal membaca frame pertama dari video.")
                self.lbl_status.config(text="Status: Gagal membaca video.")
                return

            roi = select_roi(first_frame)
            if roi is None:
                print("Pemilihan ROI dibatalkan. Proses dihentikan.")
                self.lbl_status.config(text="Status: Dibatalkan, ROI tidak dipilih.")
                cap.release()
                return

            # --- PERUBAHAN 1: Inisialisasi penghitung per kelas ---
            # Menggunakan dictionary untuk menyimpan hitungan setiap kelas
            class_counts = {name: 0 for name in class_names.values()}
            counted_track_ids = set() # Tetap gunakan ini untuk memastikan setiap kendaraan hanya dihitung sekali

            self.lbl_status.config(text="Status: Deteksi sedang berjalan... (Tekan 'q' di jendela video untuk berhenti)")
            
            # 5. Loop utama untuk memproses setiap frame video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Gambar ROI di setiap frame
                cv2.polylines(frame, [roi], isClosed=True, color=(255, 0, 0), thickness=2)

                # Lakukan deteksi dan pelacakan (tracking)
                results = model.track(frame, persist=True, verbose=False)

                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                        x1, y1, x2, y2 = box
                        center_point = (int((x1 + x2) / 2), int(y2))
                        
                        is_inside = cv2.pointPolygonTest(roi, center_point, False)
                        
                        # --- PERUBAHAN 2: Logika penghitungan diperbarui ---
                        if is_inside >= 0:
                            # Jika kendaraan ada di dalam dan ID-nya belum dihitung
                            if track_id not in counted_track_ids:
                                class_name = class_names[class_id]
                                class_counts[class_name] += 1 # Tambah hitungan ke kelas yang sesuai
                                counted_track_ids.add(track_id)
                                print(f"Kendaraan terdeteksi! Tipe: {class_name}, ID: {track_id}. Total {class_name}: {class_counts[class_name]}")

                        # --- PERUBAHAN 3: Label disederhanakan ---
                        label = class_names[class_id] # Hanya menampilkan nama kelas
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


                # --- PERUBAHAN 4: Tampilkan semua hitungan kelas di frame ---
                y_pos = 50
                # Tambahkan background semi-transparan untuk keterbacaan
                sub_img = frame.copy()
                cv2.rectangle(sub_img, (5, y_pos - 25), (350, y_pos + (len(class_counts) * 35)), (0, 0, 0), -1)
                alpha = 0.5 # Transparansi
                frame = cv2.addWeighted(sub_img, alpha, frame, 1 - alpha, 0)

                for class_name, count in class_counts.items():
                    info_text = f"{class_name.title()}: {count}" # .title() agar huruf awal besar
                    cv2.putText(frame, info_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    y_pos += 35 # Beri jarak untuk baris berikutnya

                cv2.imshow("Deteksi Kendaraan di Persimpangan", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Selesai
            cap.release()
            cv2.destroyAllWindows()
            
            # Buat ringkasan hasil untuk messagebox
            summary = "Proses deteksi selesai.\n\nHasil Hitungan:\n"
            for class_name, count in class_counts.items():
                summary += f"- {class_name.title()}: {count}\n"

            self.lbl_status.config(text="Status: Selesai. Lihat rincian di pesan popup.")
            messagebox.showinfo("Selesai", summary)

        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan: {e}")
            self.lbl_status.config(text=f"Status: Error - {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleDetectorApp(root)
    root.mainloop()