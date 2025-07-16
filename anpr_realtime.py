import cv2
import numpy as np
import time
import threading # Impor modul threading
import queue     # Impor modul queue untuk komunikasi antar thread
from imutils.video import VideoStream # Still imported, but not used if cv2.VideoCapture is used for file
from ultralytics import YOLO
import easyocr
import os
import re # Impor modul regex untuk validasi plat nomor
import torch # Impor torch untuk memindahkan model YOLO ke GPU
import datetime # Impor modul datetime untuk timestamp logging

# --- Konfigurasi dan Pemuatan Model ---

# Path ke model YOLOv8.
# Ultralytics akan mengunduh model ini secara otomatis jika belum ada.
# Pilih versi yang sesuai: 'yolov8n.pt' (nano, fastest), 'yolov8s.pt' (small), 'yolov8m.pt' (medium)
PATH_TO_YOLOV8_MODEL = 'yolov8n.pt'

# Inisialisasi EasyOCR reader
print("[INFO] Memuat model EasyOCR (mungkin butuh waktu pertama kali)...")
# EasyOCR secara default akan mencoba menggunakan GPU jika tersedia
reader = easyocr.Reader(['en']) # 'en' for Latin characters on license plates
print("[INFO] EasyOCR model loaded successfully.")

# Load the YOLOv8 model
print(f"[INFO] Memuat model deteksi objek YOLOv8 dari {PATH_TO_YOLOV8_MODEL}...")
yolo_model = YOLO(PATH_TO_YOLOV8_MODEL)

# OPTIMASI 1: Memastikan Pemanfaatan GPU Maksimal untuk YOLOv8
# Pindahkan model YOLOv8 ke GPU jika tersedia
if torch.cuda.is_available():
    yolo_model.to('cuda')
    print("[INFO] YOLOv8 model dipindahkan ke GPU.")
else:
    print("[INFO] GPU tidak tersedia, YOLOv8 akan berjalan di CPU.")
print("[INFO] YOLOv8 model berhasil dimuat.")


# Define object classes to be detected as 'vehicles' from the COCO dataset
# These class IDs are standard for the COCO dataset:
# 2: car, 3: motorcycle, 5: bus, 7: truck
VEHICLE_CLASSES_IDS = [2, 3, 5, 7]

# OPTIMASI 5: Post-processing dan Validasi Plat Nomor yang Lebih Kuat
# Regex untuk format plat nomor Indonesia (contoh sederhana: AA 1234 BBB)
# Ini bisa disesuaikan lebih lanjut untuk mencakup semua variasi
# Contoh: [A-Z]{1,2} (1 atau 2 huruf awal), \s? (spasi opsional), \d{1,4} (1-4 angka), \s? (spasi opsional), [A-Z]{1,3} (1-3 huruf akhir)
INDONESIAN_PLATE_REGEX = r"^[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{1,3}$"

# Mapping untuk koreksi kesalahan OCR umum (O vs 0, I vs 1, dll.)
OCR_CORRECTION_MAP = {
    'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'Z': '2'
}

# --- Task 6: Validasi dan Format Keluaran - Basis Data (Simulasi) ---
# Simulasi database plat nomor terdaftar. Dalam aplikasi nyata, ini akan terhubung ke database.
REGISTERED_PLATES = ["B1234ABC", "D5678XYZ", "F9012UVW"]
LOG_FILE_PATH = "anpr_log.txt" # Path untuk file log

# --- Vehicle Detection Function with YOLOv8 ---

def detect_vehicles(frame, model, confidence_threshold=0.5):
    """
    Detects vehicles in a frame using YOLOv8.
    YOLOv8 internally performs pre-processing such as resizing.
    """
    # Perform inference with the YOLOv8 model
    # conf: confidence threshold for detection. Meningkatkan ini akan mengurangi deteksi lemah.
    results = model(frame, conf=confidence_threshold, verbose=False)

    detected_vehicles = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0]) # Class ID
            conf = float(box.conf[0]) # Confidence score

            if cls in VEHICLE_CLASSES_IDS: # Check if it's a vehicle class we want
                x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box coordinates (x_min, y_min, x_max, y_max)
                w = x2 - x1
                h = y2 - y1
                label = model.names[cls] # Class name (e.g., 'car', 'truck')

                detected_vehicles.append({
                    'box': (x1, y1, w, h), # Format (x, y, width, height)
                    'label': label,
                    'confidence': conf
                })
    return detected_vehicles

# --- Task 4: Normalisasi dan Segmentasi Karakter ---
# Fungsi ini akan mengintegrasikan langkah-langkah normalisasi dan segmentasi
# untuk area plat nomor yang terdeteksi sebelum diberikan ke OCR.

def deskew_plate(image):
    """
    Melakukan deskewing (koreksi kemiringan) pada gambar plat nomor.
    Ini adalah implementasi sederhana dan mungkin perlu disempurnakan
    untuk performa dan akurasi yang lebih baik.
    """
    # Pastikan gambar adalah grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Thresholding untuk mendapatkan citra biner
    # Menggunakan THRESH_BINARY_INV untuk karakter gelap di latar belakang terang
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Temukan kontur
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Cari kontur terbesar (diasumsikan sebagai plat nomor)
    if not contours:
        return image # Return original if no contours found
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Dapatkan bounding box berputar
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]

    # Jika sudut terlalu besar, sesuaikan
    # Sudut dari minAreaRect berada dalam rentang [-90, 0)
    # Kita ingin mengoreksi agar teks menjadi horizontal.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Dapatkan matriks rotasi
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Lakukan rotasi
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def detect_and_recognize_license_plates(frame, vehicle_boxes, reader):
    """
    Detects and recognizes license plates within detected vehicle areas,
    or from the entire frame if no vehicles are detected.
    Uses EasyOCR for text detection and recognition.
    """
    detected_plates_info = []

    search_areas = []
    if vehicle_boxes:
        for vehicle in vehicle_boxes:
            x, y, w, h = vehicle['box']
            # Ciri-ciri Plat Nomor: Rasio Aspek & Lokasi Heuristik
            # Persempit area pencarian plat nomor ke bagian bawah kendaraan
            plate_search_x = max(0, x)
            plate_search_y = max(0, y + int(h * 0.5)) # Start from the middle downwards
            plate_search_w = min(w, frame.shape[1] - plate_search_x)
            plate_search_h = min(int(h * 0.5), frame.shape[0] - plate_search_y) # Bottom half

            if plate_search_w > 0 and plate_search_h > 0:
                search_areas.append((frame[plate_search_y : plate_search_y + plate_search_h,
                                            plate_search_x : plate_search_x + plate_search_w],
                                     plate_search_x, plate_search_y))

    # Fallback: if no vehicles are detected, or the area within the vehicle is empty,
    # try searching the entire frame (less efficient but can catch missed plates)
    if not search_areas:
        search_areas.append((frame, 0, 0)) # Entire frame with 0,0 offset

    for area_img, offset_x, offset_y in search_areas:
        if area_img.shape[0] == 0 or area_img.shape[1] == 0:
            continue

        # --- Task 4: Normalisasi dan Segmentasi Karakter ---
        # Ini adalah bagian penting untuk menonjolkan ciri-ciri plat nomor (tekstur, kontur, warna kontras)
        # dan mempersiapkannya untuk pengenalan karakter yang akurat.
        # Eksperimen dengan langkah-langkah ini untuk menyeimbangkan kecepatan dan akurasi.
        # Setiap langkah menambah beban komputasi.
        processed_area = area_img.copy()
        
        # Task 4.1: Koreksi Perspektif (Deskewing)
        # Meluruskan plat nomor yang miring. Sangat penting untuk akurasi OCR.
        processed_area = deskew_plate(processed_area)

        # Task 4.2: Peningkatan Kontras / Konversi Grayscale
        # Konversi Grayscale: Seringkali membantu untuk OCR karena mengurangi dimensi warna.
        processed_area = cv2.cvtColor(processed_area, cv2.COLOR_BGR2GRAY)

        # Peningkatan Kontras dengan CLAHE: Berguna untuk plat nomor dengan pencahayaan tidak merata,
        # meningkatkan kontras lokal antara karakter dan latar belakang.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed_area = clahe.apply(processed_area)

        # Pengurangan Derau (Noise Reduction) dengan Gaussian Blur: Sangat direkomendasikan
        # untuk menghaluskan gambar dan menghilangkan derau yang mengganggu karakter.
        processed_area = cv2.GaussianBlur(processed_area, (5, 5), 0)

        # Task 4.3: Thresholding (Mengubah gambar menjadi biner)
        # Baik untuk memisahkan karakter dari latar belakang plat nomor secara adaptif
        # terhadap variasi pencahayaan.
        processed_area = cv2.adaptiveThreshold(processed_area, 255, 
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
        
        # Operasi Morfologi (Erosi/Dilasi):
        # Erosi dapat menghilangkan noise kecil, Dilasi dapat menyatukan bagian karakter yang terpisah.
        # Hati-hati, bisa merusak karakter jika terlalu agresif. Coba nonaktifkan jika ada masalah.
        kernel = np.ones((3,3),np.uint8) # Kernel kecil
        processed_area = cv2.erode(processed_area, kernel, iterations = 1)
        processed_area = cv2.dilate(processed_area, kernel, iterations = 1)

        # Task 4.4: Segmentasi Karakter (Implisit oleh EasyOCR)
        # EasyOCR secara internal melakukan segmentasi karakter setelah mendeteksi area teks.
        # Jika Anda ingin segmentasi karakter eksplisit (misalnya untuk visualisasi atau OCR kustom),
        # Anda bisa menggunakan cv2.findContours pada 'processed_area' biner, lalu memfilter kontur
        # berdasarkan rasio aspek, ukuran, dan area untuk mengidentifikasi setiap karakter.
        # Namun, untuk alur dengan EasyOCR, ini tidak diperlukan secara langsung.

        # --- End of Additional Pre-processing ---

        # --- Task 5: Pengenalan Karakter Optik (OCR) ---
        # Metode Deteksi: Jaringan Saraf Tiruan (CNN) - EasyOCR
        # EasyOCR menggunakan model deep learning untuk mendeteksi area teks (plat nomor)
        # dan mengenali karakternya.
        results = reader.readtext(processed_area, detail=1)

        for (bbox, text, prob) in results:
            # Bersihkan teks: hanya ambil alfanumerik dan ubah ke huruf kapital
            cleaned_text = "".join(filter(str.isalnum, text)).upper()

            # OPTIMASI 5: Koreksi Kesalahan Umum OCR
            for original, corrected in OCR_CORRECTION_MAP.items():
                cleaned_text = cleaned_text.replace(original, corrected)

            # Task 6.1: Pemeriksaan Aturan (Validasi Format Plat Nomor)
            # Filter results based on length and probability.
            # Indonesian license plates generally have 4-10 alphanumeric characters.
            # Meningkatkan probabilitas (prob > 0.5) akan mengurangi deteksi yang kurang yakin.
            if re.match(INDONESIAN_PLATE_REGEX, cleaned_text) and prob > 0.5:
                # Transform bounding box coordinates from area_img to original frame
                (top_left, _, bottom_right, _) = bbox
                x_plate = int(top_left[0]) + offset_x
                y_plate = int(top_left[1]) + offset_y
                w_plate = int(bottom_right[0] - top_left[0])
                h_plate = int(bottom_right[1] - top_left[1])

                detected_plates_info.append({
                    'plate_box': (x_plate, y_plate, w_plate, h_plate),
                    'text': cleaned_text,
                    'confidence': prob
                })
    return detected_plates_info

# --- Task 6: Validasi dan Format Keluaran - Fungsi Pembantu ---

def log_plate_detection(plate_text, status):
    """
    Task 6.3: Penyimpanan/Tampilan - Simpan ke log.
    Mencatat deteksi plat nomor ke file log.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Plat: {plate_text}, Status: {status}\n"
    with open(LOG_FILE_PATH, "a") as f:
        f.write(log_entry)
    print(f"[LOG] {log_entry.strip()}") # Juga cetak ke konsol

def check_database(plate_text):
    """
    Task 6.2: Basis Data - Bandingkan hasil dengan database plat nomor.
    Simulasi pengecekan database. Dalam aplikasi nyata, ini akan query database.
    """
    return plate_text in REGISTERED_PLATES

# --- OPTIMASI 7: Arsitektur Multi-threading ---

# Queue untuk frame mentah dari pembaca video
raw_frame_queue = queue.Queue(maxsize=5)
# Queue untuk frame yang sudah diproses (dengan deteksi dan plat nomor)
processed_frame_queue = queue.Queue(maxsize=5)
# Event untuk sinyal berhenti ke thread
stop_event = threading.Event()

class FrameReader(threading.Thread):
    """
    Thread untuk membaca frame dari sumber video dan memasukkannya ke dalam queue.
    """
    def __init__(self, video_path, queue, stop_event):
        super().__init__()
        self.video_path = video_path
        self.queue = queue
        self.stop_event = stop_event
        self.vs = None
        self.daemon = True # Set thread sebagai daemon agar berhenti saat program utama berhenti

    def run(self):
        print("[INFO] Thread FrameReader dimulai...")
        self.vs = cv2.VideoCapture(self.video_path)
        if not self.vs.isOpened():
            print(f"[ERROR] FrameReader: Tidak dapat membuka file video: {self.video_path}")
            self.stop_event.set() # Set event untuk menghentikan thread lain
            return

        while not self.stop_event.is_set():
            ret, frame = self.vs.read()
            if not ret:
                print("[INFO] FrameReader: Akhir stream video atau masalah membaca frame.")
                self.stop_event.set() # Set event untuk menghentikan thread lain
                break
            
            # Coba masukkan frame ke queue, jika penuh, lewati (untuk menjaga real-time)
            try:
                self.queue.put(frame, block=False)
            except queue.Full:
                pass # Queue penuh, lewati frame ini

        self.vs.release()
        print("[INFO] Thread FrameReader dihentikan.")

class FrameProcessor(threading.Thread):
    """
    Thread untuk mengambil frame dari raw_frame_queue, memprosesnya (deteksi YOLO & OCR),
    dan memasukkan frame yang sudah diproses ke processed_frame_queue.
    """
    def __init__(self, raw_queue, processed_queue, stop_event, yolo_model, easyocr_reader):
        super().__init__()
        self.raw_queue = raw_queue
        self.processed_queue = processed_queue
        self.stop_event = stop_event
        self.yolo_model = yolo_model
        self.easyocr_reader = easyocr_reader
        self.daemon = True # Set thread sebagai daemon

    def run(self):
        print("[INFO] Thread FrameProcessor dimulai...")
        while not self.stop_event.is_set() or not self.raw_queue.empty():
            try:
                frame = self.raw_queue.get(timeout=1) # Tunggu frame hingga 1 detik
            except queue.Empty:
                if self.stop_event.is_set():
                    break # Keluar jika sudah disinyal berhenti dan queue kosong
                continue

            # --- Main Video Pre-processing (Optimalisasi 1: Normalisasi Ukuran Frame) ---
            # Mengurangi resolusi frame sebelum deteksi untuk mengurangi beban komputasi.
            # Sesuaikan 'new_width' sesuai kebutuhan performa dan kualitas Anda.
            new_width = 960 # Contoh: Resize ke lebar 960px (dari default 640px YOLOv8)
            new_height = int(frame.shape[0] * (new_width / frame.shape[1]))
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Example: Noise Reduction on the entire frame (opsional)
            # frame = cv2.GaussianBlur(frame, (3, 3), 0) # Smaller kernel for object detection

            # Grayscale Conversion (NOT recommended for YOLOv8 input, as it's trained with RGB)
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frame_for_yolo = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR) # Convert back to 3 channels if YOLO needs it

            # --- End of Main Video Pre-processing ---

            # Deteksi kendaraan menggunakan YOLOv8.
            # Ambang batas kepercayaan dinaikkan sedikit untuk mengurangi false positive.
            detected_vehicles = detect_vehicles(frame, self.yolo_model, confidence_threshold=0.6) # Ambang batas dinaikkan

            # Deteksi dan kenali plat nomor
            detected_plates = detect_and_recognize_license_plates(frame, detected_vehicles, self.easyocr_reader)

            # Task 6: Validasi dan Format Keluaran - Implementasi di Processor Thread
            for plate_info in detected_plates:
                plate_text = plate_info['text']
                
                # Task 6.2: Basis Data - Bandingkan hasil dengan database plat nomor.
                is_registered = check_database(plate_text)
                status = "TERDAFTAR" if is_registered else "TIDAK TERDAFTAR"
                
                # Task 6.3: Penyimpanan/Tampilan - Simpan ke log.
                log_plate_detection(plate_text, status)

                # Visualisasi hasil (tetap di sini agar data log sinkron dengan tampilan)
                (x_plate, y_plate, w_plate, h_plate) = plate_info['plate_box']
                plate_conf = plate_info['confidence']
                
                # Ubah warna bounding box dan teks berdasarkan status registrasi
                color = (0, 255, 0) if is_registered else (0, 0, 255) # Hijau jika terdaftar, Merah jika tidak
                cv2.rectangle(frame, (x_plate, y_plate), (x_plate + w_plate, y_plate + h_plate), color, 2)
                cv2.putText(frame, f"Plat: {plate_text} ({status})",
                            (x_plate, y_plate - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"Conf: {plate_conf:.2f}",
                            (x_plate, y_plate - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Visualisasi kendaraan (tetap di sini)
            for vehicle in detected_vehicles:
                (x, y, w, h) = vehicle['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{vehicle['label']}: {vehicle['confidence']:.2f}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Coba masukkan frame yang sudah diproses ke queue, jika penuh, lewati
            try:
                self.processed_queue.put(frame, block=False)
            except queue.Full:
                pass # Queue penuh, lewati frame ini

        print("[INFO] Thread FrameProcessor dihentikan.")


# --- Real-time Main Loop (Display Thread) ---

def main():
    print("[INFO] Memulai streaming video...")
    
    # Video Source: Menggunakan file video MP4.
    # Ganti "videos/Licence_Plate_Car.mp4" dengan jalur relatif atau absolut ke file video Anda.
    # Pastikan menggunakan forward slashes (/) atau double backslashes (\\) untuk path di Windows.
    video_path = "videos/Licence_Plate_Car.mp4" # <--- GANTI INI DENGAN JALUR FILE VIDEO ANDA

    # Inisialisasi dan mulai thread pembaca frame
    reader_thread = FrameReader(video_path, raw_frame_queue, stop_event)
    reader_thread.start()

    # Inisialisasi dan mulai thread pemroses frame
    processor_thread = FrameProcessor(raw_frame_queue, processed_frame_queue, stop_event, yolo_model, reader)
    processor_thread.start()

    fps_start_time = time.time()
    fps_frame_count = 0

    try:
        while not stop_event.is_set():
            # Ambil frame yang sudah diproses dari queue
            try:
                frame_to_display = processed_frame_queue.get(timeout=1) # Tunggu frame hingga 1 detik
            except queue.Empty:
                if stop_event.is_set():
                    break # Keluar jika sudah disinyal berhenti dan queue kosong
                continue # Lanjutkan menunggu frame

            # Task 6.3: Penyimpanan/Tampilan - Tampilkan hasil real-time (FPS)
            # Calculate and display FPS
            fps_frame_count += 1
            if (time.time() - fps_start_time) >= 1.0:
                fps = fps_frame_count / (time.time() - fps_start_time)
                cv2.putText(frame_to_display, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                fps_frame_count = 0
                fps_start_time = time.time()

            # Task 6.3: Penyimpanan/Tampilan - Tampilkan hasil real-time (Video Output)
            # Display frame
            cv2.imshow("Real-time ANPR (YOLOv8)", frame_to_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_event.set() # Set event untuk menghentikan semua thread
                break

    except Exception as e:
        print(f"[ERROR] An error occurred in main display loop: {e}")
    finally:
        print("[INFO] Menghentikan streaming video dan membersihkan...")
        stop_event.set() # Pastikan event berhenti diset jika ada error atau keluar
        reader_thread.join() # Tunggu thread reader selesai
        processor_thread.join() # Tunggu thread processor selesai
        cv2.destroyAllWindows()
        print("[INFO] Program selesai.")

if __name__ == "__main__":
    # OPTIMASI 2: Optimasi Model (Inferensi)
    # Setelah model YOLOv8 dilatih atau diunduh, Anda bisa mengoptimalkannya untuk inferensi lebih cepat:
    # - Konversi ke format ONNX atau OpenVINO: Ultralytics mendukung ekspor model ke format ini.
    # - NVIDIA TensorRT: Untuk GPU NVIDIA, konversi model ke format TensorRT dapat memberikan
    #   peningkatan kecepatan inferensi yang signifikan. Ini adalah framework optimasi inferensi.

    # OPTIMASI 6: Sistem Pelacakan Objek (Object Tracking)
    # Implementasi pelacakan objek (misalnya DeepSORT, ByteTrack) akan sangat meningkatkan konsistensi
    # deteksi dan pengenalan plat nomor di seluruh frame. Ini memungkinkan:
    # - Pemberian ID unik untuk setiap kendaraan/plat nomor yang dilacak.
    # - Pengumpulan hasil OCR dari beberapa frame untuk satu plat nomor, lalu memilih hasil terbaik
    #   (misalnya, yang paling sering muncul atau dengan confidence tertinggi).
    # - Mengurangi beban komputasi karena deteksi penuh tidak perlu dilakukan di setiap frame
    #   untuk objek yang sudah dilacak.
    # Implementasi ini membutuhkan pustaka tambahan dan logika yang kompleks, sehingga tidak disertakan
    # langsung dalam kode ini, namun sangat direkomendasikan untuk sistem produksi.

    # OPTIMASI 8: Pelatihan Model YOLO Kustom untuk Plat Nomor
    # Untuk akurasi deteksi lokasi plat nomor yang lebih tinggi, Anda dapat melatih
    # model YOLOv8 kedua (lebih kecil) khusus untuk mendeteksi plat nomor.
    # Alurnya akan menjadi:
    # 1. YOLOv8 umum (deteksi kendaraan) -> menghasilkan bounding box kendaraan.
    # 2. YOLOv8 kustom (deteksi plat nomor) -> mencari plat nomor di dalam bounding box kendaraan.
    # 3. EasyOCR (atau OCR kustom) -> mengenali teks dari plat nomor yang terpotong.
    # Ini membutuhkan dataset plat nomor yang diberi anotasi dan proses pelatihan yang signifikan.

    # OPTIMASI 9 (Tambahan): Database Plat Nomor
    # Jika Anda memiliki daftar plat nomor yang terdaftar, Anda dapat mengintegrasikan
    # hasil OCR dengan database tersebut untuk verifikasi, pencatatan waktu, atau tujuan lainnya.
    # Ini membutuhkan koneksi ke database (misalnya SQLite, PostgreSQL, MySQL) dan logika query.

    main()
