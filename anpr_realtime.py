import cv2
import numpy as np
import time
import threading 
import queue
from imutils.video import VideoStream
from ultralytics import YOLO 
import easyocr 
import os
import re 
import torch 
import datetime 
import csv        

# --- Konfigurasi dan Pemuatan Model ---

PATH_TO_YOLOV8_MODEL = 'yolov8n.pt' # Model YOLOv8 pre-trained untuk deteksi kendaraan

print("[INFO] Memuat model EasyOCR (mungkin butuh waktu pertama kali)...")
# Inisialisasi EasyOCR reader. Menggunakan GPU jika tersedia.
# 'en' untuk karakter Latin pada plat nomor.
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available()) 
print("[INFO] EasyOCR model loaded successfully.")

print(f"[INFO] Memuat model deteksi objek YOLOv8 dari {PATH_TO_YOLOV8_MODEL}...")
yolo_model = YOLO(PATH_TO_YOLOV8_MODEL)

# Memastikan pemanfaatan GPU maksimal untuk YOLOv8 jika tersedia.
if torch.cuda.is_available():
    yolo_model.to('cuda')
    print("[INFO] YOLOv8 model dipindahkan ke GPU.")
else:
    print("[INFO] GPU tidak tersedia, YOLOv8 akan berjalan di CPU. Performa mungkin lebih lambat.")
print("[INFO] YOLOv8 model berhasil dimuat.")

# Definisikan kelas objek yang akan dideteksi sebagai 'kendaraan' dari dataset COCO.
# ID kelas ini adalah standar untuk dataset COCO:
# 2: car (mobil), 3: motorcycle (motor), 5: bus, 7: truck (truk)
VEHICLE_CLASSES_IDS = [2, 3, 5, 7]

# --- Post-processing dan Validasi Plat Nomor yang Lebih Kuat ---
# Regex untuk format plat nomor Indonesia.
# INDONESIAN_PLATE_REGEX: Lebih akurat untuk plat Indonesia standar (1 atau 2 huruf di awal).
# ^[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{0,3}$
#   - ^[A-Z]{1,2}: 1 atau 2 huruf di awal (contoh: B, AB)
#   - \s?: Spasi opsional
#   - \d{1,4}: 1 sampai 4 digit angka
#   - \s?: Spasi opsional
#   - [A-Z]{0,3}$: 0 sampai 3 huruf di akhir (contoh: ABC, A, atau kosong)
INDONESIAN_PLATE_REGEX = r"^[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{0,3}$"

# INDONESIAN_PLATE_REGEX_FLEXIBLE: Regex yang lebih fleksibel untuk mengantisipasi OCR yang 
# sering menghilangkan spasi atau salah mengenali jumlah huruf awal/akhir karena noise.
# Memungkinkan 1 hingga 3 huruf di awal dan tanpa spasi.
INDONESIAN_PLATE_REGEX_FLEXIBLE = r"^[A-Z]{1,3}\d{1,4}[A-Z]{0,3}$" 

# Set True untuk mengaktifkan validasi regex. Disarankan True untuk produksi.
# Untuk debugging awal jika CSV kosong, set ke False untuk melihat semua deteksi teks.
ENABLE_PLATE_REGEX_VALIDATION = True 

# Mapping untuk koreksi kesalahan OCR umum.
# OCR sering salah mengenali karakter yang mirip visualnya (misal O vs 0).
# Tambahkan mapping berdasarkan kesalahan yang sering Anda temukan pada hasil OCR.
OCR_CORRECTION_MAP = {
    'O': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'Z': '2',
    'G': '6', 'Q': '0', 'D': '0', 'A': '4', 'J': '1', 'U': '0', 'V': 'Y', 'K': 'X',
    'F': 'E', 'P': 'R', 
    ' ': '', # Opsional: Hapus spasi yang mungkin dikenali OCR di tengah plat setelah pembersihan alfanumerik.
             # Ini akan diterapkan di `cleaned_text = cleaned_text.replace(original, corrected)`.
}

# --- Validasi dan Format Keluaran - Basis Data (Simulasi) ---
# Simulasi database plat nomor terdaftar.
# Dalam aplikasi nyata, ini akan terhubung ke database eksternal (misal: MySQL, PostgreSQL).
REGISTERED_PLATES = ["BDB4668", "BOK5EI", "B98ED2", "B45261D", "A8245S", "AR606L", "AE670S", "AE67OS", "APHI88", "A3K961", "A968B6", "AV619Q", "AV619O"]
LOG_FILE_PATH = "anpr_log.txt" # Path untuk file log tekstual
CSV_LOG_FILE_PATH = "anpr_detections.csv" # Path untuk file log CSV

# --- Konfigurasi untuk Penyimpanan Gambar (Output) ---
SAVE_IMAGE_ROOT_DIR = "anpr_output" 
FULL_FRAMES_WITH_DETECTIONS_DIR = os.path.join(SAVE_IMAGE_ROOT_DIR, "full_frames_with_detections")
CROPPED_VEHICLE_IMAGES_DIR = os.path.join(SAVE_IMAGE_ROOT_DIR, "cropped_vehicle_images")
PROCESSED_LICENSE_PLATE_IMAGES_DIR = os.path.join(SAVE_IMAGE_ROOT_DIR, "processed_license_plate_images")
DEBUG_PLATE_IMAGES_DIR = os.path.join(SAVE_IMAGE_ROOT_DIR, "debug_plate_images") # Direktori untuk debug gambar pra-pemrosesan plat

# Pastikan semua direktori output ada. Buat jika belum ada.
os.makedirs(FULL_FRAMES_WITH_DETECTIONS_DIR, exist_ok=True)
os.makedirs(CROPPED_VEHICLE_IMAGES_DIR, exist_ok=True)
os.makedirs(PROCESSED_LICENSE_PLATE_IMAGES_DIR, exist_ok=True)
os.makedirs(DEBUG_PLATE_IMAGES_DIR, exist_ok=True) 
print(f"[INFO] Direktori output dibuat: '{SAVE_IMAGE_ROOT_DIR}' dengan sub-direktori.")

# Fungsi untuk menulis header CSV jika file belum ada.
def write_csv_header():
    if not os.path.exists(CSV_LOG_FILE_PATH):
        with open(CSV_LOG_FILE_PATH, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Plate_Text', 'Status', 'Confidence', 'Full_Frame_Path', 'Cropped_Vehicle_Path', 'Processed_Plate_Path'])
        print(f"[INFO] Header CSV ditulis ke '{CSV_LOG_FILE_PATH}'")

write_csv_header()

# --- Fungsi Deteksi Kendaraan dengan YOLOv8 ---
def detect_vehicles(frame, model, confidence_threshold=0.5):
    """
    Mendeteksi kendaraan dalam sebuah frame menggunakan YOLOv8.
    Args:
        frame (np.array): Gambar frame input (BGR).
        model (YOLO model): Model YOLOv8 yang sudah dimuat.
        confidence_threshold (float): Ambang batas kepercayaan untuk deteksi objek.
    Returns:
        list: Daftar kamus yang berisi info kendaraan terdeteksi 
              (box, label, confidence, cropped_image, cropped_image_path).
    """
    results = model(frame, conf=confidence_threshold, verbose=False)

    detected_vehicles = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0]) 
            conf = float(box.conf[0]) 

            if cls in VEHICLE_CLASSES_IDS: 
                x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                w = x2 - x1
                h = y2 - y1
                label = model.names[cls] 

                cropped_vehicle_image = None
                cropped_vehicle_file_path = "" # Inisialisasi path
                x1_crop = max(0, x1)
                y1_crop = max(0, y1)
                x2_crop = min(frame.shape[1], x2)
                y2_crop = min(frame.shape[0], y2)
                
                if x2_crop > x1_crop and y2_crop > y1_crop:
                    cropped_vehicle_image = frame[y1_crop:y2_crop, x1_crop:x2_crop].copy()
                    ts_vehicle = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    # Nama file unik per kendaraan per frame dengan koordinat awal
                    cropped_vehicle_file_path = os.path.join(CROPPED_VEHICLE_IMAGES_DIR, f"vehicle_{label}_{ts_vehicle}_{x1_crop}-{y1_crop}.jpg")
                    cv2.imwrite(cropped_vehicle_file_path, cropped_vehicle_image)
                    # print(f"[INFO] Gambar kendaraan di-crop disimpan: {cropped_vehicle_file_path}") # Matikan untuk mengurangi log spam
                
                detected_vehicles.append({
                    'box': (x1, y1, w, h), 
                    'label': label,
                    'confidence': conf,
                    'cropped_image': cropped_vehicle_image,
                    'cropped_image_path': cropped_vehicle_file_path # Menyimpan path gambar yang di-crop
                })
    return detected_vehicles

# --- Normalisasi dan Segmentasi Karakter ---
def deskew_plate(image):
    """
    Melakukan koreksi kemiringan (deskewing) pada gambar plat nomor.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def detect_and_recognize_license_plates(full_frame, detected_vehicles_info, reader):
    """
    Mendeteksi dan mengenali plat nomor dari gambar kendaraan yang sudah dipotong.
    Menggunakan EasyOCR untuk deteksi dan pengenalan teks, serta pra-pemrosesan gambar.
    """
    detected_plates_info_list = []

    for vehicle_info in detected_vehicles_info:
        cropped_vehicle_image = vehicle_info['cropped_image']
        vehicle_offset_x = vehicle_info['box'][0]
        vehicle_offset_y = vehicle_info['box'][1]

        if cropped_vehicle_image is None or cropped_vehicle_image.shape[0] == 0 or cropped_vehicle_image.shape[1] == 0:
            continue

        # Persempit area pencarian plat nomor ke bagian bawah kendaraan yang sudah di-crop.
        # Proporsi 0.5 (setengah bagian bawah) adalah heuristik awal.
        # Anda bisa menyesuaikan `plate_search_y_relative` atau `plate_search_h_relative`
        # jika plat nomor di video Anda cenderung berada di posisi lain (misal lebih tinggi, lebih rendah).
        plate_search_x_relative = 0
        plate_search_y_relative = int(cropped_vehicle_image.shape[0] * 0.5) 
        plate_search_w_relative = cropped_vehicle_image.shape[1]
        plate_search_h_relative = int(cropped_vehicle_image.shape[0] * 0.5) 

        if plate_search_w_relative <= 0 or plate_search_h_relative <= 0:
            continue

        area_img_for_plate = cropped_vehicle_image[plate_search_y_relative : plate_search_y_relative + plate_search_h_relative,
                                                     plate_search_x_relative : plate_search_x_relative + plate_search_w_relative]

        if area_img_for_plate.shape[0] == 0 or area_img_for_plate.shape[1] == 0:
            continue

        processed_area_original = area_img_for_plate.copy() 

        processed_area_deskewed = deskew_plate(processed_area_original)
        processed_area_grayscale = cv2.cvtColor(processed_area_deskewed, cv2.COLOR_BGR2GRAY)
        
        # PENYESUAIAN CLAHE: Tingkatkan clipLimit untuk kontras lebih kuat di kondisi malam atau pencahayaan tidak merata.
        # tileGridSize (8,8) adalah umum.
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)) # DISESUAIKAN: clipLimit dari 2.0 ke 4.0
        processed_area_clahe = clahe.apply(processed_area_grayscale)

        # PENYESUAIAN GAUSSIAN BLUR: Kurangi kernel untuk mempertahankan detail karakter kecil pada plat.
        # (3,3) adalah kompromi yang baik antara pengurangan noise dan detail.
        processed_area_blurred = cv2.GaussianBlur(processed_area_clahe, (3, 3), 0) # DISESUAIKAN: kernel dari (5,5) ke (3,3)

        # PENYESUAIAN KRITIS UNTUK ADAPTIVE THRESHOLDING
        # Diganti ke ADAPTIVE_THRESH_MEAN_C karena lebih robust untuk variasi pencahayaan ekstrem (glare/bayangan).
        # blockSize (15) menentukan area di sekitar piksel untuk menghitung ambang batas lokal. Harus ganjil.
        # C (5) adalah konstanta yang dikurangkan dari rata-rata. Nilai positif membantu memisahkan teks terang dari latar belakang bervariasi.
        processed_area_thresholded = cv2.adaptiveThreshold(processed_area_blurred, 255, 
                                                           cv2.ADAPTIVE_THRESH_MEAN_C, # DISESUAIKAN: dari GAUSSIAN_C ke MEAN_C
                                                           cv2.THRESH_BINARY_INV, # THRESH_BINARY_INV: teks putih di latar belakang hitam
                                                           15, # DISESUAIKAN: blockSize dari 11 ke 15
                                                           5)  # DISESUAIKAN: C dari -1 ke 5
        
        # Operasi Morfologi (Erosi/Dilasi): Digunakan untuk membersihkan noise kecil atau menyatukan fragmen karakter.
        # Kernel (2,2) lebih halus dan cocok untuk karakter yang lebih kecil.
        kernel = np.ones((2,2),np.uint8) # DISESUAIKAN: kernel dari (3,3) ke (2,2)
        processed_area_morphed = cv2.erode(processed_area_thresholded, kernel, iterations = 1)
        processed_area_morphed = cv2.dilate(processed_area_morphed, kernel, iterations = 1)

        # Melakukan OCR pada gambar plat nomor yang sudah diproses.
        results = reader.readtext(processed_area_morphed, detail=1)

        # DEBUGGING: Simpan semua gambar pra-pemrosesan jika EasyOCR mendeteksi teks (bahkan teks sampah).
        # Ini sangat penting untuk melihat apakah tuning pra-pemrosesan berhasil membuat teks terbaca visual.
        if results: 
            timestamp_debug = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            cv2.imwrite(os.path.join(DEBUG_PLATE_IMAGES_DIR, f"debug_original_{timestamp_debug}.jpg"), processed_area_original)
            cv2.imwrite(os.path.join(DEBUG_PLATE_IMAGES_DIR, f"debug_deskewed_{timestamp_debug}.jpg"), processed_area_deskewed)
            cv2.imwrite(os.path.join(DEBUG_PLATE_IMAGES_DIR, f"debug_grayscale_{timestamp_debug}.jpg"), processed_area_grayscale)
            cv2.imwrite(os.path.join(DEBUG_PLATE_IMAGES_DIR, f"debug_clahe_{timestamp_debug}.jpg"), processed_area_clahe)
            cv2.imwrite(os.path.join(DEBUG_PLATE_IMAGES_DIR, f"debug_blurred_{timestamp_debug}.jpg"), processed_area_blurred)
            cv2.imwrite(os.path.join(DEBUG_PLATE_IMAGES_DIR, f"debug_thresholded_{timestamp_debug}.jpg"), processed_area_thresholded)
            cv2.imwrite(os.path.join(DEBUG_PLATE_IMAGES_DIR, f"debug_morphed_{timestamp_debug}.jpg"), processed_area_morphed) 
            print(f"[DEBUG] Gambar plat nomor diproses untuk debug disimpan (detected text): {os.path.join(DEBUG_PLATE_IMAGES_DIR, f'debug_morphed_{timestamp_debug}.jpg')}")

        for (bbox, text, prob) in results:
            # Membersihkan teks: hanya karakter alfanumerik, lalu diubah ke huruf besar.
            cleaned_text = "".join(filter(str.isalnum, text)).upper()
            
            # Terapkan koreksi kesalahan OCR (misal O->0, I->1).
            for original, corrected in OCR_CORRECTION_MAP.items():
                cleaned_text = cleaned_text.replace(original, corrected)

            is_valid_format = False
            if ENABLE_PLATE_REGEX_VALIDATION:
                # Coba cocokkan dengan regex ketat terlebih dahulu, lalu yang fleksibel.
                if re.match(INDONESIAN_PLATE_REGEX, cleaned_text):
                    is_valid_format = True
                elif re.match(INDONESIAN_PLATE_REGEX_FLEXIBLE, cleaned_text):
                    is_valid_format = True
            else: # Jika validasi regex dinonaktifkan, semua format dianggap valid.
                is_valid_format = True 
            
            # Ambang batas kepercayaan EasyOCR.
            # Diturunkan menjadi 0.05 untuk debugging awal agar melihat lebih banyak deteksi (termasuk yang kurang yakin).
            # Setelah pra-pemrosesan stabil, naikkan nilai ini (misal 0.5 atau 0.7) untuk mengurangi false positives.
            if is_valid_format and prob > 0.05: # DISESUAIKAN: prob dari 0.5 ke 0.05 untuk debugging awal
                (top_left, _, bottom_right, _) = bbox
                x_plate_relative_to_vehicle = int(top_left[0])
                y_plate_relative_to_vehicle = int(top_left[1])
                w_plate = int(bottom_right[0] - top_left[0])
                h_plate = int(bottom_right[1] - top_left[1])

                # Transformasi koordinat plat nomor ke koordinat frame asli.
                x_plate_on_full_frame = vehicle_offset_x + plate_search_x_relative + x_plate_relative_to_vehicle
                y_plate_on_full_frame = vehicle_offset_y + plate_search_y_relative + y_plate_relative_to_vehicle

                detected_plates_info_list.append({
                    'plate_box': (x_plate_on_full_frame, y_plate_on_full_frame, w_plate, h_plate),
                    'text': cleaned_text,
                    'confidence': prob,
                    'processed_plate_image': processed_area_morphed, 
                    'cropped_vehicle_info': vehicle_info # Menyimpan info kendaraan terkait
                })
    return detected_plates_info_list

# --- Validasi dan Format Keluaran - Fungsi Pembantu ---
def log_plate_detection(plate_text, status):
    """Mencatat deteksi plat nomor ke file log tekstual dan konsol."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Plat: {plate_text}, Status: {status}\n"
    with open(LOG_FILE_PATH, "a") as f:
        f.write(log_entry)
    print(f"[LOG] {log_entry.strip()}")

def write_to_csv(timestamp, plate_text, status, confidence, full_frame_path=None, cropped_vehicle_path=None, processed_plate_path=None):
    """Menulis data deteksi plat nomor ke file CSV."""
    with open(CSV_LOG_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, plate_text, status, confidence, full_frame_path, cropped_vehicle_path, processed_plate_path])

def check_database(plate_text):
    """Simulasi pengecekan database plat nomor terdaftar."""
    return plate_text in REGISTERED_PLATES

# --- Arsitektur Multi-threading (OPTIMASI 7) ---
# Queue untuk frame mentah dari pembaca video.
raw_frame_queue = queue.Queue(maxsize=5)
# Queue untuk frame yang sudah diproses.
processed_frame_queue = queue.Queue(maxsize=5)
# Event untuk sinyal berhenti ke thread lain.
stop_event = threading.Event()

class FrameReader(threading.Thread):
    """Thread untuk membaca frame dari sumber video."""
    def __init__(self, video_path, q, stop_e):
        super().__init__()
        self.video_path = video_path
        self.queue = q
        self.stop_event = stop_e
        self.vs = None
        self.daemon = True 

    def run(self):
        print("[INFO] Thread FrameReader dimulai...")
        self.vs = cv2.VideoCapture(self.video_path)
        if not self.vs.isOpened():
            print(f"[ERROR] FrameReader: Tidak dapat membuka file video: {self.video_path}")
            self.stop_event.set() 
            return

        while not self.stop_event.is_set():
            ret, frame = self.vs.read()
            if not ret:
                print("[INFO] FrameReader: Akhir stream video atau masalah membaca frame.")
                self.stop_event.set() 
                break
            
            try: # Coba masukkan frame ke queue. Jika penuh, lewati.
                self.queue.put(frame, block=False)
            except queue.Full:
                pass 

        self.vs.release() 
        print("[INFO] Thread FrameReader dihentikan.")

class FrameProcessor(threading.Thread):
    """Thread untuk memproses frame (deteksi YOLO & OCR) dan mengelola output."""
    def __init__(self, raw_q, processed_q, stop_e, yolo_m, easyocr_r):
        super().__init__()
        self.raw_queue = raw_q
        self.processed_queue = processed_q
        self.stop_event = stop_e
        self.yolo_model = yolo_m
        self.easyocr_reader = easyocr_r
        self.daemon = True 

    def run(self):
        print("[INFO] Thread FrameProcessor dimulai...")
        while not self.stop_event.is_set() or not self.raw_queue.empty():
            try:
                frame = self.raw_queue.get(timeout=1) 
            except queue.Empty:
                if self.stop_event.is_set():
                    break 
                continue

            # Normalisasi ukuran frame untuk pemrosesan yang konsisten.
            new_width = 960 
            new_height = int(frame.shape[0] * (new_width / frame.shape[1]))
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Deteksi kendaraan.
            detected_vehicles = detect_vehicles(frame, self.yolo_model, confidence_threshold=0.5) 

            # Deteksi dan kenali plat nomor.
            detected_plates_from_vehicles = detect_and_recognize_license_plates(frame, detected_vehicles, self.easyocr_reader)
            
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            any_detection_in_frame = False 
            
            current_full_frame_path = None 
            plates_data_for_csv = [] 

            for plate_info in detected_plates_from_vehicles:
                plate_text = plate_info['text']
                plate_conf = plate_info['confidence'] 
                
                is_registered = check_database(plate_text)
                status = "TERDAFTAR" if is_registered else "TIDAK TERDAFTAR"
                
                log_plate_detection(plate_text, status)
                any_detection_in_frame = True 

                (x_plate, y_plate, w_plate, h_plate) = plate_info['plate_box']
                color = (0, 255, 0) if is_registered else (0, 0, 255) 
                cv2.rectangle(frame, (x_plate, y_plate), (x_plate + w_plate, y_plate + h_plate), color, 2)
                cv2.putText(frame, f"Plat: {plate_text} ({status})",
                                    (x_plate, y_plate - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"Conf: {plate_conf:.2f}",
                                    (x_plate, y_plate - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                current_processed_plate_path = ""
                if 'processed_plate_image' in plate_info and plate_info['processed_plate_image'] is not None:
                    current_processed_plate_path = os.path.join(PROCESSED_LICENSE_PLATE_IMAGES_DIR, f"plate_{plate_text}_{timestamp_str}.jpg")
                    cv2.imwrite(current_processed_plate_path, plate_info['processed_plate_image'])
                    print(f"[INFO] Gambar plat nomor diproses disimpan: {current_processed_plate_path}")
                
                plates_data_for_csv.append({
                    'plate_text': plate_text,
                    'status': status,
                    'confidence': plate_conf,
                    'processed_plate_path': current_processed_plate_path,
                    'cropped_vehicle_path': plate_info['cropped_vehicle_info'].get('cropped_image_path', '') 
                })

            # Visualisasi bounding box kendaraan
            for vehicle_idx, vehicle in enumerate(detected_vehicles):
                (x, y, w, h) = vehicle['box']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{vehicle['label']}: {vehicle['confidence']:.2f}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                any_detection_in_frame = True 

            # Simpan frame lengkap jika ada deteksi kendaraan atau plat nomor.
            if any_detection_in_frame:
                current_full_frame_path = os.path.join(FULL_FRAMES_WITH_DETECTIONS_DIR, f"frame_with_detections_{timestamp_str}.jpg")
                cv2.imwrite(current_full_frame_path, frame) 
                print(f"[INFO] Gambar frame dengan deteksi disimpan: {current_full_frame_path}")

            # Tulis setiap deteksi plat nomor ke CSV.
            for plate_data in plates_data_for_csv:
                write_to_csv(timestamp_str, plate_data['plate_text'], 
                             plate_data['status'], plate_data['confidence'],
                             current_full_frame_path, 
                             plate_data['cropped_vehicle_path'], 
                             plate_data['processed_plate_path'])
                             
            # Masukkan frame yang sudah diproses ke queue untuk ditampilkan.
            try:
                self.processed_queue.put(frame, block=False)
            except queue.Full:
                pass 

        print("[INFO] Thread FrameProcessor dihentikan.")

# --- Real-time Main Loop (Display Thread) ---
def main():
    print("[INFO] Memulai streaming video...")
    
    # --- GANTI INI DENGAN JALUR FILE VIDEO ANDA YANG BENAR ---
    # Pilih salah satu video yang Anda unggah:
    video_path = "videos/Licence_Plate_Car.mp4" 
    # Untuk mencoba video kedua:
    # video_path = "videos/License_Plate_2.mp4" 

    # Inisialisasi dan mulai thread pembaca frame.
    reader_thread = FrameReader(video_path, raw_frame_queue, stop_event)
    reader_thread.start()

    # Inisialisasi dan mulai thread pemroses frame.
    processor_thread = FrameProcessor(raw_frame_queue, processed_frame_queue, stop_event, yolo_model, reader)
    processor_thread.start()

    try:
        # Loop utama untuk menampilkan frame yang sudah diproses.
        while not stop_event.is_set():
            try:
                frame_to_display = processed_frame_queue.get(timeout=1) 
            except queue.Empty:
                if stop_event.is_set():
                    break 
                continue

            cv2.imshow("Real-time ANPR (YOLOv8)", frame_to_display)

            # Cek input keyboard. Tekan 'q' untuk keluar.
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_event.set() 
                break

    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan pada loop tampilan utama: {e}")
    finally:
        print("[INFO] Menghentikan streaming video dan membersihkan sumber daya...")
        stop_event.set() 
        reader_thread.join() 
        processor_thread.join() 
        cv2.destroyAllWindows() 
        print("[INFO] Program selesai.")

if __name__ == "__main__":
    main()