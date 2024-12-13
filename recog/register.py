#register.py
from flask import Blueprint, request
import os
import time
import cv2
import numpy as np
import pickle
import face_recognition
import logging
from threading import Lock
from datetime import datetime

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    handlers=[
        logging.FileHandler('face_registration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

register_bp = Blueprint('register_bp', __name__)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))# 절대 경로 BASE_DIR 설정
BASE_DIR = os.path.join(SCRIPT_DIR, 'database_faces')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
ENCODING_FILE = os.path.join(SCRIPT_DIR, 'encodings.pickle')
encoding_lock = Lock()

if not os.path.exists(BASE_DIR):
    logger.info(f"Creating directory: {BASE_DIR}")
    os.makedirs(BASE_DIR)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    try:
        logger.debug("Starting image preprocessing")
        new_width = int(image.shape[1] * 3)
        new_height = int(image.shape[0] * 3)
        upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        gamma = 0.8
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        gamma_corrected = cv2.LUT(upscaled, table)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        if len(gamma_corrected.shape) == 3:
            channels = cv2.split(gamma_corrected)
            clahe_channels = [clahe.apply(channel) for channel in channels]
            clahe_img = cv2.merge(clahe_channels)
        else:
            clahe_img = clahe.apply(gamma_corrected)

        kernel_size = (3, 3)
        sigma = 1.0
        amount = 1.0
        blurred = cv2.GaussianBlur(clahe_img, kernel_size, sigma)
        sharpened = cv2.addWeighted(clahe_img, 1 + amount, blurred, -amount, 0)

        logger.debug("Image preprocessing completed successfully")
        return sharpened
    except Exception as e:
        logger.error(f"Error in image preprocessing: {str(e)}")
        raise

def load_encodings():
    try:
        if not os.path.exists(ENCODING_FILE):
            logger.info(f"No existing encoding file found at {ENCODING_FILE}")
            return {'encodings': [], 'names': []}

        logger.debug(f"Loading encodings from {ENCODING_FILE}")
        with open(ENCODING_FILE, 'rb') as f:
            data = pickle.load(f)
            logger.info(f"Loaded {len(data['encodings'])} existing encodings")
            return data
    except Exception as e:
        logger.error(f"Error loading encodings: {str(e)}")
        raise

def save_encodings(encoding_data):
    try:
        logger.debug(f"Saving encodings to {ENCODING_FILE}")
        with open(ENCODING_FILE, 'wb') as f:
            pickle.dump(encoding_data, f)
        logger.info(f"Successfully saved {len(encoding_data['encodings'])} encodings")
    except Exception as e:
        logger.error(f"Error saving encodings: {str(e)}")
        raise

def encode_face(image_path, user_name):
    try:
        logger.info(f"Starting face encoding for user: {user_name}")
        logger.debug(f"Processing image: {image_path}")

        image = face_recognition.load_image_file(image_path)
        logger.debug(f"Image shape: {image.shape}")

        preprocessed = preprocess_image(image)
        rgb_image = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)

        logger.debug("Detecting face locations")
        face_locations = face_recognition.face_locations(rgb_image, model='hog')
        logger.debug(f"Found {len(face_locations)} faces")

        if not face_locations:
            logger.error("No face detected in the image")
            raise ValueError("No face detected in the image")

        logger.debug("Generating face encodings")
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        if not face_encodings:
            logger.error("Failed to encode face")
            raise ValueError("Failed to encode face")

        logger.info("Face encoding completed successfully")
        return face_encodings[0]
    except Exception as e:
        logger.error(f"Error in face encoding: {str(e)}")
        raise

@register_bp.route('/register', methods=['POST'])
def register_face():
    try:
        logger.info("Starting face registration process")

        user_id = request.form.get('id')
        user_name = request.form.get('name')
        if not user_id or not user_name:
            logger.error("No user_id or user_name provided")
            return "사용자 ID 또는 이름이 없습니다.", 400

        logger.debug(f"Processing registration for user: {user_id}, {user_name}")

        
        files = request.files.getlist('file')# 다중 파일
        if not files:
            logger.error("No file part in request")
            return "파일이 없습니다.", 400

        folder_name = f"{user_id}_{user_name}"
        user_folder = os.path.join(BASE_DIR, folder_name)
        if not os.path.exists(user_folder):
            logger.debug(f"Creating user directory: {user_folder}")
            os.makedirs(user_folder)

        existing_files = [f for f in os.listdir(user_folder) if allowed_file(f)]
        
        
        collected_encodings = []# 여러 이미지 인코딩을 담을 리스트

        for file in files:
            logger.debug(f"Received file: {file.filename}")
            if not allowed_file(file.filename):
                logger.error(f"Invalid file type: {file.filename}")
                return "허용되지 않는 파일 형식입니다. jpg, jpeg, png 형식만 지원됩니다.", 400

            next_index = len(existing_files) + 1
            file_extension = file.filename.rsplit('.', 1)[1].lower()
            file_name = f"{user_name}{next_index}.{file_extension}"
            file_path = os.path.join(user_folder, file_name)

            logger.debug(f"Attempting to save file at: {file_path}")
            file.save(file_path)
            logger.debug(f"File saved successfully at: {file_path}")

            try:
                logger.info("Starting face encoding process")
                face_encoding = encode_face(file_path, user_name)
                collected_encodings.append(face_encoding)
                existing_files.append(file_name)
            except Exception as e:
                logger.error(f"Face encoding failed: {str(e)}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                return f"Face encoding failed: {str(e)}", 400

        # 평균 인코딩 계산
        if len(collected_encodings) == 0:
            logger.error("No valid face encodings found")
            return "인코딩된 얼굴이 없습니다.", 400

        import numpy as np
        
        avg_encoding = np.mean(collected_encodings, axis=0)

        logger.debug("Updating encodings file with averaged encoding")
        with encoding_lock:
            encoding_data = load_encodings()
            # 평균 인코딩 하나만 추가
            encoding_data['encodings'].append(avg_encoding)
            encoding_data['names'].append(f"{user_id}_{user_name}")
            save_encodings(encoding_data)

        logger.info(f"Successfully registered user: {user_id}, {user_name} with averaged encoding from {len(collected_encodings)} images")
        return f"Successfully registered profile for {user_id}, {user_name} with averaged encoding from {len(collected_encodings)} images", 200

    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        return f"Registration failed: {str(e)}", 500
