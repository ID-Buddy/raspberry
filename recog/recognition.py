# recognition.py

from flask import Blueprint, Response
import cv2
import numpy as np
import math
import face_recognition
from picamera2 import Picamera2
import time
import os
import pickle
from collections import deque
from notification_ws import update_recognition_status
from mtcnn import MTCNN

recognition_bp = Blueprint('recognition_bp', __name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
encoding_file = os.path.join(script_dir, 'encodings.pickle')

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (240, 320)})
picam2.configure(camera_config)
picam2.start()
time.sleep(1)

picam2.set_controls({
    "AeEnable": True,             # 자동 노출 활성화
    "AwbEnable": True,            # 자동 화이트밸런스 활성화
    "AeConstraintMode": 0,        # 자동 노출 제약 모드
    "AeExposureMode": 0,          # 자동 노출 모드
    "Brightness": 0.5,            # 기본 밝기값
    "Contrast": 1.2,              # 약간의 대비 증가
    "Saturation": 1.1,            # 약간의 채도 증가
    "Sharpness": 1.5,            # 선명도
    "AeMeteringMode": 1,          # 중앙 중점 측광 모드
    "NoiseReductionMode": 2,      # 노이즈 감소 강화
})

def load_encodings(encoding_file):
    if not os.path.exists(encoding_file):
        print(f"Encoding file not found: {encoding_file}")
        return [], []
    with open(encoding_file, 'rb') as f:
        data = pickle.load(f)
        return data['encodings'], data['names']

def gamma_correction(image, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = (np.linspace(0, 255, 256) / 255.0)**inv_gamma * 255
    table = np.clip(table, 0, 255).astype("uint8")
    return cv2.LUT(image, table)

def clahe_equalization(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    if len(image.shape) == 2:
        return clahe.apply(image)
    else:
        channels = cv2.split(image)
        clahe_channels = [clahe.apply(ch) for ch in channels]
        return cv2.merge(clahe_channels)

def unsharp_mask(image, kernel_size=(3,3), sigma=1.0, amount=0.7, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def upscale_frame(image, scale=2):
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return upscaled
    
def adjust_camera_settings(frame):
    """조명 상태를 감지하고 카메라 설정을 자동으로 조정"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray)
    brightness_std = np.std(gray)
    
    # 역광 감지 (임계값 조정)
    is_backlit = brightness_std > 40 and average_brightness < 120
    is_dark = average_brightness < 100
    
    if is_backlit:
        # 역광 상황에서의 설정 - 이미지 처리는 최소화하고 카메라 설정에 집중
        picam2.set_controls({
            "AeExposureMode": 1,        # 역광 보정 모드
            "AeMeteringMode": 1,        # 중앙 중점 측광
            "Brightness": 0.6,          # 적당한 밝기
            "Contrast": 1.2,            # 적당한 대비
            "ExposureValue": 1.0,       # 적절한 노출값
            "Sharpness": 1.5,           # 기본 선명도 유지
            "NoiseReductionMode": 1,    # 기본 노이즈 감소
            "AwbEnable": True,          # 자동 화이트밸런스
            "ColourGains": (1.2, 1.2),  # 약간의 색상 게인
            "AnalogueGain": 1.5,        # 적당한 게인
            "AeConstraintMode": 0,      # 기본 노출 제약
        })
        
        # 최소한의 이미지 처리만 적용
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    elif is_dark:
        # 어두운 상황 설정 (기존 유지 - 잘 작동하므로)
        picam2.set_controls({
            "AeExposureMode": 0,         
            "Brightness": 0.85,           
            "Contrast": 1.3,             
            "NoiseReductionMode": 3,     
            "ExposureValue": 2.0,        
            "Sharpness": 1.8,            
            "Saturation": 1.5,           
            "AwbEnable": True,           
            "AnalogueGain": 2.5,         
            "ColourGains": (1.5, 1.5),   
            "AeConstraintMode": 0,        
            "ExposureTime": 66666,       
            "FrameDurationLimits": (33333, 100000)
        })
        
        # 어두운 상황 이미지 처리 
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        frame = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        frame = cv2.filter2D(frame, -1, kernel)

    else:
        # 일반 상황 설정 
        picam2.set_controls({
            "AeExposureMode": 0,    
            "Brightness": 0.4,      
            "Contrast": 0.9,        
            "NoiseReductionMode": 1,
            "ExposureValue": -0.2,  
            "Sharpness": 1.5,       
            "Saturation": 1.1,      
            "AwbEnable": True,      
            "FrameDurationLimits": (33333, 33333)
        })

    return frame
    
    
def preprocess_frame(frame):
    upscaled = upscale_frame(frame, scale=2)
    gamma_corrected = gamma_correction(upscaled, gamma=1.2)
    gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
    clahe_gray = clahe_equalization(gray)
    processed = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2BGR)
    return processed

def preprocess_face_region(face_image):
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    clahe_face = clahe_equalization(gray_face)
    face_bgr = cv2.cvtColor(clahe_face, cv2.COLOR_GRAY2BGR)
    sharpened_face = unsharp_mask(face_bgr, kernel_size=(3,3), sigma=1.0, amount=0.7, threshold=0)
    return sharpened_face

def face_confidence(face_distance, face_match_threshold=0.65):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)
    if face_distance > face_match_threshold:
        return round(linear_val * 100, 2)
    else:
        value = (linear_val + ((1.0 - linear_val)*((linear_val - 0.5)*2)**0.2))*100
        return round(value,2)

detector = MTCNN()

@recognition_bp.route('/video_feed')
def video_feed():
    known_face_encodings, known_face_names = load_encodings(encoding_file)
    if not known_face_encodings:
        return Response("No encodings found", status=500)

    def get_alpha(fcount):
        mod = fcount % 3
        if mod == 0:
            return 1.0
        elif mod == 1:
            return 0.66
        else:
            return 0.33

    def interpolate_faces(old_faces, new_faces, alpha):
        interpolated = []
        min_len = min(len(old_faces), len(new_faces))
        for i in range(min_len):
            old_f = old_faces[i]
            new_f = new_faces[i]
            (l1,t1,r1,b1) = old_f["bbox"]
            (l2,t2,r2,b2) = new_f["bbox"]

            l = int(l1*(1-alpha) + l2*alpha)
            t = int(t1*(1-alpha) + t2*alpha)
            r = int(r1*(1-alpha) + r2*alpha)
            b = int(b1*(1-alpha) + b2*alpha)

            nm = new_f["name"] if (old_f["name"] == new_f["name"] or alpha > 0.5) else old_f["name"]
            cf = old_f["confidence"]*(1-alpha) + new_f["confidence"]*alpha
            interpolated.append({"name": nm, "confidence": cf, "bbox":(l,t,r,b)})

        if len(new_faces) > min_len:
            for nf in new_faces[min_len:]:
                if alpha > 0.5:
                    interpolated.append(nf)

        if len(old_faces) > min_len:
            for of in old_faces[min_len:]:
                if alpha < 0.5:
                    interpolated.append(of)

        return interpolated

    frame_count = 0
    recognition_history = deque(maxlen=5)
    stable_face_min_count = 2
    last_stable_faces = []
    current_detected_faces = []

    RECOGNITION_INTERVAL = 4
    trackers = []
    tracking_active = False

    def setup_trackers(display_frame, faces):
        del trackers[:]
        for f in faces:
            (l,t,r,b) = f["bbox"]
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(display_frame, (l,t,r-l,b-t))
            trackers.append((tracker, f["name"], f["confidence"]))

    def update_trackers(display_frame):
        new_faces = []
        to_remove = []
        for i, (tracker, nm, cf) in enumerate(trackers):
            success, box = tracker.update(display_frame)
            if success:
                x,y,w,h = box
                new_faces.append({"name": nm, "confidence": cf, "bbox":(int(x),int(y),int(x+w),int(y+h))})
            else:
                to_remove.append(i)
        for i in reversed(to_remove):
            trackers.pop(i)
        return new_faces

    def generate_frames():
        nonlocal last_stable_faces, current_detected_faces, frame_count, tracking_active, trackers
        while True:
            frame = picam2.capture_array()
            frame = adjust_camera_settings(frame)  # 이 줄 추가
            frame_count += 1

            preprocessed = preprocess_frame(frame)
            do_recognition = (frame_count % RECOGNITION_INTERVAL == 0)

            display_frame = preprocessed.copy()
            if do_recognition:
                # 얼굴 검출
                rgb_frame = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
                detections = detector.detect_faces(rgb_frame)
                face_locations = []
                for det in detections:
                    x, y, w, h = det['box']
                    top, left = y, x
                    bottom, right = y+h, x+w
                    face_locations.append((top, right, bottom, left))

                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                detected_faces = []
                face_match_threshold = 0.6
                distance_diff_threshold = 0.05
                min_confidence_threshold = 50.0
                min_bbox_size = 30

                for ((top, right, bottom, left), face_encoding) in zip(face_locations, face_encodings):
                    face_w = right - left
                    face_h = bottom - top

                    # bounding box가 너무 작으면 Unknown
                    if face_w < min_bbox_size or face_h < min_bbox_size:
                        detected_faces.append({"name": "Unknown", "confidence": 0, "bbox":(left, top, right, bottom)})
                        continue

                    face_image = preprocessed[top:bottom, left:right]
                    if face_image.size > 0:
                        face_image = preprocess_face_region(face_image)

                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=face_match_threshold)
                    name = "Unknown"
                    confidence = 0
                    recognized_id = ""
                    recognized_name = ""
                    if True in matches:
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        best_distance = face_distances[best_match_index]
                        sorted_distances = np.sort(face_distances)
                        if len(sorted_distances) > 1:
                            second_best_distance = sorted_distances[1]
                        else:
                            second_best_distance = 1.0

                        if best_distance <= face_match_threshold:
                            # 이름 파싱
                            recognized_value = known_face_names[best_match_index]
                            parts = recognized_value.split('_', 1)
                            if len(parts) == 2:
                                recognized_id_str, recognized_name = parts
                            else:
                                recognized_id_str = ""
                                recognized_name = recognized_value

                            try:
                                recognized_id = int(recognized_id_str)
                            except ValueError:
                                recognized_id = recognized_id_str

                            # 두 번째 거리와 차이 확인
                            if (second_best_distance - best_distance) < distance_diff_threshold:
                                # 구분 애매하면 Unknown
                                name = "Unknown"
                            else:
                                # confidence 계산
                                confidence = face_confidence(best_distance, face_match_threshold=face_match_threshold)
                                if confidence < min_confidence_threshold:
                                    name = "Unknown"
                                    confidence = 0
                                else:
                                    name = recognized_name
                                    # 인식 성공 시
                                    print(f"[recognition.py] Recognized: {name} (ID={recognized_id})")
                                    update_recognition_status(success=True, recognized_id=recognized_id, recognized_name=name)
                        else:
                            name = "Unknown"
                    else:
                        # 매칭 없음
                        name = "Unknown"

                    detected_faces.append({"name": name, "confidence": confidence, "bbox":(left, top, right, bottom)})

                recognition_history.append(detected_faces)
                if len(recognition_history) == recognition_history.maxlen:
                    recent_result = recognition_history[-1]
                    stable_count = 1
                    for prev_result in reversed(list(recognition_history)[:-1]):
                        if len(prev_result) == len(recent_result):
                            all_match = True
                            for f1, f2 in zip(prev_result, recent_result):
                                if f1["name"] != f2["name"]:
                                    all_match = False
                                    break
                            if all_match:
                                stable_count += 1
                            else:
                                break
                        else:
                            break
                    if stable_count >= stable_face_min_count:
                        last_stable_faces = recent_result

                current_detected_faces = detected_faces
                alpha = get_alpha(frame_count)
                interpolated_faces = interpolate_faces(last_stable_faces, current_detected_faces, alpha)

                setup_trackers(display_frame, interpolated_faces)
                tracking_active = True

                for f in interpolated_faces:
                    (l,t,r,b) = f["bbox"]
                    nm = f["name"]
                    cf = f["confidence"]
                    cv2.rectangle(display_frame, (l,t), (r,b), (0, 255, 0), 2)
                    cv2.rectangle(display_frame, (l, b - 35), (r, b), (0, 255, 0), -1)
                    cv2.putText(display_frame, f"{nm} ({cf}%)",
                                (l + 6, b - 6),
                                cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

            else:
                if tracking_active and len(trackers) > 0:
                    tracked_faces = update_trackers(display_frame)
                    alpha = get_alpha(frame_count)
                    interpolated_faces = interpolate_faces(last_stable_faces, tracked_faces, alpha)
                    for f in interpolated_faces:
                        (l,t,r,b) = f["bbox"]
                        nm = f["name"]
                        cf = f["confidence"]
                        cv2.rectangle(display_frame, (l,t), (r,b), (0, 255, 0), 2)
                        cv2.rectangle(display_frame, (l, b - 35), (r, b), (0, 255, 0), -1)
                        cv2.putText(display_frame, f"{nm} ({cf}%)",
                                    (l + 6, b - 6),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)
                else:
                    alpha = get_alpha(frame_count)
                    interpolated_faces = interpolate_faces(last_stable_faces, current_detected_faces, alpha)
                    for f in interpolated_faces:
                        (l,t,r,b) = f["bbox"]
                        nm = f["name"]
                        cf = f["confidence"]
                        cv2.rectangle(display_frame, (l,t), (r,b), (0, 255, 0), 2)
                        cv2.rectangle(display_frame, (l, b - 35), (r, b), (0, 255, 0), -1)
                        cv2.putText(display_frame, f"{nm} ({cf}%)",
                                    (l + 6, b - 6),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

            ret, buffer = cv2.imencode('.jpg', display_frame)
            frame_jpg = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_jpg + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
