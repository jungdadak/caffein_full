import cv2
import easyocr
import re
import csv
import torch
import os
import pandas as pd
import random
import numpy as np
import json
from ultralytics import YOLO
from matplotlib import rc
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont


# 번호판 유효성 검사를 위한 정규표현식 컴파일
license_plate_pattern = re.compile(r'[0-9]{2,3}[가-힣0-9]{1}[0-9]{4}')

def draw_text_pil(img, text, position, font_path, font_size=24, color=(255, 255, 255)):
    """
    OpenCV 이미지에 Pillow를 사용하여 텍스트를 그립니다.

    Parameters:
    - img: OpenCV 이미지 (BGR 형식)
    - text: 표시할 텍스트
    - position: 텍스트의 (x, y) 위치
    - font_path: TrueType 폰트 파일의 경로
    - font_size: 폰트 크기
    - color: 텍스트 색상 (RGB 형식)
    """
    # OpenCV 이미지를 RGB로 변환
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 폰트 로드
    font = ImageFont.truetype(font_path, font_size)
    
    # 텍스트 그리기
    draw.text(position, text, font=font, fill=color)
    
    # 다시 OpenCV 이미지로 변환
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img

# 1. 설정 로드 함수 및 설정 값
def load_config():
    """
    config.json 파일을 로드하여 설정을 반환합니다.
    """
    with open("C:/Users/PC/Desktop/caffein/montana/config.json", "r", encoding="utf-8") as f:
        return json.load(f)

_config = load_config()

def get_video_paths():
    return _config['video_paths']

def get_results_folder():
    return _config['results_folder']

def get_corrections():
    return _config['corrections']

def get_enable_profiling():
    return _config.get('enable_profiling', False)

def get_model_paths():
    return _config['model_paths']

# 2. 폰트 설정 함수
def set_malgun_gothic_font():
    """
    Matplotlib의 폰트를 Malgun Gothic으로 설정하여 한글이 깨지지 않도록 합니다.
    """
    font_name = "Malgun Gothic"
    rc('font', family=font_name)
    print(f"Font set to: {font_name}")

# 3. 번호판 교정 함수
def correct_plate_number(plate):
    """
    OCR 결과에서 특정 문자를 교정하여 번호판 인식 정확도를 높입니다.
    """
    corrections = get_corrections()
    if re.fullmatch(r'\d{7}', plate):  # 숫자 7자리
        if plate[-5] in corrections:
            plate = plate[:-5] + corrections[plate[-5]] + plate[-4:]
    elif re.fullmatch(r'\d{8}', plate):  # 숫자 8자리
        if plate[-5] in corrections:
            plate = plate[:-5] + corrections[plate[-5]] + plate[-4:]
    return plate

# 4. IoU 기반 추적 함수
def calculate_iou(box1, box2):
    """
    두 개의 바운딩 박스 간의 IoU(Intersection over Union)를 계산합니다.
    박스는 (x1, y1, x2, y2) 형식의 튜플이어야 합니다.
    """
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0  # Avoid division by zero

    iou = inter_area / union_area
    return iou

def assign_ids_to_boxes(boxes, state, max_frames_missing=20, iou_threshold=0.2):
    """
    감지된 박스와 기존 추적 객체를 IoU를 기반으로 매칭하여 ID를 할당합니다.

    Parameters:
    - boxes: 현재 프레임에서 감지된 박스 목록
    - state: 현재 추적 상태
    - max_frames_missing: 객체가 감지되지 않아도 추적을 유지하는 최대 프레임 수
    - iou_threshold: 매칭을 허용하는 최소 IoU 값
    """
    new_tracked_objects = {}
    unmatched_previous_objects = set(state['tracked_objects'].keys())

    for bbox in boxes:
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])
        confidence = bbox.conf[0]
        cls = bbox.cls[0]

        # 현재 감지된 박스의 중심점 계산
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        best_iou = 0
        best_id = None

        # 기존 추적 객체와 IoU 계산
        for obj_id, obj in state['tracked_objects'].items():
            existing_bbox = obj['bbox']
            iou = calculate_iou((x1, y1, x2, y2), existing_bbox)
            print(f"  Comparing with ID {obj_id}: IoU={iou:.2f}")  # 디버깅 출력
            if iou > best_iou:
                best_iou = iou
                best_id = obj_id

        if best_iou > iou_threshold:
            # 기존 객체에 업데이트
            obj = state['tracked_objects'][best_id]
            new_tracked_objects[best_id] = {
                'bbox': (x1, y1, x2, y2),
                'center': center,
                'color': cls,
                'confidence': confidence,
                'best_ocr': obj['best_ocr'],
                'frames_missing': 0,
                'trajectory': obj['trajectory'] + [center],
                'direction': obj['direction']
            }
            unmatched_previous_objects.discard(best_id)
            print(f"  Matched with ID {best_id} (IoU={best_iou:.2f})")
        else:
            # 새로운 객체로 추가
            new_tracked_objects[state['next_id']] = {
                'bbox': (x1, y1, x2, y2),
                'center': center,
                'color': cls,
                'confidence': confidence,
                'best_ocr': None,
                'frames_missing': 0,
                'trajectory': [center],
                'direction': None
            }
            assign_color_to_id(state, state['next_id'])
            print(f"  Assigned new ID {state['next_id']} (IoU={best_iou:.2f})")
            state['next_id'] += 1

    # 누락된 객체 처리
    for obj_id in unmatched_previous_objects:
        obj = state['tracked_objects'][obj_id]
        if obj['frames_missing'] < max_frames_missing:
            new_tracked_objects[obj_id] = {
                'bbox': obj['bbox'],
                'center': obj['center'],
                'color': obj['color'],
                'confidence': obj['confidence'],
                'best_ocr': obj['best_ocr'],
                'frames_missing': obj['frames_missing'] + 1,
                'trajectory': obj['trajectory'],
                'direction': obj['direction']
            }
            print(f"  ID {obj_id} missing frame count: {obj['frames_missing'] + 1}")

    state['tracked_objects'] = new_tracked_objects
    return state

def assign_color_to_id(state, obj_id):
    """
    새로운 ID에 대해 임의의 색상을 할당합니다.
    """
    if obj_id not in state['id_colors']:
        state['id_colors'][obj_id] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

def create_tracking_state():
    """
    추적 상태를 초기화합니다.
    """
    return {
        'tracked_objects': {},
        'next_id': 0,
        'id_colors': {}
    }

# 5. 기타 유틸리티 함수
def ensure_folder_exists(folder):
    """
    지정된 폴더가 존재하지 않으면 생성합니다.
    """
    os.makedirs(folder, exist_ok=True)

def initialize_csv(csv_filename):
    """
    CSV 파일을 초기화하고 헤더를 작성합니다.
    """
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['video', 'ID', 'color', 'ocr', 'accuracy', 'direction'])

def save_to_csv(best_ocr_results, csv_filename):
    """
    OCR 결과를 CSV 파일에 저장합니다.
    """
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['video', 'ID', 'color', 'ocr', 'accuracy', 'direction'])
        for data in best_ocr_results.values():
            writer.writerow([
                data['video'], data['ID'], data['color'],
                data['ocr'], data['accuracy'], data['direction']
            ])

# 6. 입차/출차 판단 함수
def determine_direction(trajectory):
    """
    객체의 이동 궤적을 분석하여 입차 또는 출차 방향을 판단합니다.

    Parameters:
    - trajectory: 객체의 이동 궤적을 나타내는 중심점 리스트

    Returns:
    - '입차', '출차' 또는 None
    """
    if len(trajectory) >= 2:
        y_positions = [pos[1] for pos in trajectory]
        if y_positions[-1] - y_positions[0] > 50:  # 아래로 이동
            return '입차'
        elif y_positions[0] - y_positions[-1] > 50:  # 위로 이동
            return '출차'
    return None

# 7. 번호판 이미지 전처리 함수
def preprocess_plate_image(img):
    """
    번호판 이미지를 전처리하여 OCR 정확도를 높입니다.

    Parameters:
    - img: 원본 번호판 이미지

    Returns:
    - 전처리된 그레이스케일 이미지
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 노이즈 제거 (Denoising)
    morphed_img = cv2.fastNlMeansDenoising(gray_img, None, 10, 7, 21)
    # 히스토그램 평활화 (Histogram Equalization)
    morphed_img = cv2.equalizeHist(morphed_img)
    return morphed_img

# 8. 모델 로드 함수
def load_model(model_path, device):
    """
    YOLO 모델을 로드합니다.

    Parameters:
    - model_path: 모델 파일의 경로
    - device: 사용할 디바이스 (GPU 또는 CPU)

    Returns:
    - 로드된 YOLO 모델
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        model = YOLO(model_path).to(device)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def process_video(video_path, csv_filename, output_video_path, plate_model, color_model, reader, device):
    """
    비디오를 처리하여 차량 및 번호판을 감지하고, 추적 및 OCR 결과를 저장합니다.

    Parameters:
    - video_path: 처리할 비디오 파일의 경로
    - csv_filename: OCR 결과를 저장할 CSV 파일의 경로
    - output_video_path: 처리된 비디오를 저장할 경로
    - plate_model: 번호판 감지 모델
    - color_model: 차량 감지 모델
    - reader: EasyOCR 리더 객체
    - device: 사용할 디바이스 (GPU 또는 CPU)
    """
    initialize_csv(csv_filename)
    best_ocr_results = {}

    # 추적 상태 초기화
    state = create_tracking_state()

    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # 비디오 작성자 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0.0:
        fps = 25.0  # 기본 FPS 설정
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 폰트 경로 설정 (필요에 따라 수정)
    font_path = "C:\\Windows\\Fonts\\malgun.ttf"
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"폰트 파일을 찾을 수 없습니다: {font_path}")

    while True:
        ret, img = cap.read()
        if not ret:
            break

        # YOLO 모델을 사용하여 번호판 및 차량 감지
        plate_result = plate_model.predict(source=img, imgsz=416, verbose=False)
        color_result = color_model.predict(source=img, imgsz=640, conf=0.3, verbose=False)

        # IoU 기반 ID 할당
        state = assign_ids_to_boxes(
            color_result[0].boxes,
            state,
            max_frames_missing=15,   # 기존 10에서 15로 증가
            iou_threshold=0.25       # 기존 0.3에서 0.25로 감소
        )
        print(f"Frame {frame_count}: {len(state['tracked_objects'])} objects being tracked.")
        for obj_id, obj in state['tracked_objects'].items():
            # 객체의 현재 상태를 출력하여 디버깅
            print(f"  ID: {obj_id}, BBox: {obj['bbox']}, Conf: {obj['confidence']}, IoU: {calculate_iou(obj['bbox'], obj['bbox'])}")

        # 차량 바운딩 박스 그리기 및 정보 표시
        for obj_id, obj in state['tracked_objects'].items():
            x1, y1, x2, y2 = obj['bbox']
            center_x, center_y = obj['center']
            class_name = color_model.names[int(obj['color'])]
            confidence = obj['confidence']

            # 차량 바운딩 박스 및 정보 표시 (ID와 Confidence)
            color = state['id_colors'][obj_id]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f'ID: {obj_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(img, f'Conf: {confidence:.2f}', (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(img, f'Class: {class_name}', (x1, y1 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # 입차/출차 판단
            if obj['direction'] is None:
                obj['direction'] = determine_direction(obj['trajectory'])
                if obj['direction'] is not None:
                    print(f"Object {obj_id} 방향 결정: {obj['direction']}")
                    # 방향 정보를 best_ocr_results에 저장
                    if obj_id in best_ocr_results:
                        best_ocr_results[obj_id]['direction'] = obj['direction']

            # 방향 정보 표시 (수정된 부분)
            if obj['direction'] is not None:
                # 텍스트 위치 설정 (바운딩 박스 내부 상단)
                text_position = (x1, y1 + 20)  # 필요에 따라 조정
                direction_text = f'입차' if obj['direction'] == '입차' else f'출차'

                # Pillow를 사용하여 한글 텍스트 그리기
                img = draw_text_pil(
                    img,
                    direction_text,
                    text_position,
                    font_path=font_path,
                    font_size=24,  # 필요에 따라 조정
                    color=(255, 255, 255)  # 흰색
                )

        # 번호판 바운딩 박스 그리기 및 정보 표시
        for bbox in plate_result[0].boxes:
            px1, py1, px2, py2 = map(int, bbox.xyxy[0])
            plate_confidence = bbox.conf[0]

            # 번호판 바운딩 박스 그리기 (항상 빨간색)
            plate_color = (0, 0, 255)  # 빨간색
            cv2.rectangle(img, (px1, py1), (px2, py2), plate_color, 2)

            # 번호판 중심점 계산
            plate_center_x = (px1 + px2) // 2
            plate_center_y = (py1 + py2) // 2

            # 번호판이 차량 내부에 있는지 확인
            matched_obj_id = None
            for obj_id, obj in state['tracked_objects'].items():
                x1_obj, y1_obj, x2_obj, y2_obj = obj['bbox']
                if x1_obj < plate_center_x < x2_obj and y1_obj < plate_center_y < y2_obj:
                    matched_obj_id = obj_id
                    break

            if matched_obj_id is not None:
                # 번호판 이미지 크롭 및 전처리
                plate_cropped_img = img[py1:py2, px1:px2]
                preprocessed_plate_img = preprocess_plate_image(plate_cropped_img)
                
                # OCR 수행
                ocr_result = reader.readtext(preprocessed_plate_img)
                if ocr_result:
                    _, text, prob = ocr_result[0]
                    text = re.sub('[^가-힣0-9]', '', text)
                    
                    # 정규표현식 패턴과 일치하는지 확인
                    if license_plate_pattern.fullmatch(text):
                        if (matched_obj_id not in best_ocr_results) or (prob > best_ocr_results[matched_obj_id]['accuracy']):
                            best_ocr_results[matched_obj_id] = {
                                'video': video_path,
                                'ID': matched_obj_id,
                                'color': color_model.names[int(state['tracked_objects'][matched_obj_id]['color'])],
                                'ocr': text,
                                'accuracy': prob,
                                'direction': state['tracked_objects'][matched_obj_id]['direction']
                            }
                            state['tracked_objects'][matched_obj_id]['best_ocr'] = (text, prob)

                        # OCR 결과를 프레임에 표시
                        # 위쪽 텍스트: Confidence와 OCR 정확도
                        cv2.putText(img, f'Conf: {plate_confidence:.2f} OCR Acc: {prob:.2f}', 
                                    (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, plate_color, 2)
                        # 아래쪽 텍스트: OCR 결과
                        cv2.putText(img, f'OCR: {text}', 
                                    (px1, py2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, plate_color, 2)
                    else:
                        print(f"OCR 결과 '{text}'가 번호판 패턴과 일치하지 않습니다. 저장되지 않습니다.")

        # 프레임을 동영상으로 저장
        out.write(img)
        frame_count += 1

        # 메모리 관리: GPU 메모리 정리
        torch.cuda.empty_cache()

    cap.release()
    out.release()
    print(f"Finished processing video: {video_path}")

    # CSV 파일에 저장
    save_to_csv(best_ocr_results, csv_filename)


# 9. 메인 함수
def main():
    """
    메인 함수는 설정을 로드하고, 모델을 초기화하며, 각 비디오를 처리합니다.
    """
    video_paths = get_video_paths()
    results_folder = get_results_folder()
    model_paths = get_model_paths()

    # 폰트 설정 (필요한 경우)
    set_malgun_gothic_font()

    # GPU 사용 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 모델 로드 (한 번만 수행)
    plate_model = load_model(model_paths['plate_model'], device)
    color_model = load_model(model_paths['color_model'], device)

    # EasyOCR reader 객체를 GPU를 사용하도록 설정
    reader = easyocr.Reader(['ko'], gpu=torch.cuda.is_available())

    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_results_folder = os.path.join(results_folder, video_name)
        ensure_folder_exists(video_results_folder)

        csv_filename = os.path.join(video_results_folder, f'{video_name}.csv')
        output_video_path = os.path.join(video_results_folder, f'{video_name}_output.mp4')

        process_video(
            video_path, csv_filename, output_video_path, plate_model, color_model, reader, device
        )

        # CSV 파일 로드 및 교정
        df = pd.read_csv(csv_filename)
        df['ocr'] = df['ocr'].astype(str).apply(correct_plate_number)

        # 교정된 데이터를 저장
        corrected_csv_filename = os.path.join(video_results_folder, f'corrected_{video_name}.csv')
        df.to_csv(corrected_csv_filename, index=False)

if __name__ == "__main__":
    main()
