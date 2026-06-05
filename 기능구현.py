from ultralytics import YOLO
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pytesseract
from gtts import gTTS
import os
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips

# Tesseract 경로 설정
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# YOLO 모델 로드
model = YOLO('/content/drive/MyDrive/thelast/1105_train_best.pt')

# 경로 설정
input_video_path = '/content/drive/MyDrive/thelast/green_num_real.mp4'
output_video_path = '/content/drive/MyDrive/thelast/1110_gtts_result_with_audio.mp4'

# 세 개의 ROI 이미지 경로 설정
roi_paths = [
    '/content/drive/MyDrive/thelast/roi_image/right.png',  # 오른쪽 영역
    '/content/drive/MyDrive/thelast/roi_image/left.png',   # 왼쪽 영역
    '/content/drive/MyDrive/thelast/roi_image/middle.png'  # 중앙 영역
]

# 폰트 설정
font_path = "/content/drive/MyDrive/thelast/LG_SMART_UI-BOLD.TTF"
font_size = 38
font = ImageFont.truetype(font_path, font_size)

# 임시 오디오 파일 저장 폴더 생성
audio_temp_dir = '/content/drive/MyDrive/temp_audio'
os.makedirs(audio_temp_dir, exist_ok=True)

# ROI 이미지 로드 및 오류 확인
rois = []
for path in roi_paths:
    roi_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if roi_img is None:
        print(f"오류: '{path}' 파일을 불러올 수 없습니다. 경로를 확인하세요.")
        continue
    rois.append(roi_img)

if len(rois) != 3:
    print("필요한 ROI 이미지를 모두 불러오지 못했습니다. 경로를 확인 후 다시 시도하세요.")
else:
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    resized_rois = []
    for roi_img in rois:
        if roi_img.shape[:2] != (height, width):
            roi_img = cv2.resize(roi_img, (width, height))
        resized_rois.append(roi_img)

    # 색상 설정
    colors = {
        "default": (0, 255, 0),
        "yellow": (0, 255, 255),
        "red": (0, 0, 255)
    }

    # 마지막 자막 텍스트
    last_text = None

    def put_text_with_background(frame, text, width, height):
        global last_text
        if last_text == text:
            return None  # Return None to avoid duplicate TTS processing

        last_text = text

        # TTS 음성 파일 생성 및 저장
        tts = gTTS(text=text, lang='ko')
        audio_path = f"{audio_temp_dir}/{text}.mp3"
        tts.save(audio_path)

        if not os.path.exists(audio_path):  # Check if the audio file exists
            print(f"Error: Audio file {audio_path} not created.")
            return None

        # 자막 배경과 텍스트 위치 설정
        img_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(img_pil)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        margin = 4
        box_x_start = (width - text_width) // 2 - margin
        box_x_end = (width + text_width) // 2 + margin
        box_y_start = height - text_height - 80 - margin
        box_y_end = height - 80 + margin
        draw.rectangle([(box_x_start, box_y_start), (box_x_end, box_y_end)], fill=(0, 0, 0, 255))
        text_position = ((width - text_width) // 2, box_y_start + margin)
        draw.text(text_position, text, font=font, fill=(255, 255, 255, 255))
        frame[:] = np.array(img_pil)

        return audio_path  # Return the valid audio file path

    # 비디오 처리 시작
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 비디오 생성
    out = cv2.VideoWriter('/content/temp_video_with_text.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 오디오 파일 목록
    audio_clips = []

    transparency = 0.2  # 투명도 설정 (20%)

    # 프레임 처리
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 모델로 프레임 분석
        results = model(frame)
        text = None  # 자막 텍스트 초기화

        # 각 ROI 이미지의 알파 채널을 사용하여 프레임에 추가
        for roi in resized_rois:
            if roi.shape[2] == 4:  # 알파채널만 가진거 처리
                bgr = roi[:, :, :3]
                alpha = roi[:, :, 3] / 255.0
                combined_alpha = alpha * transparency
                for c in range(3):
                    frame[:, :, c] = (1 - combined_alpha) * frame[:, :, c] + combined_alpha * bgr[:, :, c]

        # 관심 영역 마스크 생성
        mask_right = cv2.inRange(resized_rois[0][:, :, :3], np.array([0, 255, 255]), np.array([0, 255, 255]))
        mask_left = cv2.inRange(resized_rois[1][:, :, :3], np.array([0, 255, 255]), np.array([0, 255, 255]))
        mask_center = cv2.inRange(resized_rois[2][:, :, :3], np.array([0, 0, 255]), np.array([0, 0, 255]))

        # 객체별 검출 정보 표시
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            label = f"{class_name} {confidence:.2f}"
            color = colors["default"]
            text = ""

            # detection_region의 크기를 mask와 동일하게 설정
            detection_region = np.zeros_like(mask_center)  # 검출된 영역에 대한 빈 이미지

            # 검출된 바운딩 박스 영역에 대해 값을 할당
            detection_region[y1:y2, x1:x2] = 255

            # 비트 연산 수행
            in_center = cv2.bitwise_and(mask_center, detection_region).any()
            in_left = cv2.bitwise_and(mask_left, detection_region).any()
            in_right = cv2.bitwise_and(mask_right, detection_region).any()

            # 영역에 따라 색상 및 자막 설정
            if in_center:
                color = colors["red"]
                text = f"{class_name}이 중앙에 가까이 있습니다."
            elif in_left and in_right:
                color = colors["yellow"]
                text = f"양 쪽에 {class_name}이 있습니다."
            elif in_left:
                color = colors["yellow"]
                text = f"{class_name}이 왼쪽에 있습니다."
            elif in_right:
                color = colors["yellow"]
                text = f"{class_name}이 오른쪽에 있습니다."

            if text:
                # 자막 표시 및 오디오 경로 반환
                audio_path = put_text_with_background(frame, text, width, height)
                if audio_path:  # If a valid audio path is returned
                    audio_clip = AudioFileClip(audio_path)
                    audio_clips.append(audio_clip)

            # Tesseract 경로 설정 (Windows 사용자의 경우)
            # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

            # 특정 class_id에 해당하는 영역만 HSV 변환 및 OCR 적용
            if class_id == 31:  # 신호등 (traffic light)
                traffic_light_img = frame[y1:y2, x1:x2]  # 해당 객체 영역 추출

                # BGR에서 HSV로 변환
                hsv_img = cv2.cvtColor(traffic_light_img, cv2.COLOR_BGR2HSV)

                # 색상 범위 설정 (초록색, 빨간색)
                lower_green = np.array([35, 80, 80])
                upper_green = np.array([85, 255, 255])
                lower_red1 = np.array([0, 70, 50])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 70, 50])
                upper_red2 = np.array([180, 255, 255])

                # 초록색과 빨간색 마스크 생성
                mask_green = cv2.inRange(hsv_img, lower_green, upper_green)
                mask_red1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
                mask_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
                mask_red = mask_red1 | mask_red2

                # 색상 검출 (초록색 또는 빨간색) 먼저 수행
                if np.any(mask_green > 0):
                    color_state = "green"
                elif np.any(mask_red > 0):
                    color_state = "red"
                else:
                    color_state = None

                # 그레이스케일 변환
                gray_img = cv2.cvtColor(traffic_light_img, cv2.COLOR_BGR2GRAY)

                # 적응형 이진화 처리
                binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                  cv2.THRESH_BINARY, 13, 5)

                ocr_result = pytesseract.image_to_string(binary_img, config='--psm 7 digits').strip()

                # OCR 결과가 숫자로만 이루어진 문자열인지 확인
                if ocr_result.isdigit():
                    remaining_time = ocr_result  # 숫자로만 이루어진 문자열일 경우 그대로 저장
                else:
                    remaining_time = None  # 숫자가 아닌 경우 None 처리



                # 색상 상태에 따라 출력
                if color_state == "green":
                    frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.5, np.full_like(frame[y1:y2, x1:x2], (0, 255, 0), dtype=np.uint8), 0.5, 0)
                    cv2.putText(frame, "green", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if remaining_time is not None:
                        warning_text = f"green {remaining_time}"
                        cv2.putText(frame, warning_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                elif color_state == "red":
                    frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.5, np.full_like(frame[y1:y2, x1:x2], (0, 0, 255), dtype=np.uint8), 0.5, 0)
                    cv2.putText(frame, "red", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if remaining_time is not None:
                        warning_text = f"red {remaining_time}"
                        cv2.putText(frame, warning_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)








            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

    cap.release()
    out.release()

    # 비디오와 오디오 합성
    video_clip = VideoFileClip('/content/temp_video_with_text.mp4')
    final_audio = concatenate_audioclips(audio_clips)

    # 최종 비디오와 오디오를 결합
    final_video = video_clip.set_audio(final_audio)
    final_video.write_videofile(output_video_path, codec='libx264')

    print(f"완료! 최종 비디오가 '{output_video_path}'로 저장되었습니다.")
