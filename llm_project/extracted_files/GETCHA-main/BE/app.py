import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

import torch
from flask import Flask, request, jsonify, render_template_string, send_file
from PIL import Image, ImageDraw, ImageFont
import io

# Flask 앱 초기화
app = Flask(__name__)

# YOLOv5 학습된 모델 로드 
model = torch.hub.load('ultralytics/yolov5', 'custom', 
                       path='학습된 모델 경로',
                       force_reload=True)

# 빨간색 (RGB 값)
red_color = (255, 0, 0)

# 간단한 HTML 확인 및 이미지 업로드 페이지 추가
@app.route('/')
def home():
    return render_template_string('''
    <html>
    <head><title>Flask 서버 확인</title></head>
    <body>
        <h1>이미지 업로드 테스트</h1>
        <form method="POST" action="/predict" enctype="multipart/form-data">
            <label>이미지 선택:</label>
            <input type="file" name="image" accept="image/*">
            <br><br>
            <input type="submit" value="이미지 업로드 및 예측">
        </form>
    </body>
    </html>
    ''')

# 추론 함수 정의 (바운딩 박스 및 텍스트 추가)
def predict_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    results = model(img)

    # 바운딩 박스를 그리기 위한 객체
    draw = ImageDraw.Draw(img)

    # 폰트 설정 (크기 조절, 경로는 시스템에 맞게 설정해야 함)
    try:
        font = ImageFont.truetype("arial.ttf", 30)  # Arial 폰트와 크기 70
    except IOError:
        font = ImageFont.load_default()  # Arial 폰트가 없을 때 기본 폰트 사용

    # 바운딩 박스 정보 저장 (클래스, 신뢰도)
    predictions = []

    # 결과에서 좌표와 신뢰도 추출
    for i, box in enumerate(results.xyxy[0]):  # 예측된 모든 객체에 대해
        xmin, ymin, xmax, ymax, confidence, class_id = box[:6]

        # 모든 객체에 빨간색을 적용
        color = red_color

        # 바운딩 박스 그리기
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=5)  # 빨간색 바운딩 박스
        draw.text((xmin, ymin), f'{i+1}. {results.names[int(class_id)]} {confidence:.2f}', fill=color, font=font)  # 객체 번호와 클래스 이름, 신뢰도 표시

        # 클래스, 신뢰도, 바운딩 박스 좌표를 JSON에 저장
        predictions.append({
            "object_id": i + 1,  # 객체 번호 추가
            "class": results.names[int(class_id)],
            "confidence": float(confidence),
            "box": [float(xmin), float(ymin), float(xmax), float(ymax)],
            "color": color  # 빨간색 추가
        })

    # 이미지를 로컬에 저장 (웹에서 볼 수 있도록)
    img_path = "predicted_image.jpg"
    img.save(img_path)

    return img_path, predictions

# API 엔드포인트 정의 (이미지 추론)
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Image file is required'}), 400
    file = request.files['image']
    img_bytes = file.read()

    # 바운딩 박스 그려진 이미지와 예측 정보 반환
    img_path, predictions = predict_image(img_bytes)
    
    # 이미지 URL과 예측 결과 반환
    return jsonify({"image_url": "/image", "predictions": predictions})

# 이미지를 반환하는 엔드포인트
@app.route('/image', methods=['GET'])
def get_image():
    return send_file("predicted_image.jpg", mimetype='image/jpeg')

# Flask 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
