from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    if 'images' not in request.files:
        return jsonify({'error': 'Không có ảnh nào được tải lên.'})

    files = request.files.getlist('images')
    if len(files) < 2:
        return jsonify({'error': 'Vui lòng chọn ít nhất hai ảnh.'})

    images = []
    for file in files:
        if file:
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({'error': f'Không thể đọc ảnh: {file.filename}'})
            images.append(img)

    try:
        stitcher = cv2.Stitcher_create()
        stitcher.setPanoConfidenceThresh(0.1)
        stitcher.setWaveCorrection(False)
        stitcher.setSeamEstimationResol(0.1)
        
        status, pano = stitcher.stitch(images)

        if status != cv2.Stitcher_OK:
            error_msg = {
                cv2.Stitcher_ERR_NEED_MORE_IMGS: 'Cần thêm ảnh để ghép.',
                cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: 'Không thể ước lượng homography.',
                cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: 'Không thể điều chỉnh tham số camera.'
            }.get(status, 'Lỗi không xác định.')
            return jsonify({'error': error_msg})

        # Lưu ảnh tạm thời vào /tmp
        temp_path = '/tmp/result.jpg'
        cv2.imwrite(temp_path, pano)

        # Đọc ảnh và chuyển thành base64
        with open(temp_path, 'rb') as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Xóa tệp tạm
        os.remove(temp_path)

        # Trả về dữ liệu base64
        return jsonify({'result': f'data:image/jpeg;base64,{image_base64}'})

    except Exception as e:
        return jsonify({'error': f'Đã xảy ra lỗi: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)