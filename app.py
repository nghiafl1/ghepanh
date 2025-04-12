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
            # Resize ảnh để giảm tải xử lý
            img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            images.append(img)

    try:
        # Sử dụng chế độ PANORAMA để tối ưu hóa cho ảnh panorama
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        
        # Tùy chỉnh tham số để tăng khả năng thành công
        stitcher.setPanoConfidenceThresh(0.3)  # Giảm ngưỡng để chấp nhận nhiều đặc điểm hơn
        stitcher.setWaveCorrection(True)      # Bật chỉnh sóng để cải thiện chất lượng
        stitcher.setSeamEstimationResol(0.1)  # Độ phân giải ước lượng đường nối
        
        # Thêm bước tiền xử lý ảnh
        for i, img in enumerate(images):
            # Chuyển sang ảnh xám để tìm đặc điểm
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Tăng độ tương phản
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            images[i] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

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