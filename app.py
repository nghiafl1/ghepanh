from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
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
            
            # Giảm kích thước ảnh nếu quá lớn
            max_height = 800
            if img.shape[0] > max_height:
                scale = max_height / img.shape[0]
                img = cv2.resize(img, None, fx=scale, fy=scale)
            images.append(img)

    try:
        # Tạo Stitcher và sử dụng ORB (mặc định)
        stitcher = cv2.Stitcher_create()
        stitcher.setPanoConfidenceThresh(0.3)  # Tăng ngưỡng để cải thiện độ chính xác
        stitcher.setWaveCorrection(False)
        stitcher.setSeamEstimationResol(0.1)

        # Tắt bundle adjustment để tránh lỗi điều chỉnh tham số camera
        stitcher.setRegistrationResol(-1)  # Tắt ước lượng tham số camera

        status, pano = stitcher.stitch(images)

        if status != cv2.Stitcher_OK:
            error_msg = {
                cv2.Stitcher_ERR_NEED_MORE_IMGS: 'Cần thêm ảnh để ghép. Vui lòng chọn các ảnh có vùng chồng lấn (overlap) ít nhất 20-30%.',
                cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: 'Không thể ước lượng homography. Vui lòng chọn các ảnh có vùng chồng lấn rõ ràng (ít nhất 20-30%) và chứa các chi tiết nổi bật (như góc cạnh, họa tiết).',
                cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: 'Không thể điều chỉnh tham số camera. Vui lòng chọn các ảnh có vùng chồng lấn rõ ràng (ít nhất 20-30%) và chứa các chi tiết nổi bật. Nếu ảnh được chụp từ cùng một góc độ, hãy thử chụp lại với góc quay nhẹ (10-20 độ) giữa các ảnh.'
            }.get(status, 'Lỗi không xác định.')
            return jsonify({'error': error_msg})

        # Chuyển ảnh thành base64 mà không lưu vào đĩa
        _, buffer = cv2.imencode('.jpg', pano)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Trả về dữ liệu base64
        return jsonify({'result': f'data:image/jpeg;base64,{image_base64}'})

    except Exception as e:
        return jsonify({'error': f'Đã xảy ra lỗi: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)