import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import os
import base64
import logging
import uuid
from io import BytesIO

app = Flask(__name__)

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def stitch_images_using_features(image1, image2):
    """Ghép hai ảnh bằng thuật toán SIFT với BFMatcher."""
    try:
        # Chuyển sang ảnh xám
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Khởi tạo SIFT
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return None, "Không tìm thấy đặc trưng trong một hoặc cả hai ảnh."

        # Khớp đặc trưng bằng BFMatcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Lọc các khớp tốt (Lowe's ratio test)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) <= 4:
            return None, "Không tìm thấy đủ đặc trưng để ghép ảnh."

        # Lấy điểm đặc trưng
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Tính homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None, "Không thể tính toán ma trận homography."

        # Lấy kích thước ảnh
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]

        # Tính kích thước ảnh đầu ra
        pts = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        img2_bounds = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        all_points = np.concatenate((dst, img2_bounds), axis=0)

        [x_min, y_min] = np.int32(all_points.min(axis=0).flatten())
        [x_max, y_max] = np.int32(all_points.max(axis=0).flatten())

        # Tính ma trận dịch chuyển
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        # Biến đổi và ghép ảnh
        stitched_img = cv2.warpPerspective(image1, H_translation @ H, (x_max - x_min, y_max - y_min))
        stitched_img[translation_dist[1]:translation_dist[1] + h2, translation_dist[0]:translation_dist[0] + w2] = image2

        return stitched_img, None

    except Exception as e:
        logger.error(f"Lỗi khi ghép ảnh: {str(e)}")
        return None, f"Lỗi khi ghép ảnh: {str(e)}"

@app.errorhandler(Exception)
def handle_error(error):
    """Xử lý lỗi server và trả về JSON."""
    logger.error(f"Lỗi server: {str(error)}")
    return jsonify({'error': 'Đã xảy ra lỗi server.'}), 500

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Lỗi khi render index: {str(e)}")
        return jsonify({'error': 'Không thể tải trang.'}), 500

@app.route('/upload', methods=['POST'])
def upload_images():
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'Không có ảnh nào được tải lên.'}), 400

        files = request.files.getlist('images')
        if len(files) < 2:
            return jsonify({'error': 'Vui lòng chọn ít nhất hai ảnh.'}), 400

        images = []
        for file in files:
            if file:
                file_bytes = file.read()
                nparr = np.frombuffer(file_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    return jsonify({'error': f'Không thể đọc ảnh: {file.filename}'}), 400
                images.append(img)

        # Ghép ảnh lần lượt
        result = images[0]
        for i in range(1, len(images)):
            result, error = stitch_images_using_features(result, images[i])
            if error:
                return jsonify({'error': error}), 400

        # Lưu ảnh tạm thời
        temp_filename = f'/tmp/result_{uuid.uuid4().hex}.jpg'
        cv2.imwrite(temp_filename, result)

        # Đọc và chuyển thành base64
        with open(temp_filename, 'rb') as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Xóa tệp tạm
        os.remove(temp_filename)

        return jsonify({'result': f'data:image/jpeg;base64,{image_base64}'})

    except Exception as e:
        logger.error(f"Lỗi trong upload_images: {str(e)}")
        return jsonify({'error': f'Đã xảy ra lỗi: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)