from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
import base64

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
        # Tạo SIFT và tìm các đặc trưng
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(images[0], None)
        keypoints2, descriptors2 = sift.detectAndCompute(images[1], None)

        # Khớp các đặc trưng
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(descriptors1, descriptors2)

        # Sắp xếp các kết quả khớp theo khoảng cách
        matches = sorted(matches, key=lambda x: x.distance)

        # Lấy các điểm từ các kết quả khớp
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Tính toán homography
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Ghép ảnh
        height, width, channels = images[1].shape
        pano = cv2.warpPerspective(images[0], H, (width * 2, height))
        pano[0:height, 0:width] = images[1]

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
