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
        # Khởi tạo SIFT
        sift = cv2.SIFT_create()

        # Ghép từng cặp ảnh liên tiếp
        result = images[0]  # Bắt đầu với ảnh đầu tiên
        for i in range(1, len(images)):
            img1 = result
            img2 = images[i]

            # Phát hiện và mô tả đặc trưng bằng SIFT
            keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
            keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

            if descriptors1 is None or descriptors2 is None or len(descriptors1) < 4 or len(descriptors2) < 4:
                return jsonify({'error': 'Không đủ đặc trưng để ghép ảnh. Vui lòng chọn các ảnh có vùng chồng lấn rõ ràng (ít nhất 20-30%) và chứa các chi tiết nổi bật.'})

            # Khớp đặc trưng bằng Brute-Force Matcher
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)

            # Áp dụng ratio test để lọc các khớp tốt
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) < 4:
                return jsonify({'error': 'Không đủ khớp đặc trưng để ghép ảnh. Vui lòng chọn các ảnh có vùng chồng lấn rõ ràng (ít nhất 20-30%) và chứa các chi tiết nổi bật.'})

            # Trích xuất các điểm khớp
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Ước lượng ma trận homography
            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is None:
                return jsonify({'error': 'Không thể ước lượng homography. Vui lòng chọn các ảnh có vùng chồng lấn rõ ràng (ít nhất 20-30%) và chứa các chi tiết nổi bật.'})

            # Tính kích thước ảnh kết quả
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            corners2_transformed = cv2.perspectiveTransform(corners2, H)
            corners = np.concatenate((corners1, corners2_transformed), axis=0)

            # Tính kích thước ảnh panorama
            [x_min, y_min] = np.int32(corners.min(axis=0).ravel())
            [x_max, y_max] = np.int32(corners.max(axis=0).ravel())
            translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

            # Biến đổi ảnh và ghép
            img1_warped = cv2.warpPerspective(img1, translation_mat, (x_max - x_min, y_max - y_min))
            H_trans = translation_mat @ H
            img2_warped = cv2.warpPerspective(img2, H_trans, (x_max - x_min, y_max - y_min))

            # Tạo mask để pha trộn ảnh
            mask1 = (img1_warped > 0).astype(np.uint8) * 255
            mask2 = (img2_warped > 0).astype(np.uint8) * 255
            overlap = cv2.bitwise_and(mask1, mask2)
            mask1 = cv2.bitwise_and(mask1, cv2.bitwise_not(overlap))
            mask2 = cv2.bitwise_and(mask2, cv2.bitwise_not(overlap))

            # Ghép ảnh (không pha trộn phức tạp để tối ưu hiệu suất)
            result = np.zeros_like(img1_warped)
            result = cv2.bitwise_or(result, cv2.bitwise_and(img1_warped, mask1))
            result = cv2.bitwise_or(result, cv2.bitwise_and(img2_warped, mask2))
            
            # Pha trộn vùng chồng lấn (linear blending)
            overlap_area = cv2.bitwise_and(img1_warped, img2_warped, mask=overlap)
            alpha = 0.5
            overlap_area = cv2.addWeighted(img1_warped, alpha, img2_warped, 1 - alpha, 0, mask=overlap)
            result = cv2.bitwise_or(result, overlap_area)

        # Chuyển ảnh thành base64
        _, buffer = cv2.imencode('.jpg', result)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Trả về dữ liệu base64
        return jsonify({'result': f'data:image/jpeg;base64,{image_base64}'})

    except Exception as e:
        return jsonify({'error': f'Đã xảy ra lỗi: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)