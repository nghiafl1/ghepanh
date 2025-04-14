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

def stitch_images_sift(images):
    # Khởi tạo SIFT detector
    sift = cv2.SIFT_create()

    keypoints = []
    descriptors = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(des)

    # Khởi tạo BFMatcher (Brute-Force Matcher) hoặc FlannBasedMatcher
    bf = cv2.BFMatcher()
    matches = []
    for i in range(len(images) - 1):
        if descriptors[i] is not None and descriptors[i+1] is not None:
            these_matches = bf.knnMatch(descriptors[i], descriptors[i+1], k=2)
            good_matches = []
            for m, n in these_matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            matches.append(good_matches)
        else:
            return None, "Không đủ đặc trưng được tìm thấy trong một hoặc nhiều ảnh."

    if not matches:
        return None, "Không tìm thấy đủ các cặp điểm tương ứng giữa các ảnh."

    # Lấy keypoint từ các match tốt
    src_pts = []
    dst_pts = []
    for i, good_match in enumerate(matches):
        if good_match:
            src_pts_img = np.float32([keypoints[i][m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
            dst_pts_img = np.float32([keypoints[i+1][m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)
            src_pts.append(src_pts_img)
            dst_pts.append(dst_pts_img)
        else:
            return None, f"Không tìm thấy đủ match tốt giữa ảnh {i} và {i+1}."

    # Tính toán Homography
    H_matrices = []
    for i in range(len(src_pts)):
        if len(src_pts[i]) >= 4:
            H, mask = cv2.findHomography(src_pts[i], dst_pts[i], cv2.RANSAC, 5.0)
            if H is None:
                return None, f"Không thể tính toán Homography giữa ảnh {i} và {i+1}."
            H_matrices.append(H)
        else:
            return None, f"Không đủ điểm tương ứng để tính toán Homography giữa ảnh {i} và {i+1}."

    if not H_matrices:
        return None, "Không có ma trận Homography nào được tính toán."

    # Ghép ảnh (đơn giản hóa cho trường hợp 2 ảnh)
    if len(images) == 2 and len(H_matrices) == 1:
        h1, w1 = images[0].shape[:2]
        h2, w2 = images[1].shape[:2]

        # Lấy các góc của ảnh đầu tiên và transform chúng
        pts1 = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts1, H_matrices[0])

        # Tính toán kích thước panorama
        [x_min, y_min] = np.int32(dst.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(dst.max(axis=0).ravel() + 0.5)
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

        # Warp ảnh thứ hai
        warped_img2 = cv2.warpPerspective(images[1], H_matrices[0], (x_max - x_min + w1, y_max - y_min + h1))

        # Tạo canvas lớn hơn và đặt ảnh đầu tiên vào
        pano = np.zeros((y_max - y_min + h1, x_max - x_min + w1, 3), dtype=np.uint8)
        pano[translation_dist[1]:h1 + translation_dist[1], translation_dist[0]:w1 + translation_dist[0]] = images[0]

        # Kết hợp hai ảnh
        non_black_mask = np.any(warped_img2 > 0, axis=-1)
        pano[non_black_mask] = warped_img2[non_black_mask]

        return pano, None
    elif len(images) > 2:
        return None, "Ghép nhiều hơn 2 ảnh bằng SIFT cần một quy trình phức tạp hơn (ví dụ: tìm đồ thị ghép, ghép từng cặp)."
    else:
        return None, "Cần ít nhất hai ảnh để ghép."

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
        pano, error_msg = stitch_images_sift(images)

        if error_msg:
            return jsonify({'error': error_msg})

        if pano is None:
            return jsonify({'error': 'Không thể ghép ảnh.'})

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