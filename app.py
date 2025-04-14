import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
import os
import base64
from io import BytesIO
import uuid

app = Flask(__name__)

def extract_sift_features(image):
    """Extract SIFT features from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(des1, des2):
    """Match SIFT features between two images using FLANN matcher."""
    if des1 is None or des2 is None:
        return None
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

def compute_homography(kp1, kp2, matches):
    """Compute homography matrix using RANSAC."""
    if len(matches) < 4:
        return None, None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask

def stitch_images(img1, img2):
    """Stitch two images using SIFT features."""
    # Extract features
    kp1, des1 = extract_sift_features(img1)
    kp2, des2 = extract_sift_features(img2)
    
    # Match features
    matches = match_features(des1, des2)
    if not matches or len(matches) < 4:
        return None, "Không đủ điểm đặc trưng để ghép ảnh."
    
    # Compute homography
    H, mask = compute_homography(kp1, kp2, matches)
    if H is None:
        return None, "Không thể tính toán ma trận homography."
    
    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Calculate the size of the output image
    corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    corners = np.concatenate((warped_corners, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)), axis=0)
    
    [x_min, y_min] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(corners.max(axis=0).ravel() + 0.5)
    
    # Translation matrix to shift the image
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    
    # Warp first image
    warped_img1 = cv2.warpPerspective(img1, translation.dot(H), (x_max - x_min, y_max - y_min))
    
    # Create output canvas
    result = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)
    
    # Place second image
    result[-y_min:-y_min+h2, -x_min:-x_min+w2] = img2
    
    # Blend images (simple overlay for non-overlapping regions)
    mask = warped_img1 > 0
    result[mask] = warped_img1[mask]
    
    return result, None

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
        # Start with the first image
        result = images[0]
        
        # Stitch each subsequent image
        for i in range(1, len(images)):
            result, error = stitch_images(result, images[i])
            if error:
                return jsonify({'error': error})

        # Save temporary image
        temp_filename = f'/tmp/result_{uuid.uuid4().hex}.jpg'
        cv2.imwrite(temp_filename, result)

        # Read and convert to base64
        with open(temp_filename, 'rb') as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Clean up
        os.remove(temp_filename)

        return jsonify({'result': f'data:image/jpeg;base64,{image_base64}'})

    except Exception as e:
        return jsonify({'error': f'Đã xảy ra lỗi: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)