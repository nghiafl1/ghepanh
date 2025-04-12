document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('image-upload');
    const stitchBtn = document.getElementById('stitch-btn');
    const newStitchBtn = document.getElementById('new-stitch-btn');
    const selectedImages = document.getElementById('selected-images');
    const resultImage = document.getElementById('result-image');
    const noResult = document.getElementById('no-result');
    
    let files = [];
    
    const resetInterface = () => {
        files = [];
        selectedImages.innerHTML = '';
        resultImage.src = '';
        resultImage.style.display = 'none';
        noResult.textContent = 'Chưa có kết quả';
        noResult.style.color = '#ffffff';
        noResult.style.display = 'block';
        stitchBtn.disabled = true;
        newStitchBtn.style.display = 'none';
        imageUpload.value = '';
    };
    
    imageUpload.addEventListener('change', (e) => {
        files = Array.from(e.target.files);
        selectedImages.innerHTML = '';
        
        files.forEach(file => {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            selectedImages.appendChild(img);
        });

        if (files.length >= 2) {
            stitchBtn.disabled = false;
            noResult.textContent = 'Sẵn sàng để ghép ảnh.';
            noResult.style.color = '#e0aaff';
        } else {
            stitchBtn.disabled = true;
            noResult.textContent = 'Vui lòng chọn ít nhất hai ảnh.';
            noResult.style.color = '#ff4d4d';
        }
    });
    
    stitchBtn.addEventListener('click', async (event) => {
        event.preventDefault();

        if (files.length < 2) {
            noResult.textContent = 'Vui lòng chọn ít nhất hai ảnh.';
            noResult.style.color = '#ff4d4d';
            return;
        }
        
        const formData = new FormData();
        files.forEach(file => formData.append('images', file));
        
        try {
            stitchBtn.disabled = true;
            stitchBtn.textContent = 'Đang ghép ảnh...';
            noResult.textContent = 'Đang xử lý...';
            noResult.style.color = '#e0aaff';

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.error) {
                noResult.textContent = data.error;
                noResult.style.color = '#ff4d4d';
                newStitchBtn.style.display = 'inline-block';
                return;
            }
            
            noResult.style.display = 'none';
            resultImage.src = data.result; // Dữ liệu base64
            resultImage.style.display = 'block';
            
            newStitchBtn.style.display = 'inline-block';
        } catch (error) {
            noResult.textContent = 'Đã xảy ra lỗi: ' + error.message;
            noResult.style.color = '#ff4d4d';
            newStitchBtn.style.display = 'inline-block';
        } finally {
            stitchBtn.disabled = false;
            stitchBtn.textContent = 'Ghép Ảnh';
        }
    });

    newStitchBtn.addEventListener('click', (event) => {
        event.preventDefault();
        resetInterface();
    });
});