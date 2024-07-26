Chạy code
1. Bỏ các model cá nhân vào file 'weights'.
2. Chạy bằng lệnh 'streamlit run app.py' (Sau khi đã ở đúng directory).
3. Nếu lỗi thiếu 'cython-bbox' thì chạy 'pip install cython-bbox'.

Test app
1. Chọn task (Detection / Tracking / Draw Zone).
2. Lựa chọn confidence (độ tin cậy).
3. Lựa chọn model.
4. Chọn đầu vào (Image / Video / Webcam).
5. Tải video/image.
6. Tải tọa độ count zone hoặc vẽ zone ở page "Draw Zone" nếu cần thiết.