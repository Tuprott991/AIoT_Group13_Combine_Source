import os
from flask import request

# File path
FILE_PATH = 'test.mp3'

# Size of transmittion (1KB = 1024 bytes)
CHUNK_SIZE = 1024
def download_file():
    chunk_number = int(request.args.get('chunk', 0))  # Nhận số chunk từ query parameter
    file_size = os.path.getsize(FILE_PATH)  # Kích thước tổng của file

    # Đảm bảo rằng chunk_number không vượt quá số phần trong file
    if chunk_number * CHUNK_SIZE >= file_size:
        return 'No more chunks', 404

    # Đọc phần của file tương ứng với chunk_number
    with open(FILE_PATH, 'rb') as f:
        f.seek(chunk_number * CHUNK_SIZE)  # Di chuyển đến vị trí bắt đầu của phần (chunk)
        chunk_data = f.read(CHUNK_SIZE)  # Đọc dữ liệu

    # Trả phần dữ liệu (chunk) về cho client
    return chunk_data, 200
