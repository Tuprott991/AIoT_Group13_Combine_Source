import os  # to handle file paths
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import StreamingResponse

# OBJECT CLASSIFICATION PROGRAM FOR VIDEO IN IP ADDRESS
file_path = "./temp"  # Specify the directory where you want to save the image

# Ensure the directory exists
os.makedirs(file_path, exist_ok=True)


async def receive_image(request: Request):
    try:
        # Đọc toàn bộ dữ liệu binary từ body
        image_data = await request.body()

        # Tạo tên file theo thời gian hoặc cố định
        filename = "camera_image.jpg"
        save_path = os.path.join(SAVE_DIR, filename)

        # Lưu dữ liệu ảnh
        with open(save_path, "wb") as f:
            f.write(image_data)

        print(f"✅ Received binary image: {filename}")
        return {"filename": filename, "message": "Image uploaded successfully"}
    except Exception as e:
        print(f"❌ Error saving image: {str(e)}")
        return {"error": "Failed to save image"}, 500


