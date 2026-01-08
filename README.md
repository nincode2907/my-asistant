# Local AI Assistant (Memory & RAG)

Dự án trợ lý ảo AI chạy offline trên máy tính cá nhân, tích hợp bộ nhớ dài hạn (Long-term Memory) thông qua RAG (Retrieval-Augmented Generation).

## 🚀 Tính năng nổi bật

-   **Offline 100%**: Chạy hoàn toàn trên CPU/GPU cá nhân, không gửi dữ liệu ra ngoài.
-   **Bộ nhớ dài hạn (Long-term Memory)**:
    -   Tự động ghi nhớ thông tin quan trọng (Tên, sở thích...).
    -   Hỗ trợ lệnh chủ động: "Hãy nhớ...", "Quên...", "Cập nhật...".
-   **RAG (Retrieval-Augmented Generation)**: Truy xuất ký ức liên quan theo ngữ cảnh câu chuyện.
-   **Giao diện trực quan**: Chat UI xây dựng bằng Streamlit.
-   **Tối ưu hiệu năng**: Hỗ trợ Quantized Models (GGUF) chạy tốt trên máy cấu hình tầm trung.

## 🛠️ Cài đặt

1.  **Clone repository**:
    ```bash
    git clone https://github.com/nincode2907/my-asistant.git
    cd my-asistant
    ```

2.  **Cài đặt thư viện**:
    Đảm bảo bạn đã cài Python 3.10+.
    ```bash
    pip install streamlit llama-cpp-python chromadb sentence-transformers
    ```

3.  **Chuẩn bị Model**:
    -   Tải model định dạng `.gguf` (ví dụ từ [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)).
    -   Đặt file model vào thư mục `models/` (hoặc tạo thư mục nếu chưa có).

## 📖 Hướng dẫn sử dụng

Chạy ứng dụng bằng lệnh:
```bash
streamlit run app.py
```

### Các lệnh quản lý bộ nhớ

| Lệnh | Cú pháp | Ví dụ |
| :--- | :--- | :--- |
| **Ghi nhớ** | `Hãy nhớ: [nội dung]` | `Hãy nhớ: Tôi là lập trình viên Python.` |
| **Quên** | `Quên: [nội dung cũ]` | `Quên: Tôi thích ăn táo.` |
| **Cập nhật** | `Cập nhật: [nội dung mới]` | `Cập nhật: Tôi chuyển sang thích ăn cam.` |

*Lưu ý: Hệ thống cũng tự động ghi nhớ các câu giới thiệu bản thân như "Tôi tên là...", "Sở thích của tôi là...".*

## ⚙️ Thay đổi Model

Để thay đổi model AI (ví dụ: nâng cấp phiên bản mới hoặc dùng model nhẹ hơn), làm theo các bước sau:

1.  Tải file `.gguf` mới về máy.
2.  Mở file `backend.py`.
3.  Tìm dòng khai báo `MODEL_PATH` (khoảng dòng 6) và sửa đường dẫn:

```python
# backend.py

# Sửa tên file ở đây
MODEL_PATH = "models/ten-model-moi-cua-ban.gguf"
```

4.  Lưu file và Refresh lại trang Streamlit (ứng dụng sẽ tự tải model mới).

## 📂 Cấu trúc dự án

-   `app.py`: Giao diện chính (Frontend - Streamlit).
-   `backend.py`: Xử lý logic AI và Prompt Engineering.
-   `memory.py`: Quản lý Database (ChromaDB) và tìm kiếm ngữ cảnh.
-   `memory_db/`: Thư mục chứa dữ liệu ký ức (Được tạo tự động).
-   `models/`: Nơi chứa các file model `.gguf`.
