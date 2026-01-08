import os
from llama_cpp import Llama
import memory

# Path to your GGUF model
MODEL_PATH = "models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"

class LocalLLM:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {os.path.abspath(MODEL_PATH)}")
        
        print(f"Loading model from {MODEL_PATH}...")
        self.llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=0,  # CPU only
            n_ctx=4096,      # Giảm xuống 4096 để chạy nhanh hơn trên CPU
            verbose=False
        )

    def generate_response(self, messages):
        # 1. Lấy tin nhắn mới nhất
        last_user_msg = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), None)
        context_str = ""
        system_note = ""

        if last_user_msg:
            lower_msg = last_user_msg.lower()

            # --- TÍNH NĂNG MỚI: QUẢN LÝ KÝ ỨC (QUÊN / CẬP NHẬT) ---
            # Kiểm tra các lệnh rõ ràng trước
            is_forget = lower_msg.startswith("quên:") or lower_msg.startswith("forget:")
            is_update = lower_msg.startswith("thay đổi:") or lower_msg.startswith("cập nhật:") or lower_msg.startswith("update:")
            
            if is_forget:
                # Lấy nội dung sau dấu hai chấm
                content = last_user_msg.split(":", 1)[1].strip()
                # Gọi hàm xóa
                result = memory.delete_similar_memory(content)
                system_note = f"[System: {result}]"
                
            elif is_update:
                # Lấy nội dung mới
                new_content = last_user_msg.split(":", 1)[1].strip()
                # 1. Xóa thông tin cũ tương tự (The Update Hack)
                del_result = memory.delete_similar_memory(new_content)
                # 2. Lưu thông tin mới
                add_result = memory.add_memory(new_content)
                
                system_note = f"[System: Đã cập nhật. {del_result} -> {add_result}]"
                
            else:
                # --- TÍNH NĂNG CŨ: LƯU KÝ ỨC (GHI NHỚ / TỰ ĐỘNG) ---
                # Chỉ chạy nếu không phải lệnh Quên/Cập nhật
                auto_keywords = ["tôi tên là", "tên tôi là", "sở thích của tôi", "tôi thích", "nhà tôi ở"]
                is_command = lower_msg.startswith("remember:") or lower_msg.startswith("hãy nhớ:")
                is_auto_save = any(k in lower_msg for k in auto_keywords)

                if is_command or is_auto_save:
                    # Nếu là lệnh thì cắt bỏ phần đầu, nếu tự động thì lưu cả câu
                    text_to_save = last_user_msg.split(":", 1)[1].strip() if is_command else last_user_msg
                    
                    # Gọi hàm lưu vào DB
                    memory.add_memory(text_to_save)
                    
                    # Thêm ghi chú để Bot biết nó vừa lưu thành công
                    system_note = f"[System: Đã lưu thông tin '{text_to_save}' vào bộ nhớ dài hạn.]"

            # --- TÍNH NĂNG 2: HỒI TƯỞNG (RAG) ---
            # Luôn tìm kiếm ngữ cảnh liên quan trong DB
            context_str = memory.get_relevant_context(last_user_msg)

        # 2. Xây dựng Prompt
        full_prompt = ""
        
        # System Prompt được tinh chỉnh để Bot biết cách dùng ký ức
        base_system = """Bạn là trợ lý AI cá nhân hữu ích. 
        Nhiệm vụ: Trả lời người dùng một cách thân thiện.
        QUAN TRỌNG:
        1. Sử dụng thông tin trong phần 'Context information' để trả lời câu hỏi về bản thân người dùng.
        2. Nếu người dùng cung cấp thông tin mới, hãy xác nhận là bạn đã nhớ."""

        # Ghép ngữ cảnh tìm được vào Prompt
        if context_str:
            base_system += f"\n\nContext information (Ký ức đã lưu):\n{context_str}\n----------------"
        
        if system_note:
            base_system += f"\n{system_note}"

        full_prompt += f"<|im_start|>system\n{base_system}<|im_end|>\n"
        
        # Ghép lịch sử chat
        for msg in messages:
            full_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            
        full_prompt += "<|im_start|>assistant\n"
        
        # 3. Chạy Model
        output = self.llm(
            full_prompt,
            max_tokens=1024,
            stop=["<|im_end|>", "<|endoftext|>"],
            echo=False,
            temperature=0.7
        )
        
        response_text = output['choices'][0]['text'].strip()
        
        # Trả về cả câu trả lời lẫn ngữ cảnh (để debug hiển thị ra UI nếu cần)
        return response_text, context_str