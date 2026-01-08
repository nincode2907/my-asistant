import os
import datetime
import webbrowser
import subprocess
import re
from ddgs import DDGS
from llama_cpp import Llama
import memory

# --- CẤU HÌNH ---
MODEL_PATH = "models/qwen2.5-3b-instruct-q4_k_m.gguf"

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

class LocalLLM:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"❌ Model file not found at: {os.path.abspath(MODEL_PATH)}")
        
        print(f"{Colors.GREEN}Loading model from {MODEL_PATH}...{Colors.ENDC}")
        self.llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=0,
            n_ctx=4096,
            n_threads=6,      
            n_batch=512,
            verbose=False
        )

    # --- HÀM THỰC THI (AGENT) ---
    def execute_command(self, command_str):
        print(f"{Colors.BLUE}[Agent Command] ⚙️ {command_str}{Colors.ENDC}")
        try:
            if "YourSearchQueryHere" in command_str:
                webbrowser.open("https://www.google.com")
                return "Link lỗi, đã mở Google."

            if command_str.startswith("OPEN:"):
                url = command_str.replace("OPEN:", "").strip()
                if "youtube.com" in url and "watch" not in url and "search_query" not in url:
                    url = "https://www.youtube.com"
                webbrowser.open(url)
                return f"Đã mở: {url}"
            
            elif command_str.startswith("APP:"):
                app_name = command_str.replace("APP:", "").strip().lower()
                if "notepad" in app_name: subprocess.Popen("notepad.exe")
                elif "calc" in app_name: subprocess.Popen("calc.exe")
                elif "excel" in app_name: subprocess.Popen("start excel", shell=True)
                elif "word" in app_name: subprocess.Popen("start winword", shell=True)
                elif "code" in app_name: subprocess.Popen("code", shell=True)
                elif "zalo" in app_name: subprocess.Popen(r"C:\Users\admin\AppData\Local\Programs\Zalo\Zalo.exe", shell=True)
                return f"Đã bật ứng dụng: {app_name}"
                
        except Exception as e:
            print(f"Lỗi thực thi: {e}")
            return None
        return None

    def tool_search(self, query):
        print(f"{Colors.BLUE}[Tool] 🔍 Đang tra cứu: {query}{Colors.ENDC}")
        try:
            results = DDGS().text(query, max_results=3)
            if not results: return "Không tìm thấy thông tin."
            summary = ""
            for res in results:
                summary += f"- {res['title']}: {res['body']}\n"
            return summary
        except:
            return "Lỗi kết nối mạng."

    def generate_response(self, messages):
        last_user_msg = next((m['content'] for m in reversed(messages) if m['role'] == 'user'), None)
        context_str = ""
        tool_data = ""
        system_note = "" # Dùng để thông báo cho bot biết vừa lưu/xóa ký ức

        if last_user_msg:
            lower_msg = last_user_msg.lower()

            # --- [PHẦN KHÔI PHỤC LẠI]: QUẢN LÝ KÝ ỨC (Memory Management) ---
            
            # 1. Lệnh QUÊN
            if lower_msg.startswith("quên:") or lower_msg.startswith("forget:"):
                content = last_user_msg.split(":", 1)[1].strip()
                res = memory.delete_similar_memory(content)
                system_note = f"[HỆ THỐNG: {res}]"

            # 2. Lệnh CẬP NHẬT (Thay đổi)
            elif lower_msg.startswith("thay đổi:") or lower_msg.startswith("cập nhật:") or lower_msg.startswith("update:"):
                new_content = last_user_msg.split(":", 1)[1].strip()
                # Xóa cái cũ tương tự -> Lưu cái mới
                del_res = memory.delete_similar_memory(new_content)
                memory.add_memory(new_content)
                system_note = f"[HỆ THỐNG: Đã cập nhật. {del_res}. Và đã lưu thông tin mới: '{new_content}']"

            # 3. Lệnh HÃY NHỚ
            else:
                is_explicit = lower_msg.startswith("hãy nhớ:") or lower_msg.startswith("remember:")

                if is_explicit:
                    text_to_save = last_user_msg.split(":", 1)[1].strip()
                    memory.add_memory(text_to_save)
                    system_note = f"[HỆ THỐNG: Đã lưu vào bộ nhớ dài hạn: '{text_to_save}']"

            # --- KẾT THÚC PHẦN KHÔI PHỤC ---

            # 4. RAG Retrieval (Lấy ký ức ra để dùng)
            context_str = memory.get_relevant_context(last_user_msg)
            
            # 5. Search Tool
            search_triggers = ["tìm", "tra", "giá", "thời tiết", "là ai", "dân số", "sự kiện", "ở đâu"]
            if any(k in lower_msg for k in search_triggers) and "mở" not in lower_msg:
                search_res = self.tool_search(last_user_msg)
                tool_data = f"\n[DỮ LIỆU TÌM KIẾM]:\n{search_res}\n"

        current_time = datetime.datetime.now().strftime("%H:%M %d/%m/%Y")
        
        # --- SYSTEM PROMPT ---
        system_prompt = f"""Bạn là Trợ lý Ảo Thông minh. Thời gian: {current_time}.

        NHIỆM VỤ:
        1. [ĐIỀU KHIỂN]: Nếu user bảo "Mở/Bật", hãy dùng lệnh (tất cả đều cho dùng tiếng Việt):
           - [[OPEN: https://www.youtube.com/results?search_query=...]]
           - [[APP: notepad/calc/excel/code]]
           
        2. [KÝ ỨC]: 
           - Nếu có thông báo [HỆ THỐNG: Đã lưu/xóa...], hãy xác nhận với người dùng.
           - Sử dụng [KÝ ỨC DÀI HẠN] để trả lời câu hỏi cá nhân.

        3. [TRA CỨU]: Dùng [DỮ LIỆU TÌM KIẾM] cho câu hỏi thực tế.
        """

        full_prompt = f"<|im_start|>system\n{system_prompt}"
        if context_str: full_prompt += f"\n[KÝ ỨC DÀI HẠN]: {context_str}"
        if tool_data: full_prompt += tool_data
        if system_note: full_prompt += f"\n{system_note}" # Bơm thông báo hệ thống vào prompt
        full_prompt += "<|im_end|>\n"

        # Chỉ lấy 4 tin nhắn gần nhất để bot đỡ loạn
        recent_messages = messages[-4:] if len(messages) > 4 else messages
        
        for msg in recent_messages:
            full_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        full_prompt += "<|im_start|>assistant\n"

        # --- STREAM ---
        stream_generator = self.llm(
            full_prompt, max_tokens=1024, stop=["<|im_end|>"], 
            echo=False, temperature=0.6, stream=True
        )

        def smart_stream():
            full_response = ""
            command_executed = False
            for chunk in stream_generator:
                text_chunk = chunk['choices'][0]['text']
                full_response += text_chunk
                
                match = re.search(r"\[\[(.*?)\]\]", full_response)
                if match and not command_executed:
                    self.execute_command(match.group(1))
                    command_executed = True
                    yield "✅ Đang thực hiện... \n"
                    full_response = full_response.replace(match.group(0), "")
                
                if not match: yield text_chunk

        return smart_stream(), context_str