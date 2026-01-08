import streamlit as st
import time
from backend import LocalLLM

# Page Setup
st.set_page_config(
    page_title="Local Qwen 2.5 7B Assistant",
    page_icon="�",
    layout="wide"
)

st.title("� Local Qwen 2.5 7B")
st.markdown("*Powered by Qwen-2.5-7B and llama.cpp*")

# Initialize Backend with Caching
# @st.cache_resource ensures the model is loaded only once
@st.cache_resource
def get_llm():
    return LocalLLM()

try:
    llm = get_llm()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask me a complex question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Status container for progress updates (Searching, Opening apps...)
                status_box = st.empty()
                
                def on_status_update(msg):
                    status_box.info(msg, icon="⏳")

                start_time = time.time()
                
                # generate_response return (stream_generator, context_str)
                # Note: The stream itself now handles command execution internally
                stream, context_str = llm.generate_response(st.session_state.messages)
                
                # Generator wrapper to extract text from llama-cpp stream (or the custom string generator)
                def stream_generator():
                    # If stream is a generator function (from smart_stream), it yields strings directly
                    try:
                        for chunk in stream:
                            # If chunk is already a string (from smart_stream), yield it
                            if isinstance(chunk, str):
                                yield chunk
                            # If chunk is dictionary (from raw llama stream - fallback), yield text
                            elif isinstance(chunk, dict) and 'choices' in chunk:
                                yield chunk['choices'][0]['text']
                    except Exception as e:
                         # Handle any generator errors
                         yield f"\n[Stream Error: {str(e)}]"
                
                # Stream the response
                response_text = st.write_stream(stream_generator())
                end_time = time.time()
                
                st.caption(f"⏱️ Finished in {end_time - start_time:.2f}s")
                
                # Append assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Update session state for sidebar context
                st.session_state.last_context = context_str
                
            except Exception as e:
                status_box.empty() # Clear status if error
                st.error(f"An error occurred: {e}")

# Sidebar for controls or info
with st.sidebar:
    st.header("About")
    st.info("This assistant uses a quantized Qwen 2.5 7B Instruct model running locally.")
    
    with st.expander("📝 Hướng dẫn sử dụng"):
        st.markdown("""
        **1. 🧠 Quản lý Ký ức:**
        - `Hãy nhớ: [nội dung]`
        - `Quên: [nội dung cũ]`
        - `Cập nhật: [nội dung mới]`
        
        **2. 🌐 Tìm kiếm Internet:**
        - `Tìm kiếm: [từ khóa]`
        - `Tra cứu: [vấn đề]`
        - `Giá vàng/Thời tiết...`
        *(Tự động tìm nếu cần thông tin mới)*
        
        **3. 🚀 Mở Ứng dụng:**
        - `Mở Youtube / Facebook`
        - `Bật nhạc / Soundcloud`
        """)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_context = ""
        st.rerun()

    st.divider()
    st.subheader("🧠 Memory Debug of Last Reply")
    if "last_context" in st.session_state and st.session_state.last_context:
        st.info("Found relevant memories:")
        st.text(st.session_state.last_context)
    else:
        st.caption("No relevant memory found or not a query.")
