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
                # generate_response now returns a tuple (text, context_str)
                start_time = time.time()
                response_text, context_str = llm.generate_response(st.session_state.messages)
                end_time = time.time()
                
                st.markdown(response_text)
                st.caption(f"⏱️ Finished in {end_time - start_time:.2f}s")
                
                # Append assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Update session state for sidebar context
                st.session_state.last_context = context_str
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Sidebar for controls or info
with st.sidebar:
    st.header("About")
    st.info("This assistant uses a quantized Qwen 2.5 7B Instruct model running locally.")
    
    with st.expander("📝 Hướng dẫn sử dụng Ký ức"):
        st.markdown("""
        **1. Ghi nhớ:**
        - `Hãy nhớ: [nội dung]`
        - `Remember: [content]`
        *(Hoặc cứ nói "Tôi tên là..." máy sẽ tự nhớ)*
        
        **2. Quên:**
        - `Quên: [nội dung cũ]`
        - `Forget: [old content]`
        
        **3. Cập nhật:**
        - `Cập nhật: [nội dung mới]`
        - `Thay đổi: [nội dung mới]`
        *(Máy sẽ tự tìm cái cũ liên quan để xóa và lưu cái mới)*
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
