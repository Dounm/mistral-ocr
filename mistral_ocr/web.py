import os
import json
import gradio as gr
from pathlib import Path
from mistralai import Mistral
from loguru import logger
from .main import process_image, process_pdf, generate_output_content

# 配置日志输出到终端
logger.remove()  # 移除默认的处理器

# 确保logs文件夹存在
os.makedirs("logs", exist_ok=True)

logger.add(
    sink="logs/app.log",  # 输出到logs文件夹下的app.log文件
    rotation="500 MB",    # 当日志文件达到500MB时轮转
    retention="10 days",  # 保留10天的日志
    # format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="DEBUG",        # 设置日志级别为 DEBUG
    colorize=False,
    encoding="utf-8"
)

def process_file(
    file_obj,
    api_key,
    output_format="markdown",
    include_images=False,
):
    """Process uploaded file with Mistral OCR"""
    try:
        # 验证 API key
        if not api_key or api_key.strip() == "":
            logger.error("Empty API key provided")
            return "Error: Please provide a valid Mistral API key"

        # 确保 API key 格式正确
        api_key = api_key.strip()
        if not api_key.startswith(""):  # 检查 API key 前缀
            logger.warning("API key format might be incorrect")
            
        # Initialize client
        logger.debug("Initializing Mistral client")
        client = Mistral(api_key=api_key)
        
        # Save temporary file
        temp_path = Path(file_obj.name)
        logger.info(f"Processing file: {temp_path}")
        logger.debug(f"File object type: {type(file_obj)}")
        logger.debug(f"File object attributes: {dir(file_obj)}")
        
        if not temp_path.exists():
            logger.error(f"File not found: {temp_path}")
            return "Error: File not found or not accessible"
        
        # Process based on file type
        file_ext = temp_path.suffix.lower()
        logger.info(f"File extension: {file_ext}")
        logger.debug(f"File size: {temp_path.stat().st_size} bytes")
        
        if file_ext in ['.png', '.jpg', '.jpeg']:
            logger.debug("Processing as image file")
            response = process_image(
                client=client,
                image_path=temp_path,
                model="mistral-ocr-latest",
                include_images=include_images,
                silent=True
            )
            uploaded_file = None
        elif file_ext == '.pdf':
            logger.debug("Processing as PDF file")
            response, uploaded_file = process_pdf(
                client=client,
                pdf_path=temp_path,
                model="mistral-ocr-latest",
                include_images=include_images,
                silent=True
            )
        else:
            logger.warning(f"Unsupported file format: {file_ext}")
            return f"Error: Unsupported file format: {file_ext}. Please upload a PDF or image file (PNG, JPG, JPEG)."

        try:
            # Generate output
            response_json = response.model_dump_json()
            response_dict = json.loads(response_json)  # 将 JSON 字符串解析为字典
            logger.debug("Successfully parsed API response to dictionary")
            
            result = generate_output_content(
                response_dict=response_dict,
                output_format=output_format,
                image_map={}
            )
            return result
        finally:
            # Cleanup
            if uploaded_file:
                client.files.delete(file_id=uploaded_file.id)
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return f"Error: {str(e)}"

def create_web_ui():
    """Create Gradio web interface"""
    with gr.Blocks(title="Mistral OCR") as app:
        gr.Markdown("""
        # Mistral OCR
        Upload a PDF or image file to extract text using Mistral's OCR API.
        """)
        
        with gr.Row():
            with gr.Column():
                api_key = gr.Textbox(
                    label="Mistral API Key",
                    placeholder="Enter your API key or set MISTRAL_API_KEY environment variable",
                    value=os.getenv("MISTRAL_API_KEY", ""),
                    type="password"
                )
                with gr.Column():
                    file_input = gr.File(
                        label="Upload File",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                        type="filepath"
                    )
                    image_preview = gr.Image(
                        label="Image Preview",
                        visible=False,
                        width=400
                    )
                format_input = gr.Radio(
                    choices=["markdown", "html"],
                    value="markdown",
                    label="Output Format"
                )
                submit_btn = gr.Button("Process")

                # 添加文件改变事件处理
                def update_preview(file_obj):
                    if file_obj is None:
                        return None, gr.update(visible=False)
                    
                    file_ext = Path(file_obj.name).suffix.lower()
                    if file_ext in ['.png', '.jpg', '.jpeg']:
                        return file_obj.name, gr.update(visible=True)
                    return None, gr.update(visible=False)

                file_input.change(
                    fn=update_preview,
                    inputs=[file_input],
                    outputs=[image_preview, image_preview]
                )
            
            with gr.Column():
                with gr.Column():
                    output = gr.TextArea(
                        label="Results",
                        interactive=True,  # 改为可交互
                        show_copy_button=False,
                        lines=20,
                    )

        # 添加按钮点击事件处理
        submit_btn.click(
            fn=process_file,
            inputs=[file_input, api_key, format_input],
            outputs=output
        )
    
    return app

def main():
    """Launch the web interface"""
    app = create_web_ui()
    app.launch()

if __name__ == "__main__":
    main()