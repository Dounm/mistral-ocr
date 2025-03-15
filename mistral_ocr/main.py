# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
#     "mistralai",
#     "markdown",
#     "loguru",
# ]
# ///

import json
import base64
import re
from pathlib import Path
import click
import markdown
from mistralai import Mistral
from mistralai import DocumentURLChunk, ImageURLChunk
from loguru import logger


def validate_options(api_key, output, output_dir, json_, extract_images, inline_images):
    """验证命令行选项的组合是否有效"""
    logger.debug("Validating command line options")
    if not api_key:
        logger.error("No API key provided")
        raise click.ClickException(
            "No API key provided and MISTRAL_API_KEY environment variable not set."
        )

    if output and output_dir:
        raise click.ClickException("Cannot specify both --output and --output-dir")

    if json_ and output_dir:
        raise click.ClickException("JSON output is not supported with --output-dir")

    if extract_images and not output_dir:
        raise click.ClickException("--extract-images requires --output-dir")

    if inline_images and extract_images:
        raise click.ClickException(
            "Cannot specify both --inline-images and --extract-images"
        )


def process_image(client, image_path, model, include_images, silent=False):
    """处理图片文件"""
    logger.info(f"Processing image file: {image_path}")
    if not silent:
        click.echo("Processing image with OCR...", err=True)

    try:
        encoded = base64.b64encode(image_path.read_bytes()).decode()
        base64_data_url = f"data:image/jpeg;base64,{encoded}"
        
        logger.debug(f"Sending image to OCR API with model: {model}")
        response = client.ocr.process(
            document=ImageURLChunk(image_url=base64_data_url),
            model=model,
            include_image_base64=include_images,
        )
        logger.info("Image processing completed successfully")
        return response
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise


def process_pdf(client, pdf_path, model, include_images, silent=False):
    """处理PDF文件"""
    logger.info(f"Processing PDF file: {pdf_path}")
    if not silent:
        click.echo(f"Uploading file {pdf_path.name}...", err=True)

    try:
        uploaded_file = client.files.upload(
            file={
                "file_name": pdf_path.stem,
                "content": pdf_path.read_bytes(),
            },
            purpose="ocr",
        )
        logger.debug(f"File uploaded successfully with ID: {uploaded_file.id}")

        try:
            signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
            logger.debug("Got signed URL for uploaded file")

            if not silent:
                click.echo(f"Processing with OCR model: {model}...", err=True)

            response = client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model=model,
                include_image_base64=include_images,
            )
            logger.info("PDF processing completed successfully")
            return response, uploaded_file
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            if uploaded_file:
                client.files.delete(file_id=uploaded_file.id)
                logger.debug(f"Deleted uploaded file: {uploaded_file.id}")
            raise e
    except Exception as e:
        logger.error(f"Error uploading PDF: {str(e)}")
        raise


def extract_images_to_dir(response_dict, output_dir, silent=False):
    """从OCR响应中提取图片到指定目录"""
    logger.info(f"Extracting images to directory: {output_dir}")
    image_map = {}
    image_count = 0
    image_dir = Path(output_dir)

    try:
        for page in response_dict.get("pages", []):
            for img in page.get("images", []):
                if "id" in img and "image_base64" in img:
                    image_data = img["image_base64"]
                    if image_data.startswith("data:image/"):
                        image_data = image_data.split(",", 1)[1]

                    image_filename = img["id"]
                    image_path = image_dir / image_filename
                    logger.debug(f"Saving image to: {image_path}")

                    with open(image_path, "wb") as img_file:
                        img_file.write(base64.b64decode(image_data))

                    image_map[image_filename] = image_filename
                    image_count += 1

        logger.info(f"Successfully extracted {image_count} images")
        if not silent:
            click.echo(f"Extracted {image_count} images to {image_dir}", err=True)

        return image_map
    except Exception as e:
        logger.error(f"Error extracting images: {str(e)}")
        raise


def create_inline_image_map(response_dict):
    """创建内联图片映射"""
    image_map = {}
    for page in response_dict.get("pages", []):
        for img in page.get("images", []):
            if "id" in img and "image_base64" in img:
                image_id = img["id"]
                image_data = img["image_base64"]
                if not image_data.startswith("data:"):
                    ext = image_id.split(".")[-1].lower() if "." in image_id else "jpeg"
                    mime_type = f"image/{ext}"
                    image_data = f"data:{mime_type};base64,{image_data}"
                image_map[image_id] = image_data
    return image_map


def generate_output_content(response_dict, output_format, image_map):
    """根据输出格式生成内容"""
    if output_format == "json":
        return json.dumps(response_dict, indent=4)

    markdown_contents = [
        page.get("markdown", "") for page in response_dict.get("pages", [])
    ]
    markdown_text = "\n\n".join(markdown_contents)

    # Handle image references in markdown
    for img_id, img_src in image_map.items():
        markdown_text = re.sub(
            r"!\[(.*?)\]\(" + re.escape(img_id) + r"\)",
            r"![\1](" + img_src + r")",
            markdown_text,
        )

    if output_format == "html":
        return generate_html_content(markdown_text)
    return markdown_text


def generate_html_content(markdown_text):
    """生成HTML内容"""
    md = markdown.Markdown(extensions=["tables"])
    html_content = md.convert(markdown_text)
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Result</title>
    <style>
        body {{ 
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0 auto;
            max-width: 800px;
            padding: 20px;
        }}
        img {{ max-width: 100%; height: auto; }}
        h1, h2, h3 {{ margin-top: 1.5em; }}
        p {{ margin: 1em 0; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""


def save_output(result, output_format, output_dir=None, output_file=None, silent=False):
    """保存输出结果"""
    if output_dir:
        output_path = Path(output_dir) / ("index.html" if output_format == "html" else "README.md")
        output_path.write_text(result)
        if not silent:
            click.echo(f"Results saved to {output_path}", err=True)
    elif output_file:
        output_path = Path(output_file)
        output_path.write_text(result)
        if not silent:
            click.echo(f"Results saved to {output_path}", err=True)
    else:
        click.echo(result)


@click.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--api-key",
    help="Mistral API key. If not provided, will use MISTRAL_API_KEY environment variable.",
    envvar="MISTRAL_API_KEY",
)
@click.option(
    "--output",
    "-o",
    help="Output file path for the result. If not provided, prints to stdout.",
    type=click.Path(),
)
@click.option(
    "--output-dir",
    "-d",
    help="Save output to a directory. For HTML creates index.html, for markdown creates README.md, with images in same directory.",
    type=click.Path(),
)
@click.option("--model", help="Mistral OCR model to use.", default="mistral-ocr-latest")
@click.option(
    "json_",
    "--json",
    "-j",
    is_flag=True,
    help="Return raw JSON instead of markdown text.",
)
@click.option(
    "--html",
    "-h",
    is_flag=True,
    help="Convert markdown to HTML.",
)
@click.option(
    "--inline-images",
    "-i",
    is_flag=True,
    help="Include images inline as data URIs (for HTML) or base64 (for JSON).",
)
@click.option(
    "--extract-images",
    "-e",
    is_flag=True,
    help="Extract images as separate files (requires --output-dir).",
)
@click.option(
    "--silent",
    "-s",
    is_flag=True,
    help="Suppress all output except for the requested data.",
)
def ocr_pdf(
    file_path,
    api_key,
    output,
    output_dir,
    model,
    json_,
    html,
    inline_images,
    extract_images,
    silent,
):
    """Process a PDF or image file using Mistral's OCR API and output the results.

    FILE_PATH is the path to the PDF or image file to process.
    Supported formats: PDF, PNG, JPG, JPEG
    
    \b
    Output Formats:
      - Markdown (default): Plain text markdown from the OCR results
      - HTML (--html): Converts markdown to HTML with proper formatting
      - JSON (--json): Raw JSON response from the Mistral OCR API

    \b
    Output Destinations:
      - stdout (default): Prints results to standard output
      - Single file (-o/--output): Writes to specified file
      - Directory (-d/--output-dir): Creates directory structure with main file and images
        * For HTML: Creates index.html in the directory
        * For Markdown: Creates README.md in the directory

    \b
    Image Handling:
      - No images (default): Images are excluded
      - Inline (-i/--inline-images): Images included as data URIs in the output
      - Extract (-e/--extract-images): Images saved as separate files in the output directory
    """
    logger.info(f"Starting OCR process for file: {file_path}")
    
    try:
        # 验证选项
        validate_options(api_key, output, output_dir, json_, extract_images, inline_images)

        # 确定输出格式
        output_format = "json" if json_ else "html" if html else "markdown"
        logger.debug(f"Output format: {output_format}")

        # 初始化客户端和文件
        input_file = Path(file_path)
        client = Mistral(api_key=api_key)
        uploaded_file = None

        try:
            # 处理文件
            is_image = input_file.suffix.lower() in ['.png', '.jpg', '.jpeg']
            include_images = inline_images or extract_images

            if not silent:
                click.echo(f"Processing {input_file.name}...", err=True)

            if is_image:
                response = process_image(client, input_file, model, include_images, silent)
                uploaded_file = None
            else:
                response, uploaded_file = process_pdf(client, input_file, model, include_images, silent)

            response_dict = json.loads(response.model_dump_json())
            logger.debug("Successfully parsed API response")

            # 创建输出目录
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created output directory: {output_dir}")

            # 处理图片
            image_map = {}
            if extract_images and output_dir:
                image_map = extract_images_to_dir(response_dict, output_dir, silent)
            elif inline_images:
                image_map = create_inline_image_map(response_dict)
                logger.debug("Created inline image map")

            # 生成输出内容
            result = generate_output_content(response_dict, output_format, image_map)
            logger.debug(f"Generated {output_format} content")

            # 保存结果
            save_output(result, output_format, output_dir, output, silent)
            logger.info("OCR process completed successfully")

        except Exception as e:
            logger.error(f"Error during OCR process: {str(e)}")
            raise click.ClickException(f"Error: {str(e)}")
        finally:
            # 清理临时文件
            try:
                if uploaded_file:
                    client.files.delete(file_id=uploaded_file.id)
                    logger.debug("Temporary file deleted")
                    if not silent:
                        click.echo("Temporary file deleted.", err=True)
            except Exception as e:
                logger.warning(f"Could not delete temporary file: {str(e)}")
                if not silent:
                    click.echo(
                        f"Warning: Could not delete temporary file: {str(e)}", err=True
                    )

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    ocr_pdf()
