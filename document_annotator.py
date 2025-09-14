import os
import io
import base64
import json
import requests
import re
import sys
import platform
import subprocess

import mimetypes
from datetime import datetime, date
from typing import Optional, List, Type, Any, get_origin, get_args
from pydantic import BaseModel, Field, create_model
from mistralai import Mistral, TextChunk

import pytesseract
from PIL import ImageDraw, ImageFont, Image
from pdf2image import convert_from_path

from openai import OpenAI
from instructor import from_openai, Mode

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QTextEdit, QFileDialog, QLabel, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from keys import MISTRAL_API_KEY, FUELIX_API_KEY, API_KEY

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# -------------------------------------------------
# OCR Extraction
# -------------------------------------------------
def pytesseract_preprocess_file(file_path: str) -> dict:
    img = Image.open(file_path)

    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    word_bbox_map = {}

    n_boxes = len(ocr_data['level'])
    texts = []
    for i in range(n_boxes):
        word_text = ocr_data['text'][i].strip()
        if word_text:
            bbox = [
                ocr_data['left'][i],
                ocr_data['top'][i],
                ocr_data['width'][i],
                ocr_data['height'][i]
            ]
            word_bbox_map[word_text] = bbox
            texts.append(word_text)

    full_text = ' '.join(texts)
    return {'full_text': full_text, 'word_bbox_map': word_bbox_map}

def pytesseract_preprocess_pil_image(img: Image.Image) -> dict:
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    word_bbox_map = {}
    n_boxes = len(ocr_data['level'])
    texts = []

    for i in range(n_boxes):
        word_text = ocr_data['text'][i].strip()
        if word_text:
            bbox = [
                ocr_data['left'][i],
                ocr_data['top'][i],
                ocr_data['width'][i],
                ocr_data['height'][i]
            ]
            word_bbox_map[word_text] = bbox
            texts.append(word_text)

    full_text = ' '.join(texts)
    return {'full_text': full_text, 'word_bbox_map': word_bbox_map}

# -------------------------------------------------
# Document Extraction Utility
# -------------------------------------------------
class FieldDefinition(BaseModel):
    name: str
    field_type: str  # "str", "Optional[str]", "datetime", "Optional[datetime]", etc.
    description: str
    alias: Optional[str] = None

class DynamicSchema(BaseModel):
    model_name: str
    context_purpose: str
    fields: List[FieldDefinition]
    reasoning: str

class DocumentExtractionUtility:
    def __init__(self, FUELIX_API_KEY: str, schema_model: str = "gpt-4o-mini"):
        self.fuel_ix_client = from_openai(
            OpenAI(
                base_url='https://api-beta.fuelix.ai/', 
                api_key=FUELIX_API_KEY
            ),
            mode=Mode.JSON
        )
        self.schema_model = schema_model

    def _parse_field_type(self, type_str: str) -> tuple[Type, Any]:
        """Convert string type to actual Python type and ensure it's optional"""
        type_str = type_str.strip()

        # Handle Optional types
        if type_str.startswith("Optional[") and type_str.endswith("]"):
            inner_type = type_str[9:-1]  # Remove "Optional[" and "]"
            base_type, _ = self._parse_field_type(inner_type)
            return Optional[base_type], None

        # Handle basic types
        type_mapping = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "datetime": datetime,
            "date": date,
        }

        if type_str in type_mapping:
            # Always wrap in Optional to ensure all fields are optional
            return Optional[type_mapping[type_str]], None
        else:
            # Default to Optional[str] if unknown type
            return Optional[str], None

    def create_model_from_schema(self, schema: DynamicSchema) -> Type[BaseModel]:
        """Create a Pydantic model from schema definition with all fields optional"""
        fields = {}

        for field_def in schema.fields:
            field_type, default_value = self._parse_field_type(field_def.field_type)

            # Handle aliases
            field_kwargs = {}
            if field_def.alias:
                field_kwargs['alias'] = field_def.alias

            # All fields are now optional with None as default
            fields[field_def.name] = (field_type, Field(None, description=field_def.description, **field_kwargs))

        # Create the dynamic model
        DynamicModel = create_model(schema.model_name, **fields)
        return DynamicModel

    def generate_schema(self, context: str, ocr_sample: str) -> DynamicSchema:
        """Generate extraction schema based on context and document sample"""

        schema_prompt = f"""
                            Context: {context}

                            Sample Document OCR Output:
                            {ocr_sample}

                            Based on this context and document sample, define a comprehensive data extraction schema.

                            Guidelines:
                            - Use snake_case for field names
                            - Available types: str, int, float, bool, datetime, date, Optional[str], Optional[int], Optional[datetime], etc.
                            - ALL fields should be Optional[] - this is mandatory for flexible extraction
                            - Use aliases for Python keywords (e.g., "from" should have alias)
                            - Consider what fields would be most valuable for the given context
                            - Include clear descriptions for each field
                            - Avoid making big sentences as part of the output schema. We want objective, normalized (in database terms) details like list of names, address etc
                            - Remember: Every single field must be Optional to handle cases where information might be missing

                            Generate a schema with model name, context purpose, field definitions, and reasoning.
                        """
        try:
            response = self.fuel_ix_client.chat.completions.create(
                model=self.schema_model,
                response_model=DynamicSchema,
                messages=[
                    {"role": "system", "content": "You are a schema generation expert. Generate clean, practical and objective schemas for document extraction. ALL FIELDS MUST BE OPTIONAL. Examples: Optional[str] for name, Optional[int] for age, Optional[str] for phone_number, Optional[str] for pnr"},
                    {"role": "user", "content": schema_prompt}
                ],
                temperature=0,
                max_retries=5
            )
            
            # Post-process to ensure all fields are optional
            for field in response.fields:
                if not field.field_type.startswith("Optional["):
                    # Wrap non-optional types in Optional
                    field.field_type = f"Optional[{field.field_type}]"
            
            return response
        except Exception as e:
            raise Exception(f"Schema generation failed: {str(e)}")

    def extract_data(self, DynamicModel, ocr_text: str) -> BaseModel:
        """Extract data using the generated schema"""

        extraction_prompt = f"""
                                Extract the following information from this document:

                                Document OCR Text:
                                {ocr_text}

                                Extract all available fields. Use None/null for missing information.
                                All fields are optional, so don't worry if some information is not present in the document.
                            """

        try:
            extraction_response = self.fuel_ix_client.chat.completions.create(
                model=self.schema_model,
                response_model=DynamicModel,
                messages=[
                    {"role": "system", "content": "You are a data extraction expert. Extract information accurately from documents. Set fields to None if information is not available."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0,
                max_retries=5
            )
            return extraction_response
        except Exception as e:
            raise Exception(f"Data extraction failed: {str(e)}")

    def process_document(self, context: str, ocr_text: str, schema=None) -> tuple[DynamicSchema, BaseModel]:
        """Complete pipeline: generate schema and extract data"""
        if not schema:
            print("Generating extraction schema...")
            schema = self.generate_schema(context, ocr_text)

        print(f"Generated schema: {schema.model_name}")
        print(f"Fields: {len(schema.fields)} (all optional)")
        print(f"Reasoning: {schema.reasoning}")
        print()

        print("Extracting data...")
        DynamicModel = self.create_model_from_schema(schema)

        extracted_data = self.extract_data(DynamicModel, ocr_text)

        print("Data extraction complete!")

        return schema, extracted_data

def tokenize_text(text):
    return re.findall(r'[^\s]+', text.lower())

def get_bbox_for_multiword(value, word_bbox_map, full_text):
    tokens = tokenize_text(value)
    words = tokenize_text(full_text)

    for i in range(len(words) - len(tokens) + 1):
        if words[i:i+len(tokens)] == tokens:
            bboxes = []
            for token in tokens:
                for key in word_bbox_map.keys():
                    if key.lower() == token:
                        bboxes.append(word_bbox_map[key])
                        break

            if bboxes:
                xmin = min(b[0] for b in bboxes)
                ymin = min(b[1] for b in bboxes)
                xmax = max(b[0] + b[2] for b in bboxes)
                ymax = max(b[1] + b[3] for b in bboxes)
                return [xmin, ymin, xmax - xmin, ymax - ymin]
    return None

def draw_bounding_boxes(image_path: str, bboxes: dict, output_path: str = None):
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_annotated{ext}"

    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default()

    for field, bbox in bboxes.items():
        if bbox is None:
            continue
        x, y, w, h = bbox

        # Draw bounding box
        draw.rectangle([x, y, x + w, y + h], outline="red", width=1)

        # Measure text size
        try:
            text_bbox = draw.textbbox((x, y), field, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(field, font=font)

        # Label background
        text_bg = [x, y - text_height - 4, x + text_width + 4, y]
        draw.rectangle(text_bg, fill="red")

        # Label text
        draw.text((x + 2, y - text_height - 2), field, fill="white", font=font)

    # 🔑 Fix: convert RGBA → RGB if saving to JPEG
    ext = os.path.splitext(output_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        img = img.convert("RGB")

    img.save(output_path)
    print(f"Image saved with bounding boxes at {output_path}")

def draw_bounding_boxes_on_pil(pil_img, bboxes: dict):
    img = pil_img.convert("RGBA")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for field, bbox in bboxes.items():
        if bbox is None:
            continue
        x, y, w, h = bbox

        # Draw bounding box
        draw.rectangle([x, y, x + w, y + h], outline="red", width=1)

        # Measure text size with fallback
        try:
            text_bbox = draw.textbbox((x, y), field, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_width, text_height = font.getsize(field)

        # Draw label background
        text_bg = [x, y - text_height - 4, x + text_width + 4, y]
        draw.rectangle(text_bg, fill="red")

        # Draw label text
        draw.text((x + 2, y - text_height - 2), field, fill="white", font=font)

    return img

def get_bboxes_for_pii(extracted_data, word_bbox_map, full_text):
    bboxes = {}
    data_dict = extracted_data.model_dump()
    for k, v in data_dict.items():
        if not v:
            continue
        bbox = get_bbox_for_multiword(str(v), word_bbox_map, full_text)
        if bbox:
            bboxes[k] = bbox
    return bboxes

def process_pdf(file_path: str, context: str):
    pages = convert_from_path(file_path)

    base, ext = os.path.splitext(file_path)
    output_path = f"{base}_annotated{ext}"

    annotated_pages = []
    for page_img in pages:
        ocr_result = pytesseract_preprocess_pil_image(page_img)
        document_text = ocr_result['full_text']
        word_bbox_map = ocr_result['word_bbox_map']

        extractor = DocumentExtractionUtility(FUELIX_API_KEY)
        schema, extracted_data = extractor.process_document(context, document_text)

        print(extracted_data)

        bboxes = get_bboxes_for_pii(extracted_data, word_bbox_map, document_text)

        annotated_img = draw_bounding_boxes_on_pil(page_img, bboxes)
        annotated_pages.append(annotated_img.convert("RGB"))

    annotated_pages[0].save(
        output_path,
        save_all=True,
        append_images=annotated_pages[1:],
        quality=95,
    )
    print(f"Annotated PDF saved to {output_path}")

def process_img(file_path: str, context: str):
    ocr_result = pytesseract_preprocess_file(file_path)
    document_text = ocr_result['full_text']
    word_bbox_map = ocr_result['word_bbox_map']

    try:
        extractor = DocumentExtractionUtility(FUELIX_API_KEY)
        context = "PII Detection for a passenger. Need only PIIs, not verbose content."
        schema, extracted_data = extractor.process_document(context, document_text)
        bboxes = get_bboxes_for_pii(extracted_data, word_bbox_map, document_text)
        draw_bounding_boxes(file_path, bboxes)

    except Exception as e:
        print(f"Error: {e}")

def process_document(file_path: str, context: str):

    def is_image_file(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        return ext in [".png", ".jpg", ".jpeg"]

    def is_pdf_file(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        return ext == ".pdf"

    if is_image_file(file_path):
        process_img(file_path, context)
    elif is_pdf_file(file_path):
        process_pdf(file_path, context)
    else:
        print(f"Error: Unsupported file type for: {file_path}")


# -------------------------------------------------
# Execute
# -------------------------------------------------
# context = "Detect relevant PII of a passenger from the ticket."


def open_file(filepath):
    system_name = platform.system()
    try:
        if system_name == "Windows":
            os.startfile(filepath)
        elif system_name == "Darwin":
            subprocess.run(["open", filepath])
        else:
            subprocess.run(["xdg-open", filepath])
    except Exception as e:
        print(f"Failed to open file {filepath}: {e}")

class WorkerThread(QThread):
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, file_path, context):
        super().__init__()
        self.file_path = file_path
        self.context = context
        self.output_file_path = None

    def run(self):
        try:
            import builtins
            orig_print = builtins.print

            def custom_print(*args, **kwargs):
                text = ' '.join(str(a) for a in args)
                self.output_signal.emit(text)
                orig_print(*args, **kwargs)

            builtins.print = custom_print

            process_document(self.file_path, self.context)
            
            base, ext = os.path.splitext(self.file_path)
            self.output_file_path = f"{base}_annotated{ext}"

            builtins.print = orig_print
            self.finished_signal.emit(self.output_file_path)

        except Exception as e:
            self.output_signal.emit(f'Error: {str(e)}')
            self.finished_signal.emit("")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Document Extraction GUI")
        self.resize(600, 450)

        layout = QVBoxLayout()

        self.file_label = QLabel("No file selected")
        layout.addWidget(self.file_label)

        self.context_input = QLineEdit()
        self.context_input.setPlaceholderText("Enter extraction context here...")
        layout.addWidget(self.context_input)

        self.browse_button = QPushButton("Browse File")
        self.browse_button.clicked.connect(self.browse_file)
        layout.addWidget(self.browse_button)

        self.open_input_button = QPushButton("Open Input")
        self.open_input_button.setEnabled(False)
        self.open_input_button.clicked.connect(self.open_input_file)
        layout.addWidget(self.open_input_button)

        self.process_button = QPushButton("Process Document")
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setEnabled(False)
        layout.addWidget(self.process_button)

        self.open_button = QPushButton("Open Output")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self.open_output_file)
        layout.addWidget(self.open_button)

        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        layout.addWidget(self.output_area)

        self.setLayout(layout)

        self.file_path = None
        self.output_file_path = None
        self.worker_thread = None

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a file", "",
                                                   "Images and PDFs (*.png *.jpg *.jpeg *.pdf)")
        if file_path:
            self.file_path = file_path
            self.file_label.setText(f"Selected: {os.path.basename(file_path)}")
            self.process_button.setEnabled(True)
            self.open_button.setEnabled(False)
            self.open_input_button.setEnabled(True)
            self.output_area.clear()

    def start_processing(self):
        context = self.context_input.text().strip()
        if not context:
            self.output_area.append("Please enter context for extraction.")
            return
        if not self.file_path:
            self.output_area.append("Please select a file first.")
            return

        self.output_area.clear()
        self.output_area.append("Starting document processing...\n")
        self.process_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.open_button.setEnabled(False)

        self.worker_thread = WorkerThread(self.file_path, context)
        self.worker_thread.output_signal.connect(self.append_output)
        self.worker_thread.finished_signal.connect(self.on_finished)
        self.worker_thread.start()

    def append_output(self, text):
        self.output_area.append(text)

    def on_finished(self, output_file_path):
        if output_file_path and os.path.exists(output_file_path):
            self.output_file_path = output_file_path
            self.output_area.append(f"\nProcessing complete. Output saved to:\n{output_file_path}")
            self.open_button.setEnabled(True)
        else:
            self.output_area.append("\nProcessing completed but output file not found.")
            self.open_button.setEnabled(False)

        self.process_button.setEnabled(True)
        self.browse_button.setEnabled(True)

        if self.file_path and os.path.exists(self.file_path):
            self.open_input_button.setEnabled(True)
        else:
            self.open_input_button.setEnabled(False)

    def open_input_file(self):
        if self.file_path and os.path.exists(self.file_path):
            open_file(self.file_path)
        else:
            self.output_area.append("Input file is not available to open.")

    def open_output_file(self):
        if self.output_file_path and os.path.exists(self.output_file_path):
            open_file(self.output_file_path)
        else:
            self.output_area.append("Output file is not available to open.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())            