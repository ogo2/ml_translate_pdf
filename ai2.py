import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from PIL import Image
import io
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import json

@dataclass
class VisualElement:
    type: str  # 'image', 'table', 'diagram', 'line', 'other'
    bbox: Tuple[float, float, float, float]
    page_num: int
    data: Any  # Содержимое элемента
    properties: Dict  # Дополнительные свойства

class PDFVisualExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.visual_elements = []

    def extract_images(self, page_num: int) -> List[VisualElement]:
        """Извлекает изображения со страницы."""
        page = self.doc[page_num]
        images = []
        
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = self.doc.extract_image(xref)
            image_data = base_image["image"]
            
            # Получаем положение изображения
            image_rect = page.get_image_bbox(img)
            
            images.append(VisualElement(
                type='image',
                bbox=tuple(image_rect),
                page_num=page_num,
                data=image_data,
                properties={
                    'format': base_image["ext"],
                    'colorspace': base_image.get("colorspace", ""),
                    'xref': xref
                }
            ))
        return images

    def extract_tables(self, page_num: int) -> List[VisualElement]:
        """Извлекает таблицы со страницы."""
        page = self.doc[page_num]
        tables = []
        
        # Поиск таблиц по линиям и структуре
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") == 1:  # Проверяем, является ли блок таблицей
                tables.append(VisualElement(
                    type='table',
                    bbox=tuple(block["bbox"]),
                    page_num=page_num,
                    data=block,
                    properties={'cells': block.get("lines", [])}
                ))
        return tables

    def extract_lines(self, page_num: int) -> List[VisualElement]:
        """Извлекает линии разметки со страницы."""
        page = self.doc[page_num]
        paths = page.get_drawings()
        lines = []
        
        for path in paths:
            if path["type"] == "l":  # линия
                lines.append(VisualElement(
                    type='line',
                    bbox=tuple(path["rect"]),
                    page_num=page_num,
                    data=path,
                    properties={
                        'color': path.get("color", (0, 0, 0)),
                        'width': path.get("width", 1),
                        'style': path.get("style", "solid")
                    }
                ))
        return lines

    def extract_all_elements(self):
        """Извлекает все визуальные элементы из PDF."""
        for page_num in range(len(self.doc)):
            self.visual_elements.extend(self.extract_images(page_num))
            self.visual_elements.extend(self.extract_tables(page_num))
            self.visual_elements.extend(self.extract_lines(page_num))
        
        return self.visual_elements

    def save_elements_data(self, output_path: str):
        """Сохраняет данные о визуальных элементах в JSON."""
        elements_data = []
        for elem in self.visual_elements:
            elem_dict = {
                'type': elem.type,
                'bbox': elem.bbox,
                'page_num': elem.page_num,
                'properties': elem.properties
            }
            if elem.type == 'image':
                # Сохраняем изображение отдельно
                img_path = f"{output_path}_images/{len(elements_data)}.{elem.properties['format']}"
                with open(img_path, 'wb') as f:
                    f.write(elem.data)
                elem_dict['data'] = img_path
            elements_data.append(elem_dict)
            
        with open(f"{output_path}_elements.json", 'w') as f:
            json.dump(elements_data, f, indent=2)

class PDFReconstructor:
    def __init__(self, elements_data: List[VisualElement], translated_text: Dict):
        self.elements = elements_data
        self.translated_text = translated_text

    def create_pdf(self, output_path: str):
        """Создает новый PDF с переведенным текстом и оригинальными визуальными элементами."""
        c = canvas.Canvas(output_path, pagesize=letter)
        
        current_page = 0
        page_elements = [e for e in self.elements if e.page_num == current_page]
        
        # Добавляем визуальные элементы
        for elem in page_elements:
            if elem.type == 'image':
                self._add_image(c, elem)
            elif elem.type == 'line':
                self._add_line(c, elem)
            elif elem.type == 'table':
                self._add_table(c, elem)
        
        # Добавляем переведенный текст
        if current_page in self.translated_text:
            for text_block in self.translated_text[current_page]:
                c.drawString(text_block['x'], text_block['y'], text_block['text'])
        
        c.save()

    def _add_image(self, canvas, element):
        """Добавляет изображение в PDF."""
        if isinstance(element.data, bytes):
            img = Image.open(io.BytesIO(element.data))
            canvas.drawImage(img, *element.bbox)

    def _add_line(self, canvas, element):
        """Добавляет линию в PDF."""
        canvas.setStrokeColor(element.properties['color'])
        canvas.setLineWidth(element.properties['width'])
        canvas.line(*element.bbox)

    def _add_table(self, canvas, element):
        """Добавляет таблицу в PDF."""
        # Реализация добавления таблицы
        pass