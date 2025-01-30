import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from transformers import MarianTokenizer, MarianMTModel
import torch
import io
from reportlab.lib.utils import ImageReader
from PIL import Image
import logging
import os
from datetime import datetime
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

@dataclass
class VisualElement:
    type: str  # 'image', 'table', 'line', 'shape'
    bbox: Tuple[float, float, float, float]
    data: Any
    page_num: int
    properties: Dict

class CustomPDFTranslator:
    def __init__(self, model_path: str, input_pdf_path: str, output_pdf_path: str):
        """
        Инициализация транслятора с пользовательской моделью
        
        Args:
            model_path (str): Путь к обученной модели
            input_pdf_path (str): Путь к входному PDF
            output_pdf_path (str): Путь для сохранения переведенного PDF
        """
        self.input_pdf_path = input_pdf_path
        self.output_pdf_path = self._get_safe_output_path(output_pdf_path)
        self.visual_elements = []
        
        # Загрузка модели
        logger.info("Загрузка пользовательской модели перевода...")
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(model_path)
            self.model = MarianMTModel.from_pretrained(model_path)
            
            # Переключаем модель в режим оценки (не обучения)
            self.model.eval()
            
            # Если есть GPU, используем его
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Модель загружена на GPU")
            else:
                logger.info("Модель загружена на CPU")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}")
            raise

        # Инициализация шрифтов
        self._initialize_fonts()

    def _get_safe_output_path(self, desired_path: str) -> str:
        """Получение безопасного пути для сохранения файла"""
        try:
            directory = os.path.dirname(desired_path) or '.'
            filename = os.path.basename(desired_path)
            
            os.makedirs(directory, exist_ok=True)
            
            output_path = desired_path
            counter = 0
            
            while os.path.exists(output_path):
                name, ext = os.path.splitext(filename)
                counter += 1
                output_path = os.path.join(directory, f"{name}_{counter}{ext}")
            
            return output_path
            
        except (PermissionError, OSError):
            temp_dir = os.path.join(os.path.expanduser("~"), "Documents")
            os.makedirs(temp_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return os.path.join(temp_dir, f"translated_pdf_{timestamp}.pdf")

    def _initialize_fonts(self):
        """Инициализация шрифтов с поддержкой кириллицы"""
        try:
            fonts_to_try = [
                ('Arial-Unicode', 'arial-unicode-ms.ttf'),
                ('DejaVuSerif', 'DejaVuSerif.ttf'),
                ('DejaVuSans', 'DejaVuSans.ttf'),
                ('FreeSans', 'FreeSans.ttf')
            ]
            
            self.available_fonts = []
            for font_name, font_file in fonts_to_try:
                try:
                    pdfmetrics.registerFont(TTFont(font_name, font_file))
                    self.available_fonts.append(font_name)
                    logger.info(f"Зарегистрирован шрифт: {font_name}")
                except:
                    continue
            
            if not self.available_fonts:
                raise Exception("Не найдено шрифтов с поддержкой кириллицы")
            
            self.default_font = self.available_fonts[0]
            
        except Exception as e:
            logger.error(f"Ошибка инициализации шрифтов: {str(e)}")
            raise

    def translate_text(self, text: str) -> str:
        """ 
        Перевод текста с помощью пользовательской модели
        
        Args:
            text (str): Исходный текст
            
        Returns:
            str: Переведенный текст
        """
        if not text.strip():
            return text

        try:
            # Подготавливаем текст
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Если есть GPU, переносим данные на него
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Отключаем подсчет градиентов для ускорения
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
            
            # Декодируем результат
            translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated_text
            
        except Exception as e:
            logger.error(f"Ошибка перевода: {str(e)}")
            return text

    def extract_text_with_info(self) -> List[Dict]:
        """Извлечение текста и форматирования из PDF"""
        text_blocks = []
        
        try:
            doc = fitz.open(self.input_pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text_blocks.append({
                                    'text': span['text'].strip(),
                                    'bbox': span['bbox'],
                                    'font_size': span['size'],
                                    'font_name': span['font'],
                                    'color': span['color'],
                                    'page_num': page_num,
                                    'origin_width': span['bbox'][2] - span['bbox'][0]
                                })
            
            return text_blocks
            
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из PDF: {str(e)}")
            raise

    def get_text_width(self, text: str, font_name: str, font_size: float) -> float:
        """Вычисление ширины текста"""
        try:
            face = pdfmetrics.getFont(font_name).face
            return face.stringWidth(text, font_size)
        except:
            return len(text) * font_size * 0.6

    def _wrap_text_to_width(self, text: str, max_width: float, font_name: str, font_size: float) -> List[str]:
        """Разбиение текста на строки с учётом максимальной ширины"""
        words = text.split()
        lines = []
        current_line = []
        current_width = 0

        for word in words:
            word_width = self.get_text_width(word, font_name, font_size)
            
            if current_width + word_width <= max_width:
                current_line.append(word)
                current_width += word_width + self.get_text_width(" ", font_name, font_size)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_width = word_width

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def extract_visual_elements(self):
        """Извлекает изображения, линии и настоящие таблицы из PDF"""
        doc = fitz.open(self.input_pdf_path)
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Извлекаем изображения
                for img in page.get_images(full=True):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    if "image" not in base_image:
                        continue
                    image_bytes = base_image["image"]
                    image_rect = page.get_image_bbox(img)
                    self.visual_elements.append(VisualElement(
                        type='image',
                        bbox=image_rect,
                        data=image_bytes,
                        page_num=page_num,
                        properties={'format': base_image["ext"]}
                    ))
                    logger.info(f"Добавлено изображение на страницу {page_num}")
                
                # Извлекаем возможные линии через текстовые блоки
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    logger.info(f"Блок на странице {page_num}: {block}")
                    bbox = block.get("bbox", None)
                    lines = block.get("lines", [])
                    # Если блок НЕ содержит текста, но имеет границы - это может быть линия
                    if bbox and not lines:
                        x0, y0, x1, y1 = bbox
                        width = abs(x1 - x0)
                        height = abs(y1 - y0)
                        # Линии обычно имеют маленькую высоту (например, < 3 пикселей)
                        if height < 3:  
                            self.visual_elements.append(VisualElement(
                                type='line',
                                bbox=bbox,
                                data=None,
                                page_num=page_num,
                                properties={'color': (0, 0, 0), 'width': height}
                            ))
                            logger.info(f"Обнаружена линия через text_blocks() на странице {page_num}, bbox={bbox}")
                #Проверка графических примитивов
                for drawing in page.get_drawings():
                    logger.info(f"Графический примитив на странице {page_num}: type={drawing['type']}, items={drawing['items']}")
                    
                    if drawing["type"] in ("l", "s"):  # Линии и штрихи
                        x0, y0 = drawing["rect"].x0, drawing["rect"].y0
                        x1, y1 = drawing["rect"].x1, drawing["rect"].y1
                        color = drawing.get("color", (0, 0, 0))
                        width = drawing.get("width", 1)
                        
                        # Добавляем линию в список visual_elements
                        self.visual_elements.append(VisualElement(
                            type='line',
                            bbox=(x0, y0, x1, y1),
                            data=None,
                            page_num=page_num,
                            properties={'color': color, 'width': width}
                        ))
                        logger.info(f"Обнаружена линия через get_drawings() на странице {page_num}, bbox={(x0, y0, x1, y1)}")
                    
                    elif drawing["type"] == "f":  # Заливки (например, прямоугольники)
                        x0, y0 = drawing["rect"].x0, drawing["rect"].y0
                        x1, y1 = drawing["rect"].x1, drawing["rect"].y1
                        fill_color = drawing.get("fill", (0, 0, 0))
                        
                        # Добавляем заливку в список visual_elements
                        self.visual_elements.append(VisualElement(
                            type='fill',
                            bbox=(x0, y0, x1, y1),
                            data=None,
                            page_num=page_num,
                            properties={'fill_color': fill_color}
                        ))
                        logger.info(f"Обнаружена заливка через get_drawings() на странице {page_num}, bbox={(x0, y0, x1, y1)}")
                # Проверяем аннотированные линии (если они есть)
                for annot in page.annots():
                    logger.info(f"Аннотация на странице {page_num}: {annot.type}")
                    if annot.type[0] == 8:  # PDF "Line Annotation"
                        bbox = annot.rect
                        self.visual_elements.append(VisualElement(
                            type='line',
                            bbox=(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
                            data=None,
                            page_num=page_num,
                            properties={'color': annot.colors.get("stroke", (0, 0, 0)), 'width': annot.border[0]}
                        ))
                        logger.info(f"Добавлена аннотированная линия на странице {page_num}, bbox={bbox}")
                        
                logger.info(f"Всего элементов в visual_elements: {len(self.visual_elements)}")
                logger.info(f"Линий в visual_elements: {sum(1 for e in self.visual_elements if e.type == 'line')}")
                logger.info(f"Таблиц в visual_elements: {sum(1 for e in self.visual_elements if e.type == 'table')}")
        finally:
            doc.close()

    def create_translated_pdf(self, text_blocks: List[Dict]):
        """Создание нового PDF с переведённым текстом и визуальными элементами"""
        try:
            c = canvas.Canvas(self.output_pdf_path, pagesize=A4)

            current_page = -1  # Начинаем с -1, чтобы гарантированно сработал первый переход
            logger.info(f"Всего элементов в visual_elements: {len(self.visual_elements)}")
            logger.info(f"Линий в visual_elements: {sum(1 for e in self.visual_elements if e.type == 'line')}")
            logger.info(f"Таблиц в visual_elements: {sum(1 for e in self.visual_elements if e.type == 'table')}")


            for block in text_blocks:
                page_num = block['page_num']

                # Переход на новую страницу
                if page_num != current_page:  # Теперь проверяем не только >, но и любой переход
                    if current_page >= 0:  # Показываем страницу, если уже была
                        c.showPage()

                    current_page = page_num

                    # Вставляем ВСЕ визуальные элементы для текущей страницы (всегда!)
                    for elem in self.visual_elements:
                        if elem.page_num == current_page:
                            logger.info(f"Добавление {elem.type} на страницу {current_page}")
                            self._add_visual_element(c, elem)

                # Пропускаем пустые блоки
                if not block['text'].strip():
                    continue

                # Перевод текста
                translated_text = self.translate_text(block['text'])
                original_font_size = block['font_size']
                adjusted_font_size = original_font_size * 0.7
                font_name = self.default_font

                c.setFont(font_name, adjusted_font_size)

                # Конвертируем цвет
                color = block['color']
                if isinstance(color, int):
                    color = (
                        ((color >> 16) & 255) / 255,
                        ((color >> 8) & 255) / 255,
                        (color & 255) / 255
                    )
                c.setFillColorRGB(*color)

                # Разбиваем текст на строки
                available_width = block['origin_width']
                wrapped_lines = self._wrap_text_to_width(translated_text, available_width, font_name, adjusted_font_size)

                # Отрисовка текста
                x, y = block['bbox'][0], A4[1] - block['bbox'][1]
                line_height = adjusted_font_size * 1.2
                total_height = line_height * (len(wrapped_lines) - 1)
                y_offset = total_height / 2

                for i, line in enumerate(wrapped_lines):
                    y_pos = y - y_offset + (i * line_height)
                    c.drawString(x, y_pos, line)

            c.save()
            logger.info(f"Переведённый PDF сохранён: {self.output_pdf_path}")

        except Exception as e:
            logger.error(f"Ошибка создания PDF: {str(e)}")
            raise


    def _add_visual_element(self, c, elem: VisualElement):
        """Добавление визуального элемента в PDF (изображения, линии, таблицы)"""
        try:
            if elem.type == 'image':
                # Конвертируем байты в PIL Image
                image_bytes = elem.data
                if not isinstance(image_bytes, bytes):
                    logger.warning(f"Некорректные данные изображения на странице {elem.page_num}")
                    return
                img = Image.open(io.BytesIO(image_bytes))
                # Определяем координаты и размеры
                x0, y0, x1, y1 = elem.bbox
                width = x1 - x0
                height = y1 - y0
                # Корректируем координаты (PDF и ReportLab используют разные системы)
                y_pos = A4[1] - y1
                # Вставляем изображение
                c.drawImage(
                    ImageReader(img),
                    x0,
                    y_pos,
                    width=width,
                    height=height,
                    mask='auto'
                )
            elif elem.type == 'line':
                logger.info(f"Рисуем линию на странице {elem.page_num}, bbox={elem.bbox}")
                
                x0, y0, x1, y1 = elem.bbox
                if x0 == x1 and y0 == y1:
                    logger.warning(f"Линия на странице {elem.page_num} имеет нулевую длину: bbox={elem.bbox}")
                    return
                
                y0 = A4[1] - y0
                y1 = A4[1] - y1
                color = elem.properties.get('color', (0, 0, 0))
                width = elem.properties.get('width', 1)
                if not all(0 <= c <= 1 for c in color):
                    logger.warning(f"Некорректный цвет линии: {color}")
                    color = (0, 0, 0)  # Устанавливаем черный цвет по умолчанию
                if width <= 0:
                    logger.warning(f"Некорректная ширина линии: {width}")
                    width = 1  # Устанавливаем ширину 1 по умолчанию
                logger.info(f"Линия на странице {elem.page_num}: bbox={elem.bbox}, color={color}, width={width}")
                
                c.setStrokeColorRGB(*color)
                c.setLineWidth(width)
                c.line(x0, y0, x1, y1)
            elif elem.type == 'fill':
                logger.info(f"Рисуем заливку на странице {elem.page_num}, bbox={elem.bbox}")
                
                x0, y0, x1, y1 = elem.bbox
                fill_color = elem.properties.get('fill_color', (0, 0, 0))
                
                c.setFillColorRGB(*fill_color)
                c.rect(x0, A4[1] - y1, x1 - x0, y1 - y0, stroke=0, fill=1)
            elif elem.type == 'table':
                logger.info(f"Рисуем таблицу на странице {elem.page_num}, bbox={elem.bbox}, ячеек={len(elem.properties.get('cells', []))}")
                try:
                    logger.info(f"Рисуем таблицу на странице {elem.page_num}")
                    # Извлекаем данные таблицы
                    table_data = elem.properties.get("cells", [])
                    if not table_data:
                        logger.warning(f"Таблица на странице {elem.page_num} не содержит данных")
                        return
                    # Координаты таблицы
                    x0, y0, x1, y1 = elem.bbox
                    table_width = x1 - x0
                    table_height = y1 - y0
                    y0 = A4[1] - y0  # Перевод координат
                    y1 = A4[1] - y1
                    # Проверяем количество строк и столбцов
                    row_count = len(table_data)
                    col_count = max(len(row) for row in table_data) if row_count > 0 else 0
                    if row_count == 0 or col_count == 0:
                        logger.warning(f"Пустая таблица на странице {elem.page_num}")
                        return
                    row_height = table_height / row_count
                    col_width = table_width / col_count
                    # Устанавливаем цвет и толщину линий
                    c.setStrokeColorRGB(0, 0, 0)  # Чёрный цвет
                    c.setLineWidth(2)
                    # Рисуем строки
                    for i in range(row_count + 1):
                        y = y0 - (i * row_height)
                        c.line(x0, y, x1, y)
                    # Рисуем столбцы
                    for j in range(col_count + 1):
                        x = x0 + (j * col_width)
                        c.line(x, y0, x, y1)
                    # Заполняем ячейки текстом
                    c.setFont("Helvetica", 8)
                    for i, row in enumerate(table_data):
                        for j, cell in enumerate(row):
                            if isinstance(cell, str):
                                x = x0 + (j * col_width) + 2
                                y = y0 - (i * row_height) - (row_height / 2) + 2
                                c.drawString(x, y, cell)
                except Exception as e:
                    logger.error(f"Ошибка при отрисовке таблицы на странице {elem.page_num}: {str(e)}")
        except Exception as e:
            logger.error(f"Ошибка при добавлении изображения: {str(e)}")


def main():
    try:
        # Пути к файлам
        model_path = "final_model"  # путь к обученной модели
        input_pdf = "pdf_original/test3.pdf"       # входной файл
        output_pdf = "pdf_translate/output_ru.pdf"  # выходной файл
        
        # Создаём транслятор
        translator = CustomPDFTranslator(model_path, input_pdf, output_pdf)
        
        # Извлекаем текст
        logger.info("Извлечение текста из PDF...")
        text_blocks = translator.extract_text_with_info()
        
        # Извлекаем визуальные элементы
        logger.info("Извлечение визуальных элементов из PDF...")
        translator.extract_visual_elements()
        
        # Создаём переведенный PDF
        logger.info("Создание переведённого PDF...")
        translator.create_translated_pdf(text_blocks)
        
        logger.info("Перевод PDF успешно завершён!")
        
    except Exception as e:
        logger.error(f"Ошибка в процессе перевода: {str(e)}")

if __name__== "__main__":
    main()