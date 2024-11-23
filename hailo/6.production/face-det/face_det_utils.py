from PIL import Image, ImageDraw, ImageFont
import numpy as np


class ObjectDetectionUtils:
    def __init__(self, padding_color: tuple = (114, 114, 114), label_font: str = "LiberationSans-Regular.ttf"):
        """
        Initialize the ObjectDetectionUtils class.
        Args:
            padding_color (tuple): RGB color for padding. Defaults to (114, 114, 114).
            label_font (str): Path to the font used for labeling. Defaults to "LiberationSans-Regular.ttf".
        """
        self.padding_color = padding_color
        self.label_font = label_font

    def preprocess(self, image: Image.Image, model_w: int, model_h: int) -> Image.Image:
        """
        Resize image with unchanged aspect ratio using padding.
        """
        img_w, img_h = image.size
        scale = min(model_w / img_w, model_h / img_h)
        new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
        image = image.resize((new_img_w, new_img_h), Image.BICUBIC)

        padded_image = Image.new('RGB', (model_w, model_h), self.padding_color)
        padded_image.paste(image, ((model_w - new_img_w) // 2, (model_h - new_img_h) // 2))
        return padded_image

    def draw_detection(self, draw: ImageDraw.Draw, box: list, color: tuple, scale_factor: float):
        """
        Draw box around face detection.
        """
        ymin, xmin, ymax, xmax = box
        draw.rectangle([(xmin * scale_factor, ymin * scale_factor), (xmax * scale_factor, ymax * scale_factor)],
                       outline=color, width=2)

    def visualize(self, detections: dict, image: Image.Image, image_id: int, output_path: str, width: int, height: int,
                  min_score: float = 0.45, scale_factor: float = 1):
        """
        Visualize face detections on the image.
        """
        boxes = detections['detection_boxes']
        scores = detections['detection_scores']
        draw = ImageDraw.Draw(image)

        for idx in range(detections['num_detections']):
            if scores[idx] >= min_score:
                color = (255, 0, 0)  # Red color for face bounding boxes
                scaled_box = [x * width if i % 2 == 0 else x * height for i, x in enumerate(boxes[idx])]
                self.draw_detection(draw, scaled_box, color, scale_factor)

        image.save(f'{output_path}/output_image{image_id}.jpg', 'JPEG')

    def extract_detections(self, input_data: list, threshold: float = 0.5) -> dict:
        """
        Extract face detections.
        """
        boxes, scores = [], []
        num_detections = 0


        for detection in input_data:
            print("bbox: ", detection[:4])
            print("score: ", detection[4])
            print("scores: ", scores)
            exit(1)

            bbox, score = detection[:4], detection[4]
            if score >= threshold:
                boxes.append(bbox)
                scores.append(score)
                num_detections += 1

        return None
        # return {
        #     'detection_boxes': boxes,
        #     'detection_scores': scores,
        #     'num_detections': num_detections
        # }
