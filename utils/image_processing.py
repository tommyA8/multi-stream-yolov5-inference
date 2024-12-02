import cv2
import base64
from typing import Union
import numpy as np

import logging
logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    A class for processing image frames, including encoding, drawing bounding boxes and text,
    adjusting exposure, and reducing highlights.
    """

    def __init__(self) -> None:
        self.size: Union[tuple, None] = None
        self.sharped: Union[bool, None] = None
        self.exposure: Union[float, None] = None
        self.reduction_factor: Union[float, None] = None
    
    @staticmethod
    def encoding_img(frame):
        """Encode the image frame to base64 JPEG format."""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            return base64.b64encode(buffer).decode("utf8")
        except Exception as exc:
            logger.error("Error encoding image: %s", exc)
            raise

    @staticmethod
    def plotting(frame, bbox: list, color: tuple, text: str, font_size: float, box_thickness: int, track_id=None):
        """
        Draw a bounding box and text on the frame.

        Args:
            frame: The image frame.
            bbox: List containing the coordinates of the bounding box [x1, y1, x2, y2].
            color: Color of the bounding box and text background.
            text: The text to display.
            font_size: Font size of the text.
            box_thickness: Thickness of the bounding box.
            track_id: Optional tracking ID to display.

        Returns:
            The frame with the bounding box and text drawn on it.
        """
        text_color = (255, 255, 255)
        x1, y1, x2, y2 = bbox
        text_w, text_h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]

        if track_id:
            y_offset = text_h + 5 # - 60
            x_offset = text_w
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            # Draw the filled rectangle for text background
            cv2.rectangle(frame, (x1, y1), (x1 + x_offset, y1 - y_offset), color, -1)
            # Draw the text on the filled rectangle
            cv2.putText(frame, text, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, 2)
            # Optionally, draw the track ID
            # cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, thickness)
        else:
            y_offset = text_h - 35
            x_offset = text_w
            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            # Draw the filled rectangle for text background
            cv2.rectangle(frame, (x1, y1), (x1 + x_offset, y1 + y_offset), color, -1)
            # Draw the text on the filled rectangle
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, 2) 
        return frame
    
    def adjust_exposure(self, image, gamma=0.65):
        """Adjust the exposure of the image using gamma correction."""
        inv_gamma = 1 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 \
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def reduce_highlights(self, image, threshold_value=200, reduction_factor=0.75):
        """
        Reduce highlights in the image.

        Parameters:
        - image: Input image in BGR format.
        - threshold_value: Threshold value to determine highlights (0-255). Default is 200.
        - reduction_factor: Factor to reduce the intensity of highlights (0-1). Default is 0.5.

        Returns:
        - Image with reduced highlights.
        """
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Split the HSV channels
        h, s, v = cv2.split(hsv)

        # Define the threshold for highlights
        threshold_value = 200  # Adjust this value as needed
        _, highlight_mask = cv2.threshold(v, threshold_value, 255, cv2.THRESH_BINARY)

        # Reduce the intensity of the highlights
        v[highlight_mask > 0] = v[highlight_mask > 0] * reduction_factor

        # Merge the HSV channels back
        hsv_reduced_highlight = cv2.merge([h, s, v])

        # Convert back to BGR color space
        return cv2.cvtColor(hsv_reduced_highlight, cv2.COLOR_HSV2BGR)
    def image_processing(self, image, cvt=cv2.COLOR_BGR2RGB, inter=cv2.INTER_AREA):
        """
        Process the input image by applying various operations.

        Args:
            image: Input image in BGR format.
            cvt: Color space conversion code. Default is cv2.COLOR_BGR2RGB.
            inter: Interpolation method for resizing. Default is cv2.INTER_AREA.

        Returns:
            Processed image in the specified color space.
        """
        if self.size:
            image = cv2.resize(image, self.size, interpolation=inter)
        if self.exposure:
            image = self.adjust_exposure(image, gamma=self.exposure)
        if self.reduction_factor:
            image = self.reduce_highlights(image, threshold_value=150, 
                                           reduction_factor=self.reduction_factor)      
        if self.sharped:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) # Create the sharpening kernel
            image = cv2.filter2D(image, -1, kernel) # Sharpen the image
            
        return cv2.cvtColor(image, cvt)