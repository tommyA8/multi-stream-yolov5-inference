import cv2
from datetime import datetime
import os
from contextlib import contextmanager

import logging
logger = logging.getLogger(__name__)

class CameraTask:
    """
    A class for performing various camera-related tasks, such as checking camera connections,
    recording videos, and saving images with annotations to the local disk.
    """
    def __init__(self) -> None:
        pass
    
    def read_rtsp_stream(self, rtsp_url: str, timeout: int = 3000):
        """
        Open an RTSP stream and return a VideoCapture object.

        Args:
            rtsp_url (str): The RTSP URL of the stream to open.
            timeout (int): The timeout in milliseconds for opening the stream.

        Returns:
            Optional[cv2.VideoCapture]: A VideoCapture object if the stream is successfully opened, None otherwise.
        """
        cap = cv2.VideoCapture()
        cap.setExceptionMode(True)
        try:
            cap.open(rtsp_url, apiPreference=cv2.CAP_FFMPEG, params=[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout])
        except cv2.error as e:
            logger.error(f"Failed to open RTSP stream {rtsp_url}: {e}")
            return None
        return cap
    
    @staticmethod
    @contextmanager
    def check_camera_connection(rtsp_url: str, timeout: int = 3000):
        """
        Check if an RTSP stream can be successfully opened.

        Args:
            rtsp_url (str): The RTSP URL of the stream to check.
            timeout (int): The timeout in milliseconds for opening the stream.

        Returns:
            Union[bool, cv2.VideoCapture]: False if the stream cannot be opened, 
                                            cv2.VideoCapture object if the stream is successfully opened.
        """
        cap = cv2.VideoCapture() # initialise empty VideoCapture class
        cap.setExceptionMode(True)
        try:
            cap.open(rtsp_url, apiPreference=cv2.CAP_FFMPEG, params=[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout])
        except TimeoutError as exc:
            raise TimeoutError(f"Camera at {rtsp_url} timed out after {timeout} ms.") from exc
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Camera at {rtsp_url} is not accessible.")
        try:
            yield cap
        finally:
            cap.release()
    
    @staticmethod
    def record_video(fps:float, resolution:tuple, info:str, save_dir=None):
        """Record video from a source to a specified directory."""
        # Determine output directory
        output_dir = save_dir if save_dir else f"pics/record/{datetime.now().strftime('%d-%b-%Y')}"
        os.makedirs(output_dir, exist_ok=True)

        output_file = f"{output_dir}/{info}-inference.mp4"
        
        logger.info("Save Video | Output file: %s", output_file)
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cap_out = cv2.VideoWriter(output_file, fourcc, fps, resolution)
        return cap_out
    
    @staticmethod
    @contextmanager
    def save_image_to_local(frame, file_name, alert_type=None, priority=None,
                            output_root=None, color=(0, 0, 0)):
        """Save an image with annotations to local disk."""
        if output_root is None:
            output_root = f"pics/alert_image/{datetime.now().strftime('%d-%b-%Y')}"
            os.makedirs(output_root, exist_ok=True)
        
        try:
            # Add text annotations
            if alert_type is not None or priority is not None:
                cv2.putText(frame, f"{alert_type}: {priority}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                # cv2.putText(frame, f"Cam: {self.camera_name[0]}", (0, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                # cv2.putText(frame, str(datetime.now()), (0, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)     
            
            # Create the file name
            img_name = f"{output_root}/{file_name}-{datetime.now().strftime('%H-%M-%S')}.jpg"

            # Save the image
            if not cv2.imwrite(img_name, frame[:, :, ::-1]):
                raise IOError("Could not write image")
            
            yield img_name  # Return the image name to the caller
            
        except Exception as e:
            logger.error("Error saving image to local: %s", e)
            yield None # Return None if an error occurs