"""A class for receiving and processing video streams from RTSP cameras."""

import asyncio
import multiprocessing
import subprocess
import threading
from contextlib import contextmanager
from typing import Dict, List, Tuple, Union
import time
import cv2
import numpy as np

# from utils.utils import CameraTask
from utils.camera_tools import CameraTask

import logging
logger = logging.getLogger(__name__)

class StreamingVideoReceiver(CameraTask):
    """
    A class for receiving and processing video streams from RTSP cameras.

    Attributes:
        rtsp_urls (Dict[str, str]): A dictionary mapping camera names to RTSP URLs.
        queues (List): A list of queues to store the received frames.
        stop_events (Dict[str, multiprocessing.Event]): A dictionary mapping camera names to stop events.
        processes (Dict[str, multiprocessing.Process]): A dictionary mapping camera names to processes.
    """
    def __init__(self, rtsp_urls: Dict[str, str], queues: List,
                 stop_events: Dict[str, multiprocessing.Event]) -> None: # type: ignore
        super().__init__()
        self.rtsp_urls = rtsp_urls
        self.queues = queues
        self.stop_events = stop_events
        self.processes = {}
    
    def __getstate__(self):
        # Copy the object's state
        state = self.__dict__.copy()
        # Remove attributes that cannot be pickled
        if 'processes' in state:
            del state['processes']
        if 'stop_events' in state:
            del state['stop_events']
        return state

    def __setstate__(self, state):
        # Restore the object's state
        self.__dict__.update(state)
        # Reinitialize attributes that were not pickled
        self.processes = {}
        self.stop_events = {}
        
    def get_video_properties(self, url: str, timeout: int = 5000) -> Tuple[float, int, int]:
        """
        Get the video properties (FPS, width, and height) of the RTSP stream.

        Args:
            url (str): The RTSP URL of the stream.
            timeout (int): The timeout in milliseconds for opening the stream.

        Returns:
            Tuple[float, int, int]: A tuple containing the FPS, width, and height of the video stream.
        """
        with self.check_camera_connection(rtsp_url=url, timeout=timeout) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return fps, width, height
    
    # @staticmethod
    @contextmanager
    def open_ffmpeg_process(self, command: List[str]):
        """
        Open an FFmpeg subprocess with a large buffer size for stdout and stderr.

        Args:
            command (List[str]): The command to run FFmpeg.

        Yields:
            subprocess.Popen: The FFmpeg subprocess.
        """
        try:
            process = subprocess.Popen(command, bufsize=10**8, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as exc:
            logger.error("Failed to open FFmpeg process: %s", exc)
        else:
            yield process

            process.wait()
            process.terminate()
            
    def read_ffmpeg_stderr(self, stderr, pid, stop_event):
        """
        Read and log FFmpeg stderr output in a separate thread.

        Args:
            stderr (IO): The stderr stream of the FFmpeg process.
            pid (int): The process ID of the FFmpeg process.
            stop_event (multiprocessing.Event): The stop event for the FFmpeg process.
        """
        while not stop_event.is_set():
            line = stderr.readline().decode('utf-8').strip()
            if line:
                logger.error("PID:%s | FFmpeg-log:%s", pid, line)
                
    async def _ffmpeg_stream_process(self, url: str, stop_event: multiprocessing.Event, # type: ignore
                                     fps: int, width: float, height: float, camera_name: str):
        """
        Start an FFmpeg process to receive the video stream from the RTSP camera.

        Args:
            url (str): The RTSP URL of the camera.
            stop_event (multiprocessing.Event): The stop event for the FFmpeg process.
            fps (int): The frames per second of the video stream.
            width (float): The width of the video stream.
            height (float): The height of the video stream.
            camera_name (str): The name of the camera.
        """
        ffmpeg_command = [
            'ffmpeg', '-rtsp_transport', 'tcp',
            '-flags', 'low_delay', '-i', url,
            '-loglevel', 'error',
            '-pix_fmt', 'bgr24', '-rtbufsize', '1M', '-f', 'rawvideo', 'pipe:1',
            '-err_detect', 'ignore_err'
        ]
        with self.open_ffmpeg_process(command=ffmpeg_command) as process:
            for queue in self.queues:
                logger.info("Start Capture Stream | PID:%s | Queue-ID:%s | Camera-Name:%s | FPS:%d | Resolution:%dx%d",
                    process.pid, id(queue), camera_name, fps, width, height)

            stderr_thread = threading.Thread(target=self.read_ffmpeg_stderr,
                                             args=(process.stderr, process.pid, stop_event))
            stderr_thread.start()

            while not stop_event.is_set():
                try:
                    raw_frame = process.stdout.read(int(width * height * 3))
                    if len(raw_frame) != int(width * height * 3) or not raw_frame:
                        logger.error(f"[{camera_name}] - Incomplete frame read")
                        raise ValueError("Incomplete frame read")

                    frame = np.frombuffer(raw_frame, np.uint8).reshape((int(height), int(width), 3))

                    for queue in self.queues:
                        if not queue.full():
                            queue.put((camera_name, frame))
                            
                    # print(f"queue-{id(self.queues[0])}: {self.queues[0].qsize()}")
                    # print(f"queue-{id(self.queues[1])}: {self.queues[1].qsize()}")
                    
                    del raw_frame
                    del frame

                except (ValueError, Exception) as exc:
                    logger.error("PID: %s | %s | %s", process.pid, camera_name, exc)
                    stop_event.set()
                
                # time.sleep(1/fps)
                
            stop_event.set()
            stderr_thread.join()

            process.wait()
            process.terminate()

    async def capture_stream(self, camera_name: str, url: str, stop_event: multiprocessing.Event): # type: ignore
        """
        Capture the video stream from the RTSP camera and put the frames into the queues.

        Args:
            camera_name (str): The name of the camera.
            url (str): The RTSP URL of the camera.
            stop_event (multiprocessing.Event): The stop event for the FFmpeg process.
        """
        while True:
            try:
                fps, width, height = self.get_video_properties(url, timeout=5000)
                
                await self._ffmpeg_stream_process(
                    url=url, stop_event=stop_event, fps=int(fps),
                    width=width, height=height, camera_name=camera_name
                )
                
                logger.error("Lost connection to camera '%s'. Reconnecting...", camera_name)
                stop_event = multiprocessing.Event()
                self.stop_events[camera_name] = stop_event

            except KeyboardInterrupt:
                break

            except Exception:
                logger.error("Failed to connect to camera '%s'. Reconnecting in 30 seconds...", camera_name)
                await asyncio.sleep(30)
    
    async def capture_video_file(self, video_name: str, file_path: str, stop_event):
        """
        Capture the video from a file and put the frames into the queues.

        Args:
            file_path (str): The path to the video file.
            stop_event (multiprocessing.Event): The stop event for the FFmpeg process.
        """
                
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        ffmpeg_command = [
            'ffmpeg', '-i', file_path,
            '-loglevel', 'error',
            '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'
        ]
        
        with self.open_ffmpeg_process(command=ffmpeg_command) as process:
            for queue in self.queues:
                logger.info("Start Capture Stream | PID:%s | Queue-ID:%s | Camera-Name:%s | FPS:%d | Resolution:%dx%d",
                    process.pid, id(queue), video_name, fps, width, height)

            
            stderr_thread = threading.Thread(target=self.read_ffmpeg_stderr,
                                             args=(process.stderr, process.pid, stop_event))
            stderr_thread.start()
            
            frame_count = 0
            start_time = time.time()
            
            while not stop_event.is_set():
                try:
                    raw_frame = process.stdout.read((width * height * 3))
                                        
                    if len(raw_frame) != int(width * height * 3) or not raw_frame:
                        logger.error(f"[{video_name}] - Incomplete frame read")
                        raise ValueError("Incomplete frame read")

                    frame = np.frombuffer(raw_frame, np.uint8).reshape((int(height), int(width), 3))

                    for queue in self.queues:
                        if not queue.full():
                            queue.put((video_name, frame))
                            
                    # print(f"queue-{id(self.queues[0])}: {self.queues[0].qsize()}")
                    # print(f"queue-{id(self.queues[1])}: {self.queues[1].qsize()}")
                            
                    # current_time = time.time()
                    
                    # Update frame count and log FPS
                    # frame_count += 1
                    # if current_time - start_time >= 1:
                    #     # Calculate latency
                    #     print(f"PID: {process.pid} | video-name: {video_name} | Source-FPS:{fps} | Process-FPS:{frame_count}")
                    #     frame_count = 0
                    #     start_time = current_time
                    
                    # time.sleep(1/fps)
                         
                    del raw_frame
                    del frame

                except (ValueError, Exception) as exc:
                    logger.error("PID: %s | %s", process.pid, exc)
                    stop_event.set()
                    break

            stop_event.set()
            stderr_thread.join()

            process.wait()
            process.terminate()
            
    def run_async_capture_stream(self, camera_name, url, stop_event):
        """
        Run the capture stream asynchronously.

        Args:
            camera_name (str): The name of the camera.
            url (str): The RTSP URL of the camera.
            stop_event (multiprocessing.Event): The stop event for the FFmpeg process.
        """
        asyncio.run(self.capture_stream(camera_name, url, stop_event))
    
    def run_async_capture_video_file(self, video_name, file_path, stop_event):
        """
        Run the capture video file asynchronously.

        Args:
            file_path (str): The path to the video file.
            stop_event (multiprocessing.Event): The stop event for the FFmpeg process.
        """
        asyncio.run(self.capture_video_file(video_name, file_path, stop_event))
                                       
    def start(self):
        """
        Start streaming video from the RTSP cameras or video files.
        """
        for name, url in self.rtsp_urls.items():
            stop_event = multiprocessing.Event()
            self.stop_events[name] = stop_event
            if url.startswith("rtsp://"):
                p = multiprocessing.Process(target=self.run_async_capture_stream, 
                                            args=(name, url, stop_event))
            else:
                p = multiprocessing.Process(target=self.run_async_capture_video_file, 
                                            args=(name, url, stop_event))
            self.processes[name] = p
            p.start()
        
    def stop(self):
        """
        Stop streaming video from the RTSP cameras.
        """
        for name, stop_event in self.stop_events.items():
            stop_event.set()
            p = self.processes[name]
            if p.is_alive():
                p.terminate()
                p.join()
                        
if __name__ == "__main__":
    from settings import Settings
    import gc
    import time
    
    logging.basicConfig(level=logging.INFO)
        
    settings = Settings(_env_file=".env")

    rtsp_urls = {
        'test-person-1': settings.VIDEO_PERSON_1,
        'test-person-2': settings.VIDEO_PERSON_2,
        'test-fire': settings.VIDEO_FIRE,
        # "CHARAN94_1-01": settings.CHARAN94_1_01,
        # "CHARAN94_1-02": settings.CHARAN94_1_02,
        # "CHARAN94_1-03": settings.CHARAN94_1_03,
        # "CHARAN94_1-04": settings.CHARAN94_1_01,
        # "CHARAN94_1-05": settings.CHARAN94_1_02,
        # "CHARAN94_1-06": settings.CHARAN94_1_03
    }
    # Single shared queue
    tracking_frame_queue = multiprocessing.Queue(maxsize=100)
    detection_frame_queue = multiprocessing.Queue(maxsize=100)
    
    # receiving multi streaming video via RTSP
    stream_events = {}
    receiver = StreamingVideoReceiver(
        rtsp_urls=rtsp_urls, 
        queues=[tracking_frame_queue, detection_frame_queue], 
        stop_events={}
    )
    
    try:
        # Run for a limited time for demonstration, then stop
        receiver.start_streaming()
        time.sleep(120)  # Run for 30 seconds
    except KeyboardInterrupt:
        logger.info("User Interrupt Capture Stream")
    finally:
        receiver.stop_streaming()
        gc.collect()
        logger.info("All camera streams have been processed.")