import os
import multiprocessing
from collections import defaultdict

import cv2
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Union, Any

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from tracking.yolox.tracker.byte_tracker import BYTETracker

from utils.camera_tools import CameraTask
from utils.image_processing import ImageProcessor

import os
from dotenv import load_dotenv

load_dotenv()

TRACKING_FPS = os.getenv('TRACKING_FPS')
# GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
# SERVICE_ACCOUNT_FILE = os.getenv('SERVICE_ACCOUNT_FILE')
# STORAGE_BUCKET_NAME = os.getenv('STORAGE_BUCKET_NAME')

import logging
logger = logging.getLogger(__name__)

class BYTETrackerArgs:
    track_thresh = 0.5 # High_threshold
    track_buffer = 40 # Number of frame lost tracklets are kept
    match_thresh = 0.8 # Matching threshold for first stage linear assignment
    aspect_ratio_thresh = 10.0 # Minimum bounding box aspect ratio
    min_box_area = 1.0 # Minimum bounding box area
    mot20 = False # If used, bounding boxes are not clipped.

class ModelRunner(ImageProcessor, CameraTask):
    def __init__(self, 
                 input_queue: multiprocessing.Queue, 
                 output_queue: multiprocessing.Queue, 
                 stop_events: Dict[str, multiprocessing.Event] # type: ignore
                 ) -> None:
        ImageProcessor.__init__(self)
        # queue and processes management
        self.input_queue = input_queue  # shared queue with receiver
        self.output_queue = output_queue  # shared queue with intrusion checker
        self.stop_events = stop_events
        self.processes = {}
        # inference job
        self.model_path: Union[str, None] = None
        self.conf: Union[float, None] = None
        self.iou_thres: Union[float, None] = None
        self.model: Union[torch.nn.Module, None] = None
        self.sahi_model = None
        self.device: Union[torch.device, None] = None
        self.tracker = {}
        # Initialize IDs_history and IDs_duration as defaultdicts of dicts to separate per camera
        self.IDs_history = defaultdict(dict)
        self.IDs_duration = defaultdict(dict)
        self.colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        self.roi_cameras = {
            'Charan94_1-01': 180,
            'Charan94_1-02': 180,
            'Charan94_1-03': 200,
            'Charan94_1-04': 100,
            'Charan94_1-05': 130,
            'Charan94_1-06': 200,
        }
        
    @staticmethod
    def not_present() -> str: return "Not Present"
    
    def _tracking_id(self, camera_name: str, track_id: int, timestamp: datetime) -> Union[timedelta, None]:
        # Use camera_name to separate IDs_history per camera
        if track_id not in self.IDs_history[camera_name]:
            self.IDs_history[camera_name][track_id] = [timestamp, timestamp]
            return None
        else:
            self.IDs_history[camera_name][track_id][1] = timestamp
            duration = self.IDs_history[camera_name][track_id][1] - self.IDs_history[camera_name][track_id][0]
            return duration
        
    def _init_tracker(self, cameras: list, tracking_args: Any = None) -> None:
        args = tracking_args if tracking_args else BYTETrackerArgs
        for cam_name in cameras:
            self.tracker[cam_name] = BYTETracker(args)
            
    def _init_image_processor(self, size: Tuple[int, int] = None, sharped: bool = False, exposure: float = None, reduction_factor: float = None) -> None:
        self.size = size
        self.sharped = sharped
        self.exposure = exposure
        self.reduction_factor = reduction_factor

    def _define_model_parameter(self, model_path: str, model_conf: float, iou_thres: float, classes: Dict[int, str], cuda: bool = True) -> None:
        self.model_path = model_path
        self.conf = model_conf
        self.iou_thres = iou_thres
        self.classes = classes
        self.cuda = cuda

    def _init_model(self, SAHI: bool = False) -> None:
        self.device = torch.device('cuda' if self.cuda and torch.cuda.is_available() else 'cpu')
        if SAHI:
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="yolov5",
                model_path=self.model_path,
                confidence_threshold=self.conf,
                device='cuda:0' if self.cuda and torch.cuda.is_available() else 'cpu',
                image_size=self.size,
                category_mapping={"0": 'Smoke', "1": 'Fire'},
            )
        else:
            # Initialize model
            self.model = torch.hub.load('yolov5', 'custom', path=self.model_path, source='local', force_reload=True, verbose=False)
            self.model.to(self.device)  # Ensure the model is on the GPU
            self.model.share_memory()
            self.model.conf = self.conf
            self.model.iou = self.iou_thres
            self.model.classes = list(self.classes.keys())
    
    def _annotate_frame(self, frame: np.ndarray, bbox: List[int], color: Tuple[int, int, int], 
                        text: str, font_size: float, box_thickness: int, track_id: Union[int, None] = None) -> np.ndarray:
        return self.plotting(frame, bbox, color, text, font_size, box_thickness, track_id)
    
    def tracking(self, 
                 stop_event: multiprocessing.Event, # type: ignore
                 save_video: bool = False,
                 show_fps: bool = False
                 ) -> None:
        # Initialize model
        self._init_model(SAHI=False)
        logger.info(f"pedestrian_tracking-PID:{os.getpid()} | Tracking-Queue-ID: {id(self.input_queue)} | Tracket-Queue-ID: {id(self.output_queue)}")
        
        frame_count = 0
        start_time = time.time()
        # fps = 30
        # time_interval = 1 / fps
        # last_written_time = 0.0
        
        # if save_video:cle
        #     cap_out = self.record_video(fps=fps, resolution=self.size, info=f"tracking-{os.getpid()}")
        
        while not stop_event.is_set():
            try:
                loop_start_time = time.time()
                online_targets = []
                durations = []

                try:
                    camera_name, frame = self.input_queue.get(block=False)
                except Exception:
                    time.sleep(0.1) # logger.error(f"[Tracking] - PID:{os.getpid()} - Got Nothing from the queue")
                    continue
                
                # Pre-process the frame
                frame_rgb = frame.copy()
                frame_rgb = self.image_processing(frame_rgb)

                # Detection
                results = self.model(frame_rgb).xyxy[0].cpu()
                
                # Tracking
                try:
                    tracker = self.tracker[camera_name]
                    if online_targets := tracker.update(
                        output_results=results, img_info=frame_rgb.shape, img_size=frame_rgb.shape
                    ):
                        for target in online_targets:
                            track_id = int(target.track_id)
                            x1, y1, x2, y2 = target.tlbr
                            bbox = [int(x1), int(y1), int(x2), int(y2)]
                            
                            # Filter out bounding boxes outside the region of interest
                            cy = (int(y1) + int(y2)) / 2
                            # cv2.line(frame_rgb, (0, self.roi_cameras[camera_name]), (frame_rgb.shape[1], self.roi_cameras[camera_name]), (0, 255, 0), 2)
                            if cy < self.roi_cameras[camera_name]:
                                continue
                                
                            # Use camera_name in _tracking_id
                            if duration := self._tracking_id(camera_name, track_id, timestamp=datetime.now()):
                                # Update IDs_duration for the specific camera
                                self.IDs_duration[camera_name][track_id] = duration
                                
                                durations.append(duration)
                                
                                frame_rgb = self._annotate_frame(
                                    frame=frame_rgb, 
                                    bbox=bbox, 
                                    color=(0, 255, 0),
                                    text=str(duration).split('.')[0], 
                                    font_size=0.68, box_thickness=1, track_id=track_id
                                )
                     
                    if online_targets and durations:       
                        if not self.output_queue.full():
                            # Pass IDs_duration specific to the camera
                            self.output_queue.put(
                                (
                                    camera_name, {"frame": frame_rgb, "id_durations": self.IDs_duration[camera_name]}
                                )
                            )
                except IndexError as exc:
                    logger.error(f"Error updating tracker: {exc}")
                    # online_targets = []
                
                # Update frame count and log FPS
                if show_fps:
                    current_time = time.time()
                    frame_count += 1
                    
                    if current_time - start_time >= 1:
                        logger.info(f"Tracking: Process-FPS: {frame_count}")
                        # with self.save_image_to_local(frame=frame_rgb, file_name=f"tracked-{camera_name}-{os.getpid()}") as img_name:
                        #     print(f"Tracking: {camera_name} | Process-FPS: {frame_count}")
                        frame_count = 0
                        start_time = current_time
                
                # if save_video and (current_time - last_written_time >= time_interval): 
                #     # Convert RGB to BGR for OpenCV
                #     cap_out.write(frame_rgb[:,:,::-1])
                #     last_written_time = current_time
                
                # Ensure memory is freed
                del camera_name
                del frame
                del frame_rgb
                
                processing_time = time.time() - loop_start_time
                sleep_time = max(0.001, 1/TRACKING_FPS - processing_time)
                time.sleep(sleep_time)
                
            except Exception as exe:
                logger.error(f"Unexpected error: {exe}", exc_info=True)
        
        # if save_video:
        #     cap_out.release()
                             
    def detection(self, 
                  stop_event: multiprocessing.Event, # type: ignore
                  save_video:bool=False,
                  show_fps:bool=False) -> None: 
        
        # initial model
        self._init_model(SAHI=False)

        logger.info(f"fire_detection-PID:{os.getpid()} | Detection-Queue-ID:{id(self.input_queue)} | Detected-Queue-ID:{id(self.output_queue)}")
        
        frame_count = 0
        start_time = time.time()
        # save_video_fps = 25
        # time_interval = 1 / save_video_fps
        # last_written_time = 0.0
        
        # if save_video:
        #     cap_out = self.record_video(fps=save_video_fps, resolution=self.size, info=f"tracking-{os.getpid()}")
        
        while not stop_event.is_set():
            try:
                loop_start_time = time.time()
                
                try:
                    camera_name, frame = self.input_queue.get(block=False)
                except Exception:
                    time.sleep(0.1)
                    # logger.error(f"[Detection] - PID:{os.getpid()} - Got Nothing from the queue")
                    continue
                # Pre-process the frame
                # frame_rgb = frame.copy()
                frame_rgb = self.image_processing(frame)

                # Detection
                try:
                    try:
                        results = self.model(frame_rgb).xyxy[0].cpu()
                    except Exception as e:
                        logger.error(f"Error during detection: {e}", exc_info=True)
                        
                    for x1, y1, x2, y2, conf, cls_id in results:
                        if self.classes[int(cls_id)] == 'Fire':
                            # Filter out the bounding boxes that out of region of interest
                            cy = (int(y1) + int(y2)) / 2
                            # cv2.line(frame_rgb, (0, self.roi_cameras[camera_name]), (frame_rgb.shape[1], self.roi_cameras[camera_name]), (0, 255, 0), 2)
                            if cy < self.roi_cameras[camera_name]:
                                continue
        
                            frame_rgb = self._annotate_frame(
                                frame=frame_rgb, 
                                bbox=[int(x1), int(y1), int(x2), int(y2)],
                                color=self.colors[int(cls_id)],
                                text=f"{self.classes[int(cls_id)]} {conf:.2f}",
                                font_size=0.68,
                                box_thickness=1,
                                track_id=None
                            )

                    # Filter predictions for the specific class (e.g., class ID 1)
                    fire_detected = [box for box in results if self.classes[int(box[-1])] == 'Fire']
                    
                    if not self.output_queue.full():
                        self.output_queue.put(
                            (
                                camera_name, {"frame": frame_rgb, "results": fire_detected}
                            )
                        )
                        
                except Exception as e:
                    logger.error(f"Error during detection: {e}", exc_info=True)
                    # stop_event.set()
                
                if show_fps:
                    current_time = time.time()
                    frame_count += 1
                    
                    if current_time - start_time >= 1:
                        logger.info(f"Detection: Process-FPS: {frame_count}")
                        # with self.save_image_to_local(frame=frame_rgb, file_name=f"detected-{camera_name}-{os.getpid()}") as img_name:
                        #     # Do something with the img_name
                        #     print(f"Image saved as {img_name}")
                        frame_count = 0
                        start_time = current_time
                
                # if save_video and (current_time - last_written_time >= time_interval): 
                #     # Convert RGB to BGR for OpenCV
                #     cap_out.write(frame_rgb[:,:,::-1])
                #     last_written_time = current_time
                processing_time = time.time() - loop_start_time
                sleep_time = max(0.01, 1/TRACKING_FPS - processing_time)
                time.sleep(sleep_time)
                
                # Ensure memory is freed
                del camera_name
                del frame
                del frame_rgb
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
        
        # if save_video:
        #     cap_out.release()
            
    def start_tracking(self, 
                       task_name: str="tracking-inference", 
                       save_video: bool=False,
                       show_fps: bool=False):
        stop_event = multiprocessing.Event()
        self.stop_events[task_name] = stop_event
        p = multiprocessing.Process(target=self.tracking, args=(stop_event, save_video, show_fps))
        self.processes[task_name] = p
        p.start()
    
    def start_detection(self, 
                       task_name: str = "detection-inference", 
                       save_video: bool = False,
                       show_fps: bool=False):
        stop_event = multiprocessing.Event()
        self.stop_events[task_name] = stop_event
        p = multiprocessing.Process(target=self.detection, args=(stop_event, save_video, show_fps))
        self.processes[task_name] = p
        p.start()
           
    def stop(self):
        for task_name, stop_event in self.stop_events.items():
            stop_event.set()
            p = self.processes[task_name]
            if p.is_alive():
                p.terminate()
                p.join()