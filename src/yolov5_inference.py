import time
import threading
import logging
from collections import defaultdict
from multiprocessing import Process, Queue
import torch
import cv2
import yaml
import numpy as np

from tracking.yolox.tracker.byte_tracker import BYTETracker
from src.parked_detector import ParkedDetector

# Set up logging configuration
def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    return logger

logger = setup_logging()  # Default to INFO level

class BYTETrackerArgs:
    track_thresh = 0.5  # High_threshold
    track_buffer = 40  # Number of frame lost tracklets are kept
    match_thresh = 0.8  # Matching threshold for first stage linear assignment
    aspect_ratio_thresh = 10.0  # Minimum bounding box aspect ratio
    min_box_area = 1.0  # Minimum bounding box area
    mot20 = False  # If used, bounding boxes are not clipped.
                    
class VehicleDetector:
    def __init__(self, rtsp_urls, model_path, device, yaml_path, queue_size=100, conf=0.25, iou_thres=0.45) -> None:
        self.rtsp_urls = rtsp_urls
        self.queues = [Queue(maxsize=queue_size) for _ in rtsp_urls]
        self.class_names = self.load_class_names(yaml_path)
        self.class_colors = self.generate_class_colors(len(self.class_names))
        self.device = device
        self.conf = conf
        self.iou_thres = iou_thres
        self.model_path = model_path
        self.trackers = [BYTETracker(BYTETrackerArgs) for _ in rtsp_urls]
        self.positions = {}
        self.parked_detectors = None
        self.stop_events = [threading.Event() for _ in self.rtsp_urls]
        
        logger.info("VehicleDetector initialized.")

    def init_parked_detector(self, dist_sencitivity=5, time_limit_sec=0.25):
        self.parked_detectors = [ParkedDetector(dist_sencitivity, time_limit_sec) for _ in self.rtsp_urls]
        logger.info("ParkedDetector initialized.")
        
    def load_class_names(self, yaml_path):
        logger.debug(f"Loading class names from {yaml_path}.")
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)
        return data.get('names', {})

    def generate_class_colors(self, num_classes):
        np.random.seed(42)
        colors = np.random.randint(0, 256, size=(num_classes, 3))
        logger.debug(f"Generated {num_classes} class colors.")
        return colors

    def stream_worker(self, cam_id, rtsp_url, queue):
        logger.info(f"Starting stream worker for {rtsp_url}.")
        cap = cv2.VideoCapture(rtsp_url)
        
        while cap.isOpened():
            if self.stop_events[cam_id].is_set():
                logger.info(f"Stopping stream worker for {rtsp_url}.")
                break
            
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame from {rtsp_url}.")
                break
            
            if not queue.full():
                queue.put(frame)
                
        cap.release()
        logger.info(f"Stream worker for {rtsp_url} finished.")

    def visualize(self, frame, results, window_name, show, tracking=False):
        if tracking:
            for target in results:
                track_id = int(target.track_id)
                x1, y1, x2, y2 = map(int, target.tlbr)
                color = (0, 255, 0)

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if track_id not in self.positions:
                    self.positions[track_id] = []
                    
                if len(self.positions[track_id]) > 10:
                    self.positions[track_id].pop(0)

                for i in range(1, len(self.positions[track_id])):
                    cv2.line(frame, self.positions[track_id][i-1], self.positions[track_id][i], color, 2)

                cv2.circle(frame, (cx, cy), 3, color, -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            for x1, y1, x2, y2, conf, cls_id in results:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_name = self.class_names.get(int(cls_id), "Unknown")
                color = tuple(self.class_colors[int(cls_id)])
                color = tuple(map(int, color))

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                frame = cv2.circle(frame, (cx, cy), 3, color, -1)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                frame = cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if show:
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

    def define_classes(self, classes):
        for cls in classes:            
            if cls not in self.class_names.values():
                raise ValueError(f"Class '{cls}' not found in the class names.")
        self.class_names = {k: v for k, v in self.class_names.items() if self.class_names[k] in classes}
 
    def __init_model(self):
        if "." in self.model_path or "/" in self.model_path:
            model = torch.hub.load('yolov5', 'custom', path=self.model_path, source='local', force_reload=True, verbose=False)
        else:
            model = torch.hub.load('ultralytics/yolov5', self.model_path, force_reload=True, verbose=False)

        model.to(self.device)
        model.conf = self.conf
        model.iou = self.iou_thres
        model.classes = [k for k, _ in self.class_names.items()]
        return model
    
    def process_worker(self, cam_id, queue, window_name, show, tracking):
        if show:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # initialize model
        model = self.__init_model()
        
        logger.info(f"Starting process worker for camera {cam_id}.")
        while not self.stop_events[cam_id].is_set():
            try:
                frame = queue.get(timeout=5)
            except Exception:
                if queue.empty():
                    logger.warning(f"Queue for camera {cam_id} is empty.")
                    self.stop(cam_id, window_name)
                    
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (640, 640))
            with torch.no_grad():
                results = model(frame).xyxy[0].cpu().clone()

            logger.debug(f"Detection results for camera {cam_id}: {results}")

            if tracking:
                results = self.trackers[cam_id].update(output_results=results, img_info=frame.shape, img_size=frame.shape)
                if self.parked_detectors is not None:
                    self.parked_detectors[cam_id].detect(results, frame, cam_id)
                
            self.visualize(frame, results=results, window_name=window_name, show=show, tracking=tracking)

        logger.info(f"Process worker for camera {cam_id} finished.")

    def run(self, window_name="Detection", show=False, tracking=False, debug=False):
        # Adjust logging level based on the `debug` flag
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        self.worker_processes = defaultdict(list)
        self.stream_processes = defaultdict(list)
        
        for cam_id, rtsp_url in enumerate(self.rtsp_urls):
            logger.info(f"Starting process for camera {cam_id} with RTSP URL: {rtsp_url}")
            stream_proc = Process(target=self.stream_worker, args=(cam_id, rtsp_url, self.queues[cam_id]))
            process_proc = Process(target=self.process_worker, args=(cam_id, self.queues[cam_id], f"{cam_id}-{window_name}", show, tracking))
            
            stream_proc.start()
            process_proc.start()
            
            self.stream_processes[cam_id].append(stream_proc)
            self.worker_processes[cam_id].append(process_proc)


    def stop(self, cam_id, window_name):
        logger.info("Stopping the system...")
        self.stop_events[cam_id].set()
        
        self.stream_processes[cam_id][0].terminate()
        self.stream_processes[cam_id][0].join()
        self.worker_processes[cam_id][0].terminate()
        self.worker_processes[cam_id][0].join()

        cv2.destroyWindow(window_name)
        logger.info("System stopped successfully.")

if __name__ == "__main__":
    rtsp_urls = ["data/video/Relaxing_highway_traffic.mp4", "data/video/buffalo5.mp4"]  # Example RTSP streams or local video files
    detector = VehicleDetector(rtsp_urls, model_path="yolov5s", yaml_path="yolov5/data/coco.yaml", device="cpu")
    detector.run(show=True, tracking=True, debug=True)  # Set debug=True to show detailed logs
