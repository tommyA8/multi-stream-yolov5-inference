from collections import defaultdict
import cv2
import math
import time
import logging

# Set up logging configuration
def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    return logger

logger = setup_logging()  # Default to INFO level

class ParkedDetector:
    def __init__(self, dist_sencitivity=5, time_limit_sec=10):
        self.positions = {}
        self.track_ids_time = defaultdict(list)
        self.time_limit = time_limit_sec * 60
        self.dist_sencitivity = dist_sencitivity
    
    def __position_process(self, track_id, cx, cy):
        if track_id not in self.positions:
            self.positions[track_id] = []
                
        # Avoid duplicates
        prev_cx, prev_cy = self.positions[track_id][-1] if self.positions[track_id] else (cx, cy)
        distance = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
        
        if (cx, cy) not in self.positions[track_id] and distance == 0:
            self.positions[track_id].append((cx, cy))
            
        if distance > self.dist_sencitivity:
            self.positions[track_id].append((cx, cy))
            
    def __parked_vehicle(self, track_id):
        if len(self.positions[track_id]) > 3:
            return False
        
        if track_id not in self.track_ids_time:
            self.track_ids_time[track_id] = {'start_time': time.time(), 'end_time': None}
        
        self.track_ids_time[track_id]['end_time'] = time.time()
        
        # check if the vehicle is parked for more than {time_limit} minutes
        if self.track_ids_time[track_id]['end_time'] - self.track_ids_time[track_id]['start_time'] > self.time_limit:
            return True
        else:
            return False
    
    def detect(self, results, frame, cam_id):
        for target in results:
            track_id = int(target.track_id)
            x1, y1, x2, y2 = map(int, target.tlbr)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            self.__position_process(track_id, cx, cy)
            
            parked = self.__parked_vehicle(track_id)
            
            if parked:
                logger.info(f"Track ID: {track_id} is parked for more than {self.time_limit} minutes.")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Track ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.putText(frame, f"Parked for more than {self.time_limit} minutes", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.imwrite(f"assets/parked_vehicle_pic/cam_{cam_id}_parked_vehicle_{track_id}.jpg", frame)
            
            # Remove the first element if the list is greater than 10
            if len(self.positions[track_id]) > 10:
                    self.positions[track_id].pop(0)