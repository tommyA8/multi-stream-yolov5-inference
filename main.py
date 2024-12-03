from multiprocessing import Process, Queue
import torch
import cv2
import time

class VehicleDetector:
    def __init__(self, rtsp_urls, model_path, conf=0.25, iou_thres=0.45) -> None:
        self.rtsp_urls = rtsp_urls
        self.queues = [Queue(maxsize=10) for _ in rtsp_urls]
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf = conf
        self.iou_thres = iou_thres
        
        self.model_path = model_path
        self.model = torch.hub.load('yolov5/yolov5', 'custom', path=model_path, source='local', force_reload=True, verbose=False)
        self.model.to(self.device)  # Ensure the model is on the GPU
        self.model.conf = self.conf
        self.model.iou = self.iou_thres
        
    def stream_worker(self, rtsp_url, queue):
        cap = cv2.VideoCapture(rtsp_url)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if not queue.full():
                queue.put(frame)
                
        cap.release()
        
    @staticmethod
    def visualize(frame, results, window_name, show):
        for x1, y1, x2, y2, conf, cls_id in results:
            # Filter out the bounding boxes that are out of the region of interest
            cy = (int(y1) + int(y2)) // 2  # Integer division for center
            cv2.circle(frame, (x1, cy), 10, (0, 0, 255), -1)  # Red circle at vertical center
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Class-ID {cls_id} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if show:
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)  # Display for 1 ms and wait for key press
        
    def process_worker(self, queue, window_name="Detection", show=False):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        start = time.time()
        while True:
            try:
                frame = queue.get(timeout=5)  # Timeout after 5 seconds to avoid blocking forever
            except Exception:
                time.sleep(0.1)  # Wait and try again if the queue is empty
                continue
            
            frame = cv2.resize(frame, (640, 640))  # Resize frame to match model input size
            results = self.model(frame[:,:,::-1]).xyxy[0].cpu()  # Model inference
            self.visualize(frame, results, window_name, show=show)
            cv2.imwrite('data/output.png', frame[:,:,::-1])
            print(f"FPS: {1/(time.time()-start):2f}")
            start = time.time()  # Reset start time for FPS calculation

    def run(self):
        self.processes = []
        for cam_id, rtsp_url in enumerate(self.rtsp_urls):
            stream_proc = Process(target=self.stream_worker, args=(rtsp_url, self.queues[cam_id],))
            process_proc = Process(target=self.process_worker, args=(self.queues[cam_id],))
            self.processes.extend([stream_proc, process_proc])
            stream_proc.start()
            process_proc.start()
            
    def stop(self):
        try:
            while True:
                time.sleep(1)  # Keep the main thread alive until user interrupts
        except KeyboardInterrupt:
            print("Stopping processes...")
            for proc in self.processes:
                proc.terminate()  # Terminate all processes
                proc.join()  # Wait for processes to finish

if __name__ == "__main__":
    rtsp_urls = ["rtsp://admin:swd12345@192.168.22.22:554/sub"]  # RTSP stream URL
    detector = VehicleDetector(rtsp_urls, model_path="yolov5/yolov5/runs/best.pt")
    detector.run()
