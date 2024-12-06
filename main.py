from src.yolov5_inference import VehicleDetector
import dotenv
import os
import warnings
import logging

# Set up logging configuration
def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    return logger

logger = setup_logging()  # Default to INFO level 

warnings.simplefilter("ignore", category=FutureWarning)
dotenv.load_dotenv()

MODEL = str(os.getenv("MODEL", "yolov5m")) 
CONF = float(os.getenv("CONF", 0.5))
IOU = float(os.getenv("IOU", 0.45))
QSIZE = int(os.getenv("QSIZE", 500))
DEVICE = str(os.getenv("DEVICE", "cpu"))
SHOW = bool(int(os.getenv("SHOW", 1)))
TRACKING = bool(int(os.getenv("TRACKING", 0)))
DEBUG = bool(int(os.getenv("DEBUG", 0)))
RTSP_URLS = os.getenv("RTSP_URLS", "data/video/Khao-Chi-On_CCTV34R.mp4").split(",")

detector = VehicleDetector(
    rtsp_urls=RTSP_URLS,
    model_path=MODEL,
    conf=CONF,
    iou_thres=IOU,
    queue_size=QSIZE,
    yaml_path="data/coco.yaml",
    device=DEVICE
)

if __name__ == "__main__":
    detector.run(show=SHOW, tracking=TRACKING, debug=DEBUG)  # Set debug=True to show detailed logs
    
    
    
    # สำหรับดูรอจอดนิ่งๆ ให้จับเวลารถที่ไม่มี tail
    # if len(tail) <= 5:
    #    print("No tail")
    #    จับเวลา
    #    ถ้าเวลาที่จับได้มากกว่า 5 นาที
    #    ให้ส่งข้อมูลไปยัง API