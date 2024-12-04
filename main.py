from src.yolov5_inference import VehicleDetector
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Detection with YOLOv5 and ByteTracker")
    parser.add_argument('--debug', action='store_true', help="Enable debug level logging")
    parser.add_argument('--show', action='store_true', help="Show the detection results in a window")
    parser.add_argument('--tracking', action='store_true', help="Enable object tracking")
    parser.add_argument('--model', type=str, default="yolov5s", help="YOLOv5 model size (e.g., yolov5s, yolov5m, yolov5l, yolov5x)")
    parser.add_argument('--conf', type=float, default=0.25, help="Object confidence threshold")
    parser.add_argument('--iou', type=float, default=0.45, help="IOU threshold for NMS")
    parser.add_argument('--qsize', type=int, default=128, help="Queue size for each camera")
    # parser.add_argument('--rtsp_urls', nargs='+', required=True, help="List of RTSP URLs or video files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    rtsp_urls = ["data/video/Khao-Chi-On_CCTV34R.mp4"]  # Example RTSP streams or local video files

    detector = VehicleDetector(
        rtsp_urls,
        model_path=args.model,
        conf=args.conf,
        iou_thres=args.iou,
        queue_size=args.qsize,
        yaml_path="yolov5/data/coco.yaml",
        device="cuda"
    )

    detector.run(show=args.show, tracking=args.tracking, debug=args.debug)  # Set debug=True to show detailed logs
    # example: python3 main.py --show --tracking --model yolov5m --conf 0.001

    # สำหรับดูรอจอดนิ่งๆ ให้จับเวลารถที่ไม่มี tail
    # if len(tail) <= 5:
    #    print("No tail")
    #    จับเวลา
    #    ถ้าเวลาที่จับได้มากกว่า 5 นาที
    #    ให้ส่งข้อมูลไปยัง API