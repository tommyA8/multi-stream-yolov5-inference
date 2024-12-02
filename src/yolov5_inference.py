import torch

class VehicleDetector:
    def __init__(self, model_path):
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)

    def detect(self, frame):
        results = self.model(frame)
        return results.xyxy[0].cpu().numpy()[:, :4].astype(int)