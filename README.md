# multi-stream-yolov5-inference

change BYTETrack module paths

```bash
sed -i 's/from yolox.tracker import matching/from tracking.yolox.tracker import matching/' tracking/yolox/tracker/byte_tracker.py
sed -i 's/from yolox.tracker import kalman_filter/from tracking.yolox.tracker import kalman_filter/' tracking/yolox/tracker/matching.py
```

Allow adding custom models to the 'models' folder and export them to any format, you can clarify the instruction and example as follows:
```bash
python3 yolov5/export.py --weights models/best.pt --include torchscript --device 0
```

# Installation

```bash
sudo docker compose up --build -d
```

Or

```bash
sudo docker build -t <image-name>
sudo docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix <image-name>
```
