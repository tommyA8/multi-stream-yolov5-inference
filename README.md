# multi-stream-yolov5-inference

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
