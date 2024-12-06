# doh-highway-monitoring

```bash
sudo docker compose up --build -d
```

Or

```bash
sudo docker build -t <image-name>
sudo docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix <image-name>
```

# Convert model format

```bash
docker exec -it <image-name> bash -c "python3 yolov5/export.py --weights yolov5m.pt --include torchscript --device 0 && python3 main.py"
```