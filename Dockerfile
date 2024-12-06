FROM python:3.10-slim

# Install necessary packages and clean up the apt cache
RUN apt-get update && apt-get install -y \
    build-essential cmake git vim \
    libopencv-dev gcc libpq-dev \
    libxcb1 libx11-dev libxext6 libxi6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the project code into the container
COPY . /app
WORKDIR /app

# Initialize and update submodules
RUN git submodule init && git submodule update --recursive

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir lap \
    && rm -rf /root/.cache/pip

# Create directory for custom models and copy models into the container
RUN mkdir -p /app/yolov5/custom_models
COPY ./models/ /app/yolov5/custom_models/

# Modify Python files using sed
RUN sed -i 's/from yolox.tracker import matching/from tracking.yolox.tracker import matching/' /app/tracking/yolox/tracker/byte_tracker.py && \
    sed -i 's/from yolox.tracker import kalman_filter/from tracking.yolox.tracker import kalman_filter/' /app/tracking/yolox/tracker/matching.py

    # Set the working directory for running the application
WORKDIR /app/yolov5
RUN python3 export.py --weights yolov5m.pt --include torchscript --device 0

# Set the working directory for running the application
WORKDIR /app

# Run the main.py script with the desired parameters
CMD ["python3", "main.py"]
# CMD ["python3", "main.py", "--show", "--tracking", "--model", "yolov5m.torchscript", "--conf", "0.5", "--qsize", "1000", "--device", "cuda"]

# sudo docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix doh-vehicle-tracking