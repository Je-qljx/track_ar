# Model Files

Place YOLOv8 model files here:

- `yolov8n.pt` — YOLOv8 Nano (fastest, recommended for 60fps)
- `yolov8s.pt` — YOLOv8 Small (better accuracy)

Download:
```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

For TensorRT acceleration:
```bash
pip install tensorrt
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='engine')"
```
