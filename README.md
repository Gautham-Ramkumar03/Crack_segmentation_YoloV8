# Crack Segmentation using YOLOv8 🔍

A robust deep learning solution for detecting and segmenting cracks in images and video streams using YOLOv8. This project provides real-time crack detection with advanced preprocessing and visualization capabilities.

## 🌟 Features

- Real-time crack detection and segmentation
- Multiple YOLOv8 model size options
- Advanced image preprocessing pipeline
- CUDA-optimized inference
- Live performance metrics
- Support for images, videos, and webcam input

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Gautham-Ramkumar03/Crack_segmentation_YoloV8.git
cd Crack_segmentation_YoloV8
cd Models
```

### 2. Download Pre-trained Models
```bash
git clone https://huggingface.co/OpenSistemas/YOLOv8-crack-seg 
```

### 3. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate crack_detection
```

## 🔧 Configuration

The 

config.yaml

 file contains all configurable parameters:

```yaml
preprocessing:
  resize_dims: [640, 640]
  clahe:
    clip_limit: 2.0
    tile_grid_size: [8, 8]
  # ...other preprocessing parameters

inference:
  conf_threshold: 0.35
  iou_threshold: 0.45
  model_size: "medium"
  device: "cuda"
  # ...other inference parameters
```

## 🚀 Usage

### Basic Command
```bash
python main.py --source <input> --model-size <size> --device <device>
```

### Arguments
- `--source`: Input source (0 for webcam, or path to image/video)
- `--model-size`: YOLOv8 model variant [nano|small|medium|large|xlarge]
- `--conf-thres`: Confidence threshold (default: 0.25)
- `--device`: Computing device [cuda|cpu]

### Examples
```bash
# Webcam inference
python main.py --source 0 --model-size medium

# Video file inference
python main.py --source path/to/video.mp4 --model-size large

# CPU inference
python main.py --source 0 --model-size small --device cpu
```

## 📁 Project Structure

```
.
├── config.yaml           # Configuration parameters
├── main.py              # Main inference script
├── utils/
│   ├── preprocessing.py # Image preprocessing pipeline
│   ├── model.py        # YOLOv8 model wrapper
│   └── metrics.py      # Performance metrics tracking
└── Models/             # Pre-trained YOLOv8 models
```

## 🔍 Code Components

- 

main.py

: Orchestrates the detection pipeline
- 

preprocessing.py

: Implements adaptive image preprocessing
- 

model.py

: Handles model loading and inference
- 

metrics.py

: Tracks FPS and detection metrics

## 🎯 Key Features

1. **Adaptive Preprocessing**
   - CLAHE enhancement
   - Bilateral filtering
   - Intelligent illumination adaptation

2. **Optimized Inference**
   - CUDA acceleration
   - Batch processing
   - Memory-efficient large image handling

3. **Real-time Visualization**
   - Side-by-side comparison
   - Performance metrics overlay
   - Segmentation mask visualization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎓 Citation

If you use this code in your research, please cite:

```
@misc{crack_segmentation_yolov8,
  author = {Gautham Ramkumar},
  title = {Crack Segmentation using YOLOv8},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Gautham-Ramkumar03/Crack_segmentation_YoloV8}}
}
```

## 📧 Contact

For questions or feedback, please open an issue or reach out to [your-email@example.com](mailto:your-email@example.com)

Similar code found with 2 license types
