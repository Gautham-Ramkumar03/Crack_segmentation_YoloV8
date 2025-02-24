import cv2
import yaml
import numpy as np
import logging
import torch
import time  # Add time module import
import argparse
from pathlib import Path
from utils.preprocessing import ImagePreprocessor
from utils.model import CrackDetector
from utils.metrics import MetricsTracker

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('crack_detection.log'),
            logging.StreamHandler()
        ]
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Crack Detection and Segmentation')
    parser.add_argument('--source', type=str, default='0',
                       help='source (0 for webcam, or video file path)')
    parser.add_argument('--model-size', type=str, default='medium',
                       choices=['nano', 'small', 'medium', 'large', 'xlarge'],
                       help='YOLOv8 model size')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                       help='confidence threshold')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='device to use (cuda/cpu)')
    return parser.parse_args()

def visualize_results(original, preprocessed, results, config, metrics):
    viz_config = config['visualization']
    
    # Get original dimensions
    h, w = original.shape[:2]
    
    # Resize preprocessed image to match original dimensions
    preprocessed = cv2.resize(preprocessed, (w, h))  # Fixed: correct order (width, height)
    
    # Create side-by-side display
    display = np.zeros((h, w * 2, 3), dtype=np.uint8)
    
    # Copy images to display
    try:
        display[:, :w] = original
        display[:, w:] = preprocessed
        
        # Plot detection results
        if results.masks is not None:
            for mask in results.masks.data:
                mask = mask.cpu().numpy() * 255
                mask = cv2.resize(mask, (w, h))  # Fixed: correct order (width, height)
                mask = mask.astype(np.uint8)
                display[:, w:][mask > 127] = [0, 0, 255]
        
        # Add metrics overlay
        metrics_display = metrics.get_metrics_display()
        y_offset = 30
        for text in metrics_display.values():
            cv2.putText(
                display, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            y_offset += 30
    except Exception as e:
        logging.error(f"Visualization error: {str(e)} - Shapes: original {original.shape}, preprocessed {preprocessed.shape}")
        raise
    
    return display

def main():
    setup_logging()
    args = parse_args()
    
    metrics = MetricsTracker()
    
    try:
        # Load and update config with command line arguments
        config = load_config()
        config['inference'].update({
            'model_size': args.model_size,
            'conf_threshold': args.conf_thres,
            'device': args.device
        })
        
        preprocessor = ImagePreprocessor(config)
        detector = CrackDetector(config)
        
        # Set CUDA device properties
        if torch.cuda.is_available() and args.device == 'cuda':
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
        
        # Handle source input
        source = int(args.source) if args.source.isdigit() else args.source
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {args.source}")
        
        # Get video properties for potential saving
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process based on image size
            if frame.shape[0] * frame.shape[1] > 640 * 640:
                results = detector.process_large_image(frame)
            else:
                preprocessed = preprocessor.preprocess(frame)
                results = detector.predict(preprocessed)
            
            # Update metrics
            process_time = time.time() - start_time
            metrics.update_fps(process_time)
            metrics.log_detection(results)
            
            # Visualize with metrics
            display = visualize_results(
                frame, preprocessed, results, config, metrics
            )
            
            cv2.imshow(config['visualization']['window_name'], display)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        raise
    finally:
        cv2.destroyAllWindows()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
