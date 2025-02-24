import torch
import logging
from ultralytics import YOLO
import torch.nn.functional as F

MODEL_PATHS = {
    'nano': 'Models/YOLOv8-crack-seg/yolov8n/weights/best.pt',
    'small': 'Models/YOLOv8-crack-seg/yolov8s/weights/best.pt',
    'medium': 'Models/YOLOv8-crack-seg/yolov8m/weights/best.pt',
    'large': 'Models/YOLOv8-crack-seg/yolov8l/weights/best.pt',
    'xlarge': 'Models/YOLOv8-crack-seg/yolov8x/weights/best.pt'
}

class CrackDetector:
    def __init__(self, config):
        self.config = config['inference']
        self.setup_device()
        self.load_model()
        self.warmup()
        self.setup_cuda_optimizations()
        
    def setup_device(self):
        if torch.cuda.is_available() and self.config['device'] == 'cuda':
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            logging.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            logging.warning("CUDA not available, using CPU")
    
    def setup_cuda_optimizations(self):
        if self.device.type == 'cuda':
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Set optimal CUDNN algorithms
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            # Set memory allocation to be as efficient as possible
            torch.cuda.set_per_process_memory_fraction(
                self.config.get('cuda_memory_fraction', 0.8)
            )
    
    def load_model(self):
        try:
            self.model = YOLO(MODEL_PATHS[self.config['model_size']])
            self.model.to(self.device)
            if self.device.type == 'cuda':
                self.model.fuse()  # Fuse layers for optimal CUDA performance
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise
    
    def warmup(self):
        """Warm up GPU with dummy inference."""
        if self.device.type == 'cuda':
            dummy_input = torch.zeros(1, 3, 640, 640).to(self.device)
            for _ in range(self.config['warmup_iterations']):
                with torch.no_grad():
                    self.model(dummy_input)
            torch.cuda.synchronize()
    
    def process_large_image(self, image):
        """Process large images by splitting into tiles."""
        h, w = image.shape[:2]
        tile_size = 640
        
        if h <= tile_size and w <= tile_size:
            return self.predict(image)
        
        # Calculate tiles
        tiles = []
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                tile = image[y:min(y + tile_size, h), 
                           x:min(x + tile_size, w)]
                tiles.append(tile)
        
        # Process tiles in parallel
        batch = torch.stack([self.preprocess_tile(t) for t in tiles])
        with torch.amp.autocast('cuda'):  # Updated autocast syntax
            results = self.model(batch, batch_size=len(tiles))
        
        return self.merge_results(results, h, w, tile_size)
    
    @staticmethod
    def preprocess_tile(tile):
        """Preprocess single tile for batched inference."""
        tile = F.interpolate(
            torch.from_numpy(tile).unsqueeze(0).permute(0, 3, 1, 2),
            size=(640, 640),
            mode='bilinear',
            align_corners=False
        )
        return tile
    
    def predict(self, image):
        try:
            with torch.no_grad(), torch.amp.autocast('cuda'):  # Updated autocast syntax
                # Dynamic batch size handling
                if self.config['dynamic_batch_size']:
                    batch_size = min(
                        self.config['max_batch_size'],
                        torch.cuda.get_device_properties(0).total_memory // (1024**3)
                    )
                else:
                    batch_size = 1
                
                results = self.model(
                    image,
                    conf=self.config['conf_threshold'],
                    iou=self.config['iou_threshold'],
                    batch=batch_size
                )
            return results[0]
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise
