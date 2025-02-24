import time
import numpy as np
import cv2

class MetricsTracker:
    def __init__(self):
        self.fps_buffer = []
        self.processing_times = []
        self.detection_counts = []
        self.confidence_scores = []
        
    def update_fps(self, frame_time):
        self.fps_buffer.append(1.0 / frame_time)
        if len(self.fps_buffer) > 30:
            self.fps_buffer.pop(0)
    
    def get_average_fps(self):
        return np.mean(self.fps_buffer) if self.fps_buffer else 0
    
    def log_detection(self, results):
        if results.boxes is not None:
            self.detection_counts.append(len(results.boxes))
            self.confidence_scores.extend(
                results.boxes.conf.cpu().numpy().tolist()
            )
    
    def get_metrics_display(self):
        avg_fps = self.get_average_fps()
        avg_detections = np.mean(self.detection_counts) if self.detection_counts else 0
        avg_confidence = np.mean(self.confidence_scores) if self.confidence_scores else 0
        
        return {
            'fps': f'FPS: {avg_fps:.1f}',
            'detections': f'Avg Detections: {avg_detections:.1f}',
            'confidence': f'Avg Confidence: {avg_confidence:.2f}'
        }
