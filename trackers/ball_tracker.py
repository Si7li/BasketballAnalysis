from ultralytics import YOLO
import sys
sys.path.append("../")
from utils import read_stub,save_stub

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frame):
        batch_size = 20
        detections = []
        for i in range(0, len(frame), batch_size):
            batch = frame[i:i + batch_size]
            results = self.model.predict(source=batch, conf=0.5)
            detections.extend(results)
        return detections
    
    def get_object_tracks(self, frames,read_from_stub=False ,stub_path=None):
        
        tracks = read_stub(read_from_stub, stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks
