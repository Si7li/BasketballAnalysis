import cv2
import sys
sys.path.append("../")
from utils import get_bbox_width,get_center_of_bbox

def draw_ellipse(frame, bbox, color, track_id=None):
    y2 = int(bbox[3])
    x_center,_ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    cv2.ellipse(frame, center=(x_center, y2), 
                axes=(int(width), int(0.35*width)),
                angle= 0, 
                startAngle=-45, 
                endAngle=235, 
                color=color, 
                thickness=2,
                lineType=cv2.LINE_4)
