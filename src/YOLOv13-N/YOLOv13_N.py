import cv2
import time                                                             
import os                                                                      
import sys                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                         
sys.path.append(os.path.abspath('../../third_party_libraries'))  
                                                                                                                                                                                             
from ultralytics import YOLO    
                           
import logging
from ultralytics.utils import LOGGER                                                                                      
LOGGER.setLevel(logging.WARNING)     
                                                                              
                                                                        
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout                                      
        self._original_stderr = sys.stderr                                                                                                 
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()                                              
        sys.stderr.close()                                       
        sys.stdout = self._original_stdout                                                        
        sys.stderr = self._original_stderr                                                                                      
                                                                                                                                                                                                                                                  
base_path = os.path.dirname(__file__)                                                                                                                   
model_weight = os.path.join(base_path, 'models', 'yolov13n.pt')                                                                        
model_config = os.path.join(base_path, '../../third_party_libraries/ultralytics/cfg/models/v13', 'yolov13n.yaml')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
model = YOLO(model_config)                                                                                                                                                                     
model.load(model_weight)       

model_features = []                                                             
                                                                                                                           
# model name
model_features.append("yolov13n")                                 

parameters_GFLOPs = model.info()                                                             
# parameters                                                                              
model_features.append(f"{(parameters_GFLOPs[1] / 1e6):.1f}")  
# GFLOPs                                           
model_features.append(f"{parameters_GFLOPs[3]: .1f}")        
                                                                                                                                         
cap = cv2.VideoCapture("../../demos/YOLOv13-N/source/traffic.mp4")
                                                                                   
with SuppressOutput():                                                                                                                                                                                                                                              
    time_list = []                                                         
    n = 0
    while True: 
        ret, frame = cap.read()   
        
        if not ret:
            break 
        
        s_time = time.time()                                                                                    
        model.predict(frame, imgsz=640, device='cpu')    
                
        e_time = time.time()                                                                    
        latency = (e_time - s_time)                                                          
                                                      
        if n == 0:                                                                            
            pass                  
        else:                                                                                                                                      
            time_list.append(latency)         
                                                                                                                                                  
        n += 1                                                              
                                                      
    average_latency = (sum(time_list) / len(time_list)) * 1000 
                                                                              
    # average_latency                                                                                         
    model_features.append(f"{(average_latency):.2f}")                                                                   
                                                                                                                                     
    metrics = model.val(data='coco.yaml',  verbose=False)                                                       
    mAP50_95 = metrics.results_dict["metrics/mAP50-95(B)"]                                  
    # mAP50_95                                                                                          
    model_features.append(f"{(mAP50_95 * 100):.1f}")                                                
                                                                                                                                                                                                                                            
features = ' '.join(model_features)                                                                                                                                   
                                                                                                                             
print("features:", features)                                                                                                                
                                                                                                                    
                                                                                                                    
                                                                                                                                                           
                                                                                                                                                              