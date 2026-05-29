import os                                                                      
import sys                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                         
sys.path.append(os.path.abspath('../../third_party_libraries'))                                                                        
                                                                                                                                                                                             
from ultralytics import YOLO                                                                             
                                                                                                                                                                                                                              
base_path = os.path.dirname(__file__)                                                                                             
model_weight = os.path.join(base_path, 'models', 'yolov13n.pt')                                       

model = YOLO(model_weight)                                                                                  
                                                                                                                                
results = model("/home/gpnpu/Desktop/AI_development/frame_2_0139.png")                                                                  
#results[0].show()         
results[0].save("/home/gpnpu/Desktop/AI_development/YOLO_E_Development/yoloe_ultralytics/yolov13N_frame_2_0139.png")                                                               


                                                                                                                                                         
                                                                              
                                                                                                        
