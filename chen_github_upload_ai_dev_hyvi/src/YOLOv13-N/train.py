import sys                 
import os                                                                                                     
                                                                                                                                                                                                                                                                                                                                                         
sys.path.append(os.path.abspath('../../third_party_libraries'))                                     
                                                                                                                                                                  
from ultralytics import YOLO                                                                                              
import yaml                                                        
                                                                                                                                                                                                                                                                                  
# Load model configuration (YOLOv13-nano)                                                                                                       
model = YOLO('yolov13n.yaml')                                                                                          
                                                                                                                                     
# read the configuration parameters from cofig.yml file               
with open('config.yml', 'r') as file:                                                                                                                                                                           
    config = yaml.safe_load(file)                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
# Train the model                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
results = model.train(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
  data="coco.yaml",    # Dataset config file                                             
  epochs=config["train"]["epochs"],     # Number of training epochs  
  batch=config["train"]["batch"],    # Batch size per GPU
  imgsz=config["train"]["imgsz"],   # Image size (height and width) for training 
  scale=config["train"]["scale"],    # Image scale factor for multi-scale training
  mosaic=config["train"]["mosaic"],   # Probability of using mosaic augmentation
  mixup=config["train"]["mixup"],    # Probability of using mixup augmentation
  copy_paste=config["train"]["copy_paste"],    # Probability of using copy-paste augmentation 
  device=config["train"]["device"],        # Train on GPU device 0           
  project=config["train"]["project"],        # save model to desired path
  exist_ok=config["train"]["exist_ok"]       # overwrite if folder exists
)                                                                                                                                                                
                                                                                                                                     
                                                                 