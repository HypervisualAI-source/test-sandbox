import numpy as np                                                                            
import cv2     
import os                                                                                          
import sys                                                                                            
import torch                                                                                                
import glob                                                                                                                                                                    
import time    
from vit_latent_hvai import ViT_latent_hvai                  
                                           
import warnings
warnings.filterwarnings(                  
    "ignore"                                                                                                                               
)                                                  
                                                                                                               
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)  

# the categories of CIFAR-10                       
categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# testing data                                                                                                                                                                 
frame = cv2.imread("test.png")                                                         
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                                  
frame = torch.from_numpy(np.expand_dims(np.transpose(frame.astype(np.float32), (2, 0, 1)), axis = 0)) 

# testing data with FP32
frame = frame / 255.0     
# testing data with BF16
frame_bf16 = frame.to(torch.bfloat16)   

device = 'cpu'
                                                                          
# loading the pre-trained model                                                                             
model = torch.load("HVAI_MLA_ViT.pth", map_location=device)                                                                                                                                                          
vit_model = model["model"]                                                                                                                                                                                                                                                                          
vit_model.eval()                            
                                                                                                                                                                                                   
                         
## case 1: classifcation by model with FP32
# model with FP32
vit_model_fp32 = vit_model     
# verifying dtype
for name, param in vit_model_fp32.named_parameters():
    if name == "transformer.norm.weight":
        print("ViT with FP32:")
        print("transformer.norm.weight[360]:", param[360], param[360].dtype)
        break

with torch.no_grad():       
    print("input:", frame.device, frame.dtype)    
    output = vit_model_fp32(frame)                                                                                                  
    print("output;", output.device, output.dtype)
                  
probs = torch.nn.functional.softmax(output[0], dim=0)                                   
top_prob, top_class_idx = torch.max(probs, dim=0)  
top_prob = f"{(top_prob.item() * 100):.2f}" 
                                                                                                                                                             
# the predicated category
predicated_class = categories[top_class_idx]
print("Category: ", predicated_class)
print("Probability:", top_prob)                                                     
                                                                       
print("\n")                                                                                                                                                  
                                                                                                                                                           
## case 2: classifcation by model with BF16                
# model with BF16                                                                                                            
vit_model_bf16 = vit_model.to(torch.bfloat16)         
# verifying dtype
for name, param in vit_model_bf16.named_parameters():
    if name == "transformer.norm.weight":
        print("ViT with BF16:")
        print("transformer.norm.weight[360]:", param[360], param[360].dtype)                        
        break                            
with torch.no_grad():                                                                
    print("input:", frame_bf16.device, frame_bf16.dtype)        
    output = vit_model_bf16(frame_bf16)   
    print("output;", output.device, output.dtype)
                  
probs = torch.nn.functional.softmax(output[0], dim=0)                                   
top_prob, top_class_idx = torch.max(probs, dim=0)  
top_prob = f"{(top_prob.item() * 100):.2f}" 

# the predicated category
predicated_class = categories[top_class_idx]
print("Category: ", predicated_class)
print("Probability:", top_prob)






    
