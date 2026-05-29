import numpy as np                                                                            
import cv2     
import os                                                                                          
import sys                                                                                            
import torch                                                                                                
import glob                                                                                                                                                                    
import time    
from vit_latent_hvai import ViT_latent_hvai                                                                                               
                                    

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)                               
                                                                            
model = torch.load("HVAI_MLA_ViT.pth", map_location='cpu')                                                                                                                                                          
vit_model = model["model"] 
                                                      
                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                            
vit_model.eval()                                                                                                                                                                                                                                                                                                                                                                           
                                                                                                                                                                  
frame = cv2.imread("test.png")    # the image size has to be 32 x 32                                                          
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                                  
frame = torch.from_numpy(np.expand_dims(np.transpose(frame.astype(np.float32), (2, 0, 1)), axis = 0)) 
frame = frame / 255.0      
                                             
                                                                                                            
with torch.no_grad():                                                                                                                             
    output = vit_model(frame)   
                  

probs = torch.nn.functional.softmax(output[0], dim=0)                                   
top_prob, top_class_idx = torch.max(probs, dim=0)  

top_prob = f"{(top_prob.item() * 100):.2f}" 

categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# the predicated category
predicated_class = categories[top_class_idx]
print("Category: ", predicated_class)
print("Probability:", top_prob)




    