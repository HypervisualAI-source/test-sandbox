import numpy as np                                                                            
import cv2     
import os                                                                                          
import sys                                                                                            
import torch                                                                                                
import glob                                                                                                                                                                    
import time                                                                                                 
from fvcore.nn import FlopCountAnalysis, flop_count_str
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
                                                                                                      
                                                                                                                                                                                                                    
def demo(vit_model, test_path, categories, flops_total, all_params, model_name):                                                                                                          
                                                                                          
    y_true = []                                                                             
    y_pred = []                          
    cpu_ops_list = [] 
    fps_batch_size_1 = []    
    fps_batch_size_8 = []   
                                                                                
    test_class = os.listdir(test_path)        

    for n in range(2):                                                                                                                                                       
        if n == 0:                       
            for i in range(len(test_class)):                                 
                true_name = test_class[i]                                                                                         
                frame_num = 0                                                                   
                                                                                                                                                    
                image_data = glob.glob(test_path + test_class[i] + "/*.png")         
                                                                                                                 
                for path in image_data:                                               
                    frame = cv2.imread(path)                                                              
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                                  
                    frame = torch.from_numpy(np.expand_dims(np.transpose(frame.astype(np.float32), (2, 0, 1)), axis = 0)) 
                                                                                                
                    if model_name == "vit_model_fp32":                                                  
                        frame = frame / 255.0                                                                                                           
                    if model_name == "vit_model_bf16":                                                                 
                        frame = frame / 255.0                                                                                           
                        frame = frame.to((torch.bfloat16))              
                                                                                                                 
                    with torch.no_grad():                                                                                                                                                                                                                 
                        if frame_num == 0:                                                                                                                            
                            # input resolution                                                                                       
                            input_resolution = frame.shape    
                            print("\n") 
                            print(f"input (batch size = 1): {frame.device} {frame.dtype}\n")                          
                            output = vit_model(frame)    
                            print(f"output (batch size = 1): {output.device} {output.dtype}\n")                           
                        else:                                                                                         
                            start = time.perf_counter()                                                                        
                            output = vit_model(frame) 
                                                                                                                                                                                                  
                            end = time.perf_counter()
                            inference_time = (end - start) 
                            
                            # fps_batch_size_1                                                                                          
                            fps = 1 / inference_time
                            fps_batch_size_1.append(fps)
                            
                                                                                                                                                
                            probs = torch.nn.functional.softmax(output[0], dim=0)                                   
                            top_prob, top_class_idx = torch.max(probs, dim=0)    
                                                                                                                     
                            # get y_true
                            if true_name == categories[0]:
                                y_true.append(0)
                            if true_name == categories[1]:
                                y_true.append(1)
                            if true_name == categories[2]:
                                y_true.append(2)
                            if true_name == categories[3]:
                                y_true.append(3)
                            if true_name == categories[4]:
                                y_true.append(4)
                            if true_name == categories[5]:
                                y_true.append(5)
                            if true_name == categories[6]:
                                y_true.append(6)
                            if true_name == categories[7]:
                                y_true.append(7)
                            if true_name == categories[8]:
                                y_true.append(8)
                            if true_name == categories[9]:
                                y_true.append(9)
                                
                            # get y_pred                                                                                                                                                                  
                            y_pred.append(top_class_idx.item())
                
                            top_prob = f"{(top_prob.item() * 100):.2f}" 
                                                                             
                    frame_num += 1
                                                                                                                                                                    
        if n == 1:                                                                                                                                      
            for i in range(len(test_class)):                                                                                                 
                true_name = test_class[i]                                                                                         
                frame_num = 0                                                                   
                                                                                                                    
                image_data = glob.glob(test_path + test_class[i] + "/*.png")                     
                                                                                                                                                                                                                     
                for path in image_data:                                                                                                           
                    frame = cv2.imread(path)                                                              
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                                  
                    frame = torch.from_numpy(np.expand_dims(np.transpose(frame.astype(np.float32), (2, 0, 1)), axis = 0)) 
                    
                    if model_name == "vit_model_fp32":                                                  
                        frame = frame / 255.0                                                                                                           
                    if model_name == "vit_model_bf16":                                                                 
                        frame = frame / 255.0                                                                                           
                        frame = frame.to((torch.bfloat16))                                  
                                                                                                                                                                                             
                    # batch size = 8                                                                                                                                                                 
                    frame = frame.repeat(8, 1, 1, 1)                                                                                                                   
                                                                                                                                                                                 
                    with torch.no_grad():                                                                                                                                                                                                                 
                        if frame_num == 0: 
                            print(f"input (batch size = 8):{frame.device} {frame.dtype}\n")                                 
                            output = vit_model(frame)            
                            print(f"output (batch size = 8):{output.device} {output.dtype}\n")                                   
                        else:                                                                                         
                            start = time.perf_counter()
                                                                                                              
                            output = vit_model(frame) 
                                                                                                                                                                                                  
                            end = time.perf_counter()
                            inference_time = (end - start) 
                            
                            # fps_batch_size_8                                                                                          
                            fps = 1 / inference_time
                            fps_batch_size_8.append(fps)
       
                                                                                                                                                
                            probs = torch.nn.functional.softmax(output[0], dim=0)                                       
                            top_prob, top_class_idx = torch.max(probs, dim=0)    
                                                                                                                     
                            # get y_true
                            if true_name == categories[0]:
                                y_true.append(0)
                            if true_name == categories[1]:
                                y_true.append(1)
                            if true_name == categories[2]:
                                y_true.append(2)
                            if true_name == categories[3]:
                                y_true.append(3)
                            if true_name == categories[4]:
                                y_true.append(4)
                            if true_name == categories[5]:
                                y_true.append(5)
                            if true_name == categories[6]:
                                y_true.append(6)
                            if true_name == categories[7]:
                                y_true.append(7)
                            if true_name == categories[8]:
                                y_true.append(8)                      
                            if true_name == categories[9]:
                                y_true.append(9)
                                
                            # get y_pred                                                                                                                                                                  
                            y_pred.append(top_class_idx.item())

                            top_prob = f"{(top_prob.item() * 100):.2f}" 

                    frame_num += 1                                                                  
        
                                                                                                                    
                                                                                                     
    # indicator 1: Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")
    
    # indicator 2: FPS (batch size = 1)
    fps_b1 = int(sum(fps_batch_size_1) / len(fps_batch_size_1))
    print(f"FPS (batch size = 1): {fps_b1}\n")           

    # indicator 3: FPS (batch size = 8)
    fps_b8 = int((sum(fps_batch_size_8) / len(fps_batch_size_8)) * 8)
    print(f"FPS (batch size = 8): {fps_b8}\n")                                                                                                                                                                               
                                                                                                                                                   
    # indicator 4: Input resolution
    print(f"Input resolution: {input_resolution}\n") 
    
    # indicator 5: Params                                                                                        
    print(f"Params: {all_params}\n")              
    
    # indicator 6: OPS                                                               
    OPS = flops_total / (10**9)                                      
    print(f"OPS(G): {OPS:.4f}\n")    
                                                                      
                                                              
                                                                                                                                                                           
if __name__ == "__main__":
                                                                                                                      
    # the path of evaluation data                                                         
    test_path = "./val/"                                         
    # The categories of CIFAR-10                                                                  
    categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # load model
    model = torch.load("HVAI_MLA_ViT.pth", map_location='cpu')                                                                                                                                                          
    vit_model = model["model"] 
                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                
    vit_model.eval()      
    
    ## case 1: classifcation by model with FP32
    # model with FP32
    vit_model_fp32 = vit_model         
    # verifying dtype
    for name, param in vit_model_fp32.named_parameters():
        if name == "transformer.norm.weight":
            print("ViT with FP32:", name, param.dtype)
            break                                                                                  
                                                                                                                                                                                                                             
    # parameters                                                                                                                       
    all_params = sum(p.numel() for p in vit_model_fp32.parameters())                                                                                                                       
                                            
    # warm up stage                                               
    with torch.no_grad():                                                                                                                             
        for i in range(3):           
            dummy_frame = torch.randn(1,3,32,32).to("cpu")
            output = vit_model_fp32(dummy_frame)   

    # measure FLOPs stage
    with torch.no_grad():                  
            flops = FlopCountAnalysis(vit_model_fp32, dummy_frame)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)                           
            flops_total = flops.total()  # Total FLOPs for the batch size                                                          
    
    model_name = "vit_model_fp32"    
    # demo function                                                                                  
    demo(vit_model_fp32, test_path, categories, flops_total, all_params, model_name)                 

    print("\n")

    ## case 2: classifcation by model with BF16
    # model with BF16
    vit_model_bf16 = vit_model.to(torch.bfloat16)          
    # verifying dtype
    for name, param in vit_model_bf16.named_parameters():
        if name == "transformer.norm.weight":
            print("ViT with BF16", name, param.dtype)
            break
                                                                                                                                                                                                                             
    # parameters                                                                                                                       
    all_params = sum(p.numel() for p in vit_model_bf16.parameters())                                                                                                                       
                                            
    # warm up stage                                               
    with torch.no_grad():                                                                                                                             
        for i in range(3):           
            dummy_frame = torch.randn(1,3,32,32).to("cpu")
            dummy_frame = dummy_frame.to(torch.bfloat16)  
            output = vit_model_bf16(dummy_frame)   

    # measure FLOPs stage
    with torch.no_grad():                  
            flops = FlopCountAnalysis(vit_model_bf16, dummy_frame)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)                           
            flops_total = flops.total()  # Total FLOPs for the batch size
    
    model_name = "vit_model_bf16"   
    # demo function                                                                                  
    demo(vit_model_bf16, test_path, categories, flops_total, all_params, model_name)                                                                  
    


                                                                                                                                                                                           