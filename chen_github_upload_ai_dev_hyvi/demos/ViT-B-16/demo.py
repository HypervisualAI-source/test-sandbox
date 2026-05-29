import numpy as np                                                                            
import cv2     
import os    
import sys                
import torch                                                        
import copy                                                                                                                                                                                          
from PIL import Image                                                                                     
import time                                                                                   
import random                                                                                                                                          
import yaml                                
                                                                                                                                                                                  
import torchvision.models as models   
                                                                                                                                                                                                                                                                                               
def demo(vit_model,  video_path_vit_list, task_vit_cp, weights, fontScale_vit, thickness_vit, org_vit_category, org_vit_probability, color_vit, device, up_logo_cp, up_points, font, window_name, video_name, fps):
    
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, up_points)
    
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
                                                                                      
    while True:                                                                                                                                                           
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break                                                             
                                                                                                                  
                                                                                                                                    
        for i in range(len(video_path_vit_list)) :   
            cap = cv2.VideoCapture(video_path_vit_list[i])   
            vit_display_time = []                
            vit_inference_time = []                                                                       
            vit_fps_list = []                                                                                             
            frame_num = 0                      
            
            prev_frame_time = 0
            new_frame_time = 0                
                                                                                                                                                                          
            while True:                                                                                                                                                                                                                                                               
                ret, frame = cap.read()                                                                                                                                                                                 
                                                                                                                                          
                frame_ori = frame.copy()                                                                                                                                                       
                frame = cv2.resize(frame, (224, 224), interpolation= cv2.INTER_LINEAR)                                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                                  
                frame = torch.from_numpy(np.expand_dims(np.transpose(frame.astype(np.float32), (2, 0, 1)), axis = 0)) 
                frame = frame / 255.0
                                          
                start = time.time()                                                    
                output = vit_model(frame)  
                end = time.time()
                inference_time =  int((end - start)  * 1000)  
                
                if frame_num == 0:
                    pass
                else:
                    vit_inference_time.append(inference_time)
                
                
                probs = torch.nn.functional.softmax(output[0], dim=0)
                top_prob, top_class_idx = torch.max(probs, dim=0)                                                                                                                                                                                                                                   
                top_prob = f"{(top_prob.item() * 100):.2f}"                                                      
                label = weights.meta["categories"][top_class_idx]
                frame = cv2.resize(frame_ori, up_points, interpolation= cv2.INTER_LINEAR)
                frame = cv2.putText(frame, "Category: " +  label, org_vit_category, font, fontScale_vit, color_vit, thickness_vit, cv2.LINE_AA)
                frame = cv2.putText(frame, "Probability: " +  top_prob + "%",  org_vit_probability, font, fontScale_vit, color_vit, thickness_vit, cv2.LINE_AA)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                    
                frame = Image.fromarray(frame)                                                                                                                                                                                                                                                                                                                
                frame_cp = frame.copy()                                                                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                      
                frame_cp.paste(up_logo_cp, (1650, 20))                                                                                                                                                                                                                       
                frame_cp.paste(task_vit_cp, (10, 20))                                                                                                                                                                                                                                                                                  
                frame = np.asarray(frame_cp)                                                                                           
                                                                                                                                                                  
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                display_frame = frame.copy() 
                
                new_frame_time = time.time()                                                                     
                
                frame_time = (new_frame_time-prev_frame_time)
                fps = 1/frame_time
                prev_frame_time = new_frame_time
                fps = int(fps)                 
                                                                                                                                                                
                cv2.imshow(window_name, display_frame)
                cv2.waitKey(1)                                                                                                                                                             
                                                                      
                print("\n")
                print("per frame at shape: "  + str((224, 224, 3)) + "    "  + "inference: "  + str(inference_time) + ("ms") + "    " + "fps: " + str(fps) + "    " + "device: " + device)
                print("\n")                                                
                                                                                                
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break     
                    
                if frame_num == 0:                   
                    pass
                else:                                                                                                                                                                                              
                    vit_display_time.append((frame_time))                        
                    vit_fps_list.append(fps)                                      
                    
                vit_total = sum(vit_display_time) 
                                                                                                      
                if vit_total > 5:
                    average_inference_time = int(sum(vit_inference_time) / (len(vit_inference_time)))
                    average_fps = int(sum(vit_fps_list) / (len(vit_fps_list)))            
                    
                    print("\n")                           
                    print("============================================ average inference time and average fps =================================")
                    print("per frame at shape: "  + str((224, 224, 3)) + "    "  + "average_inference_time: " + str(average_inference_time) + "  " + "average_fps: " + str(average_fps))
                    print("\n")
                    
                    break                     
                                                                                                              
                video.write(frame) 
                frame_num += 1                  
                                           
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                           
if __name__ == "__main__":
                                                                                                                                                                     
      
    # read the configuration parameters from cofig.yal file
    with open('config.yml', 'r') as file:                                        
        config = yaml.safe_load(file)

    # video source     
    video_path_vit_0 = config["video_source"]["video_path_vit_0"]
    video_path_vit_1 = config["video_source"]["video_path_vit_1"]
    video_path_vit_2 = config["video_source"]["video_path_vit_2"]
    video_path_vit_3 = config["video_source"]["video_path_vit_3"]
    video_path_vit_4 = config["video_source"]["video_path_vit_4"]
    video_path_vit_5 = config["video_source"]["video_path_vit_5"]
    video_path_vit_6 = config["video_source"]["video_path_vit_6"]
    video_path_vit_7 = config["video_source"]["video_path_vit_7"]

    # make a list of videos for vit demo                                                                                                        
    video_path_vit_list  = []                                                                                                                                                                    
    video_path_vit_list.append(video_path_vit_0)
    video_path_vit_list.append(video_path_vit_1)
    video_path_vit_list.append(video_path_vit_2)
    video_path_vit_list.append(video_path_vit_3)
    video_path_vit_list.append(video_path_vit_4)
    video_path_vit_list.append(video_path_vit_5)
    video_path_vit_list.append(video_path_vit_6)
    video_path_vit_list.append(video_path_vit_7)
                                                  
    # check the necessary files
    if not os.path.exists(video_path_vit_0):
        raise FileNotFoundError(f"'{video_path_vit_0}' video file not found. Please provide a video to test.")   
    if not os.path.exists(video_path_vit_1):
        raise FileNotFoundError(f"'{video_path_vit_1}' video file not found. Please provide a video to test.")   
    if not os.path.exists(video_path_vit_2):
        raise FileNotFoundError(f"'{video_path_vit_2}' video file not found. Please provide a video to test.") 
    if not os.path.exists(video_path_vit_3):
        raise FileNotFoundError(f"'{video_path_vit_3}' video file not found. Please provide a video to test.") 
    if not os.path.exists(video_path_vit_4):
        raise FileNotFoundError(f"'{video_path_vit_4}' video file not found. Please provide a video to test.")  
    if not os.path.exists(video_path_vit_5):
        raise FileNotFoundError(f"'{video_path_vit_5}' video file not found. Please provide a video to test.") 
    if not os.path.exists(video_path_vit_6):
        raise FileNotFoundError(f"'{video_path_vit_6}' video file not found. Please provide a video to test.")     
    if not os.path.exists(video_path_vit_7):
        raise FileNotFoundError(f"'{video_path_vit_7}' video file not found. Please provide a video to test.")          

                             
    # optional device
    device = "cpu"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
                                                                                                         
    #  load ViT-B-16                                                                                                                                                                                                                                                                                  
    vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    vit_model.eval()    
                                                                                                                                            
    weights=models.ViT_B_16_Weights.IMAGENET1K_V1   
                                                                                                         
    # logo
    logo_path = config["logo"]["logo_path"]
    logo = cv2.imread(logo_path)
    up_width = int(logo.shape[1] * config["logo"]["up_width"])
    up_height = int(logo.shape[0] * config["logo"]["up_width"])
    up_points = (up_width, up_height)                       
    up_logo = cv2.resize(logo, up_points, interpolation= cv2.INTER_LINEAR)       
    up_logo = cv2.cvtColor(up_logo, cv2.COLOR_BGR2RGB)
    up_logo = Image.fromarray(up_logo)
    up_logo_cp = up_logo.copy()                                            

    # title
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = config["title"]["fontScale"]
    thickness = config["title"]["thickness"]
    org = tuple(config["title"]["org"])
    text_color = tuple(config["title"]["text_color"])
    text_background_color = config["title"]["text_background_color"]
    text_area_vit = tuple(config["title"]["text_area_vit"])
    task_background_vit = np.full(text_area_vit, text_background_color, dtype=np.uint8)
                               
    # task 0: ViT-B-16 classification                       
    text_vit = config["task_vit"]["task_text"]
    task_vit = cv2.putText(task_background_vit, text_vit,  org, font, fontScale, text_color, thickness, cv2.LINE_AA)
    task_vit = cv2.cvtColor(task_vit, cv2.COLOR_BGR2RGB)
    task_vit = Image.fromarray(task_vit)  
    task_vit_cp = task_vit.copy()

    #  the display features for ViT-B-16                                                                                                                                                                  
    fontScale_vit = config["display_ViT"]["fontScale_vit"]
    thickness_vit = config["display_ViT"]["thickness_vit"]
    org_vit_category = config["display_ViT"]["org_vit_category"]
    org_vit_probability = config["display_ViT"]["org_vit_probability"]
    color_vit = config["display_ViT"]["color_vit"]
                                                                                                                                              
    # display resolution: 1920 x 1080
    up_width = config["display_resolution"]["up_width"]
    up_height = config["display_resolution"]["up_height"]
    up_points = (up_width, up_height)                                              

    # the name of window
    window_name = config["window"]["window_name"]
                                                               
    # record the video                                                                                                                                                 
    video_name = config["output_video"]["video_name"]
    fps = config["output_video"]["fps"]
                                                   
                              
    # demo function                                                                                  
    demo(vit_model,  video_path_vit_list, task_vit_cp, weights, fontScale_vit, thickness_vit, org_vit_category, org_vit_probability, color_vit, device, up_logo_cp, up_points, font, window_name, video_name, fps)

