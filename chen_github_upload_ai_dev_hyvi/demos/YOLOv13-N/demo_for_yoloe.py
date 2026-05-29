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

import sys
import os                                                                                                     
                                                                                                                                                                                                                                                                                                                                                         
sys.path.append(os.path.abspath('../../third_party_libraries'))        
                                                                                                                                                                                                                                                                                                                                                                      
from ultralytics import YOLO                                                                                                                


                                                                                                                                                                                                                                                                                               
def demo(yolov13_model, video_path_yolo, imgsz, conf_thres, iou_thres,  task_yolo_cp,  coco_class_names, colors_yolo , fontScale_yolo_ori, device, up_logo_cp, up_points, font, window_name, video_name, fps):
    
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, up_points)
    
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)                                                                                      
                                                                                                                                                                 
    while True:                                      
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            video.release()
            break                                                                                                                                                                                                                                                      
                                                                                                                                                                                    
        cap = cv2.VideoCapture(video_path_yolo)                                                                                     
        prev_frame_time = 0
        new_frame_time = 0                                                                                                                                                                                                                          
        yolo_display_time = []                    
        yolo_inference_time = []                                                                       
        yolo_fps_list = []
                                                                           
        frame_num = 0                                                                                                                            
        yolo_total = 0                                                              
        while True:                                                          
            ret, frame = cap.read()     
                                                                                                 
            frame = cv2.imread("/home/gpnpu/Desktop/AI_development/YOLO_E_Development/yoloe_ultralytics/frame_0692.png")
            
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue                
                                                                                                                                                     
            start = time.time()       
            results = yolov13_model.predict(
                    source=frame,
                    imgsz=imgsz,                    
                    conf=0.20,                      
                    iou=0.30,
                    device=device, 
                    show=False,
                    save = False,                                                      
                    verbose=False                                   
                )   
                                                                    
            
            end = time.time()                                                                                                                             
            inference_time =  int((end - start)  * 1000)   
            
            if frame_num == 0:
                pass
            else:
                yolo_inference_time.append(inference_time)
            
                                                                                                                                                        
            object_boxes = results[0].boxes.xyxy
            object_colors = []                                                                        
            object_names = []                
                                                                                                                     
            for i, cls_id in enumerate(results[0].boxes.cls):                                         
                name = coco_class_names[int(cls_id.item())]                     
                object_names.append(name)                                                                                                                                                                                                     
                id_object = int(cls_id.numpy())                     
                object_colors .append(colors_yolo[id_object])                      
                                                               
            object_confs_0 = (results[0].boxes.conf).numpy()
            object_confs = []                                                             
            for i in range(len(object_confs_0)):
                a = round(object_confs_0[i], 2)                                              
                object_confs.append(a)                                       
                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            thickness = 2                                                                                                                                                                                                                        
            for object_box, object_color, object_name, object_conf in zip(object_boxes, object_colors, object_names, object_confs):
                fontScale_yolo = fontScale_yolo_ori
                text =  object_name + " "  +  str(object_conf)                                                                                                                                     
                (text_width, text_height), baseline = cv2.getTextSize(text, font, fontScale_yolo, thickness)  
                                                                                                                                                                                                                                
                while (object_box[2] - object_box[0]) < text_width:                                                                                                                                                                                                                       
                    fontScale_yolo = fontScale_yolo - 0.1                                                                                   
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, fontScale_yolo, thickness)
                                                                                                                                                                    
                top_left = (int(object_box[0].numpy()), int((object_box[1]).numpy() - text_height * 1.5) )                   
                bottom_right = (int(object_box[0].numpy() + text_width), int((object_box[1]).numpy()))      

                if object_name == "person":                                                                                                                                                                                              
                    object_color = (0, 128, 0)     
                    
                fill_color = object_color                                                                
                                                                                                                                                                          
                cv2.rectangle(frame, top_left, bottom_right, fill_color, thickness=-1)
                cv2.rectangle(frame, (int(object_box[0].numpy()), int(object_box[1].numpy())), (int(object_box[2].numpy()), int(object_box[3].numpy())), object_color, 4)  
                org_0 = (int(object_box[0].numpy()), int((object_box[1] - 8).numpy()))
                
                if object_color[0] == 255 and object_color[1] == 0 and object_color[2] == 0:
                    print("*****************00 ****")
                    cv2.putText(frame, object_name + " "  +  str(object_conf), org_0, font, fontScale_yolo, [255, 255, 255], thickness, cv2.LINE_AA)
                else:
                    print("***************** 111 ****")
                    cv2.putText(frame, object_name + " "  +  str(object_conf), org_0, font, fontScale_yolo, [0,0,0], thickness, cv2.LINE_AA)                        
            
       


             
                                                     
            frame = cv2.resize(frame, up_points, interpolation= cv2.INTER_LINEAR)                                                                                                                                     
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                                                                                                                                                       
            frame = Image.fromarray(frame)                                                                                                                                                                                                                        
            frame_cp = frame.copy()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
            frame_cp.paste(up_logo_cp, (1650, 20))                                                                       
            frame_cp.paste(task_yolo_cp, (10, 20))                                                                                                                                                  
            frame = np.asarray(frame_cp)                                                                                                                                                                                                   
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
            
            
            cv2.imwrite("/home/gpnpu/Desktop/AI_development/YOLO_E_Development/yoloe_ultralytics/helmet_3_test/new_0/yolov13n_frame_0692.jpg", frame)    
            
            break                                                
        break     
    
    """    
                new_frame_time = time.time()       
                
                frame_time = new_frame_time - prev_frame_time
                fps = 1/ frame_time
                                                                                      
                prev_frame_time = new_frame_time
                fps = int(fps)                
                
                cv2.imshow(window_name, frame)
                cv2.waitKey(1)  
                
                

                print("\n")
                print("per frame at shape: "  + str((imgsz, imgsz, 3)) + "    "  + "inference: "  + str(inference_time) + ("ms") + "    " + "fps: " + str(fps) + "    " + "device: " + device)
                print("\n")                                                                                                                                   
                                                                       
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    video.release()
                    break
                                                          
                if frame_num == 0:
                    pass               
                else:                           
                    yolo_display_time.append((frame_time))
                    yolo_fps_list.append(fps)
                                 
                yolo_total = sum(yolo_display_time)    
                
                if frame_num == 0:                                                      
                    pass
                else:                                                                                                          
                    average_inference_time = int(sum(yolo_inference_time) / (len(yolo_inference_time)))
                    average_fps = int(sum(yolo_fps_list) / (len(yolo_fps_list)))            

                    print("\n")
                    print("per frame at shape: "  + str((imgsz, imgsz, 3)) + "    "  + "average_inference_time: " + str(average_inference_time) + "  " + "average_fps: " + str(average_fps))
                    print("\n")
                              

                video.write(frame)                                                                                                                                                                                                                                                                                                                                                                                  
                                          
                frame_num += 1  
    """                                                                                                                                                                                   
                                                          
                                                                                                                           
if __name__ == "__main__":
    

    # read the configuration parameters from cofig.yal file
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # video source     
    video_path_yolo = config["video_source"]["video_path_yolo"]
                                                  
    # the neccessary file of YOLov13-N
    yolov13_weights_path = config["yolov13_weight_cfg"]["yolov13_weights_path"]
    yolov13_cfg_path = config["yolov13_weight_cfg"]["yolov13_cfg_path"]
                        
               
    # check the necessary files
    if not os.path.exists(yolov13_weights_path):                           
        raise FileNotFoundError(f"'{yolov13_weights_path}' YOLOs' weights file not found. Please download it and place it in the correct path.")
    if not os.path.exists(video_path_yolo):
        raise FileNotFoundError(f"'{video_path_yolo}' video file not found. Please provide a video to test.")
    if not os.path.exists(yolov13_cfg_path):
        raise FileNotFoundError(f"'{yolov13_cfg_path}' yolov13 configuration file not found. Please provide a configuration file.")
                                                                                                   
                                                                                      
    # optional device                          
    device = "cpu"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                 
    # load YOLOv13-N                                                                                                                                            
    yolov13_model = YOLO(yolov13_cfg_path)
    yolov13_model.load(yolov13_weights_path)                                                                                                                                                                                                                                                                   
                                                                                                         
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
    text_area_yolo = tuple(config["title"]["text_area_yolo"])
    task_background_yolo = np.full(text_area_yolo, text_background_color, dtype=np.uint8)

    # task 0: YOLOv13-N  detection
    text_yolo = config["task_yolo"]["task_text"]
    task_yolo = cv2.putText(task_background_yolo, text_yolo,  org,  font, fontScale, text_color, thickness, cv2.LINE_AA)
    task_yolo= cv2.cvtColor(task_yolo, cv2.COLOR_BGR2RGB)                                             
    task_yolo = Image.fromarray(task_yolo)  
    task_yolo_cp = task_yolo.copy()                                      

                                                                                                                                            
    #  the features of YOLOv13-N detection                                                                                                                                                              
    imgsz = config["yolov13_features"]["imgsz"]
    conf_thres = config["yolov13_features"]["conf_thres"]
    iou_thres = config["yolov13_features"]["iou_thres"]
    fontScale_yolo_ori = config["yolov13_features"]["fontScale_yolo_ori"]
    colors_yolo = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]
    
    # the distinct colors
    colors_yolo[0] = config["distinct_colors"]["colors_0"]
    colors_yolo[1] = config["distinct_colors"]["colors_1"]
    colors_yolo[2] = config["distinct_colors"]["colors_2"]
    colors_yolo[3] = config["distinct_colors"]["colors_3"]
    colors_yolo[4] = config["distinct_colors"]["colors_4"]
    colors_yolo[5] = config["distinct_colors"]["colors_5"]
    colors_yolo[6] = config["distinct_colors"]["colors_6"]
    colors_yolo[7] = config["distinct_colors"]["colors_7"]

    # display resolution: 1920 x 1080
    up_width = config["display_resolution"]["up_width"]
    up_height = config["display_resolution"]["up_height"]
    up_points = (up_width, up_height)                                              
                                                                                                                                                             
    # the class name of COCO
    coco_path = config["coco"]["coco_path"]
    with open(coco_path, "r") as f:
        coco_class_names = [line.strip() for line in f .readlines()]
                                                                         
    # the name of window                                    
    window_name = config["window"]["window_name"]
    
    # record the video                                                                                                                                                 
    video_name = config["output_video"]["video_name"]
    fps = config["output_video"]["fps"]

    # demo function                                                                                                                         
    demo(yolov13_model, video_path_yolo, imgsz, conf_thres, iou_thres,  task_yolo_cp,  coco_class_names, colors_yolo , fontScale_yolo_ori, device, up_logo_cp, up_points, font, window_name, video_name, fps)

