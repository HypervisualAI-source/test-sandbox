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
                                                                                    
from ultralytics import YOLOE
                                                                                                                                                            
COLOR_FAMILY_RANGES = {                                                                                                                                          
    "black":  ((0, 0, 0),        (179, 255, 46)),
    "white":  ((0, 0, 217),      (179, 36, 255)),
    "red":    ((0, 71, 29),      (10, 255, 255)),        # lower red
    "red_wrap": ((170, 71, 29),  (179, 255, 255)),       # upper red (wrap-around)
    "lime":   ((45, 55, 100),    (85, 255, 255)),
    "green":  ((60, 50, 0),      (90, 255, 255)),
    "blue":   ((100, 45, 50),    (130, 255, 255)),
    "yellow": ((20, 40, 100),    (40, 255, 255)),
    "aqua":   ((85, 50, 55),     (100, 255, 255)),
    "magenta":((140, 50, 50),    (170, 255, 255)),
    "silver": ((0, 0, 180),      (179, 30, 220)),
    "gray":   ((0, 0, 80),       (179, 40, 180)),
    "maroon": ((0, 50, 20),      (10, 255, 165)),
    "olive":  ((25, 50, 40),     (45, 255, 170)),
    "purple": ((130, 50, 50),    (155, 255, 255)),
    "teal":   ((85, 50, 50),     (100, 255, 200)),
    "navy":   ((110, 50, 0),     (130, 255, 180)),                                                                                                            
}                                                                                                             
                                                                                                                             
def detect_color_family(crop):
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    scores = {}
    for family in COLOR_FAMILY_RANGES.keys():
        if family == "red_wrap": 
            continue
        if family == "red":
            lower1, upper1 = COLOR_FAMILY_RANGES["red"]
            lower2, upper2 = COLOR_FAMILY_RANGES["red_wrap"]
            mask1 = cv2.inRange(hsv_crop, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv_crop, np.array(lower2), np.array(upper2))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower, upper = COLOR_FAMILY_RANGES[family]
            mask = cv2.inRange(hsv_crop, np.array(lower), np.array(upper))
        scores[family] = cv2.countNonZero(mask)
    
    return max(scores, key=scores.get)
                                                       
                                                                                                                                                                                                                                                                                          
def demo(model, video_path_yolo, imgsz, conf_thres, iou_thres,  task_yolo_cp,  all_category, colors_yolo , fontScale_yolo_ori, device, up_logo_cp, up_points, font, window_name, bgr_helmet):
    
    model.set_classes(all_category, model.get_text_pe(all_category))   
    
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)    
    
    while True:    
        for v in range(len(video_path_yolo)):  
            cap = cv2.VideoCapture(video_path_yolo[v])                                                                                     
            prev_frame_time = 0                                                                                                                  
            new_frame_time = 0                                                                                                                                                                                                                          
            yolo_display_time = []                                                                                          
            yolo_inference_time = []                                                                       
            yolo_fps_list = []
                                                                               
            frame_num = 0                                                                                                                            
            yolo_total = 0      
            
            while True:                                                          
                ret, frame = cap.read()                              
                
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue   
                
                
                    
                frame_copy = frame.copy()
                start = time.time()       
                results = model.predict(                                                            
                        source=frame,                                                  
                        imgsz=imgsz,                    
                        conf=conf_thres,                      
                        iou=iou_thres,
                        device=device,                                                     
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
                    name = all_category[int(cls_id.item())]                     
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
                    if object_name == "helmet":  
                        helmet_crop = frame_copy[int(object_box[1]) : int(object_box[3]), int(object_box[0]) : int(object_box[2])]
                        color_name = detect_color_family(helmet_crop)
                        
                        fontScale_yolo = 0.7                                                               
                        text = color_name + " helmet"  +  str(object_conf)                                                         
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, fontScale_yolo, thickness)  
                        
                        object_box_copy_2 = object_box[2].clone()                                                          
                        while (object_box_copy_2 - object_box[0]) < text_width:                                                                 
                            object_box_copy_2 = object_box_copy_2 + 30                                                                                                           
                                                                                                                                                                                                                                                            
                        top_left = (int(object_box[0].numpy()), int((object_box[1]).numpy() - text_height * 1.5) )                 
                        bottom_right = (int(object_box_copy_2), int((object_box[1]).numpy()))   
                        
                        if color_name == "red" or color_name == "red_wrap":
                             color_name == "red"
                             
                        for n in range(len(bgr_helmet)):
                            if color_name == bgr_helmet[n][0]:
                                object_color = bgr_helmet[n][1]                                                                                                                                                            
                                                                                                                                   
                    else:                                                                                                                                                                                                                           
                        fontScale_yolo = fontScale_yolo_ori                                                                                                     
                        text =  object_name + " "  +  str(object_conf)                                                         
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, fontScale_yolo, thickness)                                                                                                                                                                                                                  
                        while (object_box[2] - object_box[0]) < text_width:                                                                                                                                                                                                                       
                            fontScale_yolo = fontScale_yolo - 0.25                                                                                                            
                            (text_width, text_height), baseline = cv2.getTextSize(text, font, fontScale_yolo, thickness)    
                                                                  
                        top_left = (int(object_box[0].numpy()), int((object_box[1]).numpy() - text_height * 1.5) )                 
                        bottom_right = (int(object_box[2].numpy() ), int((object_box[1]).numpy()))      
                                                                                                                                                                                                                                                                                                       
                    cv2.rectangle(frame, top_left, bottom_right, object_color, thickness=-1)                                                              
                    cv2.rectangle(frame, (int(object_box[0].numpy()), int(object_box[1].numpy())), (int(object_box[2].numpy()), int(object_box[3].numpy())), object_color, 4)                                                                                                                                                                     
                    org_0 = (int(object_box[0].numpy()), int((object_box[1] - 8).numpy()))                                                                                      
                                                                                                                           
                    if object_color[0] == 0 and object_color[1] == 0 and object_color[2] == 0:                                                                                                                                      
                        cv2.putText(frame, text, org_0, font, fontScale_yolo, [255, 255, 255], thickness, cv2.LINE_AA)                      
                    else:                                                                                                                                                                                                                                                                                                                                                                                                                                                        
                        cv2.putText(frame, text, org_0, font, fontScale_yolo, [0, 0, 0], thickness, cv2.LINE_AA)        
                                                                  
                frame = cv2.resize(frame, up_points, interpolation= cv2.INTER_LINEAR)       
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                                                                                                                                                       
                frame = Image.fromarray(frame)                                                                                                                                                                                                                        
                frame_cp = frame.copy()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                frame_cp.paste(up_logo_cp, (1650, 20))                                                                       
                frame_cp.paste(task_yolo_cp, (10, 20))                                                                                                                                                  
                frame = np.asarray(frame_cp)                                                                                                                                                                                                   
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
                                                                                  
                new_frame_time = time.time()       
                
                frame_time = new_frame_time - prev_frame_time
                fps = 1/ frame_time                                                                     
                prev_frame_time = new_frame_time
                fps = int(fps)                      
                                                                                                                                             
                cv2.imshow(window_name, frame)                                                                               
                cv2.waitKey(1)  
                                
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
                    print("per frame at shape: "  + str((imgsz, imgsz, 3)) + "    "  + "average_inference_time: " + str(average_inference_time) + "  " + "average_fps: " + str(average_fps) + "  " +  "device: " + device)
                    print("\n")                                       
                                                                                                                                         
                frame_num += 1                                                                                                                                                           
                                                                                                                
                if yolo_total > 10:                                                                
                    break                
                                                         
                                                                              
                                                                              
                                                                                                                                                                                                   
if __name__ == "__main__":                                                                                                                                                                                                    
                                                                                                                                                                                    
    # the basic 16 color classes   
    bgr_helmet = [["black", (0,0,0)], ["white", (255,255,255)], ["red", (0,0,255)], ["lime", (0,255,0)], ["blue", (255,0,0)], ["yellow", (0,255,255)], ["aqua", (255,255,0)], ["magenta", (255,0,255)], ["silver", (192,192,192)], ["gray", (128,128,128)], ["maroon", (0,0,128)], ["olive", (0,128,128)], ["green", (0,128,0)], ["purple", (128,0,128)], ["teal", (128,128,0)], ["navy", (128,0,0)]]    
   
    reservation_color = []
    for i in range(len(bgr_helmet)):
        reservation_color.append(bgr_helmet[i][1])
   
    with open('config.yml', 'r') as file:                                                                                                                                      
        config = yaml.safe_load(file)                                                                                                                                                              
                                                                                                                                                                 
    # video sources                                                                                                                       
    video_path_yolo = []                                                                                            
    video_path_yolo.append(config["video_source"]["video_1"])
    video_path_yolo.append(config["video_source"]["video_2"])       
    
    # model                            
    model = YOLOE(config["model"]["weight_path"])                                                                                                                                   
                                                                                                                                                                                                 
    # device                                                                                             
    device = "cpu"   
                                                                                                                                                                                                                      
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

    # task 0: YOLOE-11S-N  detection
    text_yolo = config["task_yolo"]["task_text"]                                                                                                                                            
    task_yolo = cv2.putText(task_background_yolo, text_yolo,  org,  font, fontScale, text_color, thickness, cv2.LINE_AA)
    task_yolo= cv2.cvtColor(task_yolo, cv2.COLOR_BGR2RGB)                                             
    task_yolo = Image.fromarray(task_yolo)                                                                                                       
    task_yolo_cp = task_yolo.copy()                                                              
                                                                                                  
                                                                                                                                               
    #  the features of YOLOE-11S detection                                                                                                                                                              
    imgsz = config["yolo_features"]["imgsz"]                                                                                                                                           
    conf_thres = config["yolo_features"]["conf_thres"]                                                                                                                                 
    iou_thres = config["yolo_features"]["iou_thres"]
    fontScale_yolo_ori = config["yolo_features"]["fontScale_yolo_ori"]                                                            
    colors_yolo = [[random.randint(0, 255) for _ in range(3)] for _ in range(96)]
    
    # keep reservation colors                                                                            
    for i in range(len(reservation_color)):
        while (reservation_color[i] in colors_yolo):
            ran_num = random.randint(10, 30)
            reservation_color[i][0] = min(reservation_color[i][0] + ran_num, 255)
            reservation_color[i][1] = min(reservation_color[i][1] + ran_num, 255)
            reservation_color[i][2] = min(reservation_color[i][2] + ran_num, 255)
                                                                                                                                                    
    # display resolution: 1920 x 1080
    up_width = config["display_resolution"]["up_width"]                                                                                                                                           
    up_height = config["display_resolution"]["up_height"]                                                                                                                               
    up_points = (up_width, up_height)                                                                                                        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    # the class name of COCO                                                                                                                                                                                              
    coco_path = config["coco"]["coco_path"]                                                                                                     
    with open(coco_path, "r") as f:                                                                                                                                                                                                            
        coco_category = [line.strip() for line in f .readlines()]                                                           
                                                                                                                                         
    # the deteciotn categories                                                                                                                                              
    all_category = coco_category + ["helmet"]                                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                           
    # the name of window                                                                                                                                                                                          
    window_name = config["window"]["window_name"]                                                                                         
                                                                        
                                                                                   
    # demo function 
    while True:                                                                                                                                                                       
        demo(model, video_path_yolo, imgsz, conf_thres, iou_thres,  task_yolo_cp,  all_category, colors_yolo , fontScale_yolo_ori, device, up_logo_cp, up_points, font, window_name, bgr_helmet)




