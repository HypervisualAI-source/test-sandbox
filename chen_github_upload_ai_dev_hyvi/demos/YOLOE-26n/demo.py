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
                                                                                                                                                                                
# the color ranges for helmet
COLOR_FAMILY_RANGES = {                                                                                                                                          
    "black":  ((0, 0, 0),        (179, 255, 46)),
    "white":  ((0, 0, 217),      (179, 36, 255)),
    "red":    ((0, 71, 29),      (10, 255, 255)),    # lower red
    "red_wrap": ((170, 71, 29),  (179, 255, 255)),   # upper red (wrap-around)
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
                                                                  
                                                                                                                     
# the color ranges for vehicles   to video 26                                                                                                                                
VEHICLES_COLOR_FAMILY_RANGES = {                                                                                                                                                      
    "black":   ((100, 45, 50),    (130, 255, 255)),                                                                                                                
    "white":  ((0, 0, 217),      (179, 36, 255)),
    "red":    ((0, 71, 29),      (10, 255, 255)),    # lower red
    "red_wrap": ((170, 71, 29),  (179, 255, 255)),   # upper red (wrap-around)
    "red_wrap_1":((140, 45, 50),    (170, 255, 255)),
    "lime":   ((45, 55, 100),    (85, 255, 255)),
    "green":  ((60, 50, 0),      (90, 255, 255)),
    "blue":  ((0, 0, 0),        (179, 255, 46)),
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
                                        
                                                                                                                                                                                             
                                                                
# the color detection for helmet                                                                                                                         
def vehicles_detect_color_family(crop):
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    scores = {}
    for family in VEHICLES_COLOR_FAMILY_RANGES.keys():
        if family == "red_wrap":                                                                                          
            continue
        if family == "red_wrap_1":                                                                                          
            continue

        if family == "red":
            lower1, upper1 = VEHICLES_COLOR_FAMILY_RANGES["red"]
            lower2, upper2 = VEHICLES_COLOR_FAMILY_RANGES["red_wrap"]
            lower3, upper3 = VEHICLES_COLOR_FAMILY_RANGES["red_wrap_1"]
            mask1 = cv2.inRange(hsv_crop, np.array(lower1), np.array(upper1))
            mask2 = cv2.inRange(hsv_crop, np.array(lower2), np.array(upper2))
            mask3 = cv2.inRange(hsv_crop, np.array(lower3), np.array(upper3))
            mask = cv2.bitwise_or(mask1, mask2)
            mask = cv2.bitwise_or(mask, mask3)                                                                 
         
        else:                                                                                                                                            
            lower, upper = VEHICLES_COLOR_FAMILY_RANGES[family]
            mask = cv2.inRange(hsv_crop, np.array(lower), np.array(upper))
        scores[family] = cv2.countNonZero(mask)
                                                               
    return max(scores, key=scores.get)
    

                                                                                                                                                                  
def demo(model, video_path_yolo, imgsz, conf_thres, iou_thres,  all_category, coco_color, fontScale_yolo_ori, device, up_points, font, window_name, bgr_classes):
    
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
                start = time.perf_counter()    
                                                                                    
                results = model.predict(                                                            
                        source=frame,                                                  
                        imgsz=imgsz,        # the images size detected 
                        rect=False,                                                     
                        conf=conf_thres,                      
                        iou=iou_thres,                                                                                                                                                     
                        device=device,                                                     
                        verbose=False,                                                      
                        retina_masks=True                        
                )                                                                                                                                                                                                       
                                               
         
                end = time.perf_counter()                                                                                                                    
                inference_time =  int((end - start) * 1000) 
                                 
                if frame_num < 3:    #warm-up stage
                    pass
                else:
                    yolo_inference_time.append(inference_time)
                    
                                                  
                if len(results[0].boxes) == 0:                                        
                    pass
                else:
                    object_boxes = results[0].boxes.xyxy                                                                    
                    object_names = []                                                                                  
                    id_classes = []          
                                                                                                                                                   
                    object_confs_0 = (results[0].boxes.conf).numpy()
                    object_confs = []                                                             
                    for i in range(len(object_confs_0)):
                        a = round(object_confs_0[i], 2)                    
                        object_confs.append(a)     

                    mask_data = []
                                                                                                                             
                    for i, cls_id in enumerate(results[0].boxes.cls):                                         
                        name = all_category[int(cls_id.item())]                                                                                                                                           
                        object_names.append(name)                                                                                                                                                                                                     
                        id_object = int(cls_id.numpy())   
                        id_classes.append(id_object)              
                                                                                                                      
                        if name == "helmet":                                                     
                            mask_data.append(None)                                                                                                     
                        else:                                                                                                                                                          
                            mask_data.append(results[0].masks[i].data.cpu().numpy().squeeze())
                                                                                                                                                                                                  
                    thickness = 1                                                                                                                                            
                                                                                                                         
                                                                                                                                                                                                                                             
                    for object_box, object_name, object_conf, mask, id_object in zip(object_boxes, object_names, object_confs, mask_data, id_classes):
                                                                                          
                        fontScale_yolo = 0.5                                                                                                                                                                                                  
                                                                                                                                                                                                 
                        if object_name == "car" or object_name == "bus" or object_name == "truck" :                                                                                                                                              
                            ys, xs = np.nonzero(mask)                                                                                      
                            img_pixels = frame[ys, xs]                                                                                                                                                      
                            img_pixels = img_pixels.reshape(-1,1,3)                                                                                             
                            color_name = vehicles_detect_color_family(img_pixels)                                                         
                                                                                                                    
                        if object_name == "helmet":                                        
                            helmet_crop = frame_copy[int(object_box[1]) : int(object_box[3]), int(object_box[0]) : int(object_box[2])]
                            color_name = detect_color_family(helmet_crop) 
                                                                                                                                                                                                   
                        if object_name == "car" or object_name ==  "bus" or object_name == "truck" or object_name == "helmet":
                            text = color_name + " " + object_name +  " " + str(object_conf)  
                        else:                                     
                            text =  object_name +  " " + str(object_conf)  
                                                                                                                                     
                        (text_width, text_height), baseline = cv2.getTextSize(text, font, fontScale_yolo, thickness)  
                                                                                                                                                                                         
                                                                                                      
                        object_box_copy_0 = int(object_box[0].clone().numpy())
                        object_box_copy_1 = int(object_box[1].clone().numpy())   
                        object_box_copy_2 = int(object_box[2].clone().numpy())   
                        object_box_copy_3 = int(object_box[3].clone().numpy())   

                                                                 
                        top_left = (int(object_box_copy_0), int(object_box_copy_1 - text_height * 1.2) )                   
                        bottom_right = (int(object_box_copy_0 + text_width), int(object_box_copy_1))                                  
                                                                                                                                   
                        if object_name == "car" or object_name ==  "bus" or object_name == "truck" or object_name == "helmet":
                            if color_name == "red" or color_name == "red_wrap":
                                 color_name == "red"                 
                                                                                                                                                                                             
                            for n in range(len(bgr_classes)):                                                                                                                           
                                if color_name == bgr_classes[n][0]:                                                                                                                                      
                                    object_color = bgr_classes[n][1]                                                                                                     
                        else:                                                                                                                             
                            object_color = coco_color[id_object]                
                                                                                                                                    
                        cv2.rectangle(frame, top_left, bottom_right, object_color, thickness=-1)                                                              
                        cv2.rectangle(frame, (int(object_box_copy_0), int(object_box_copy_1)), (int(object_box_copy_2), int(object_box_copy_3)), object_color, 4)                                                                                                                                                                     
                        org_0 = (int(object_box_copy_0), int(object_box_copy_1))                                                                                      
                                                                                                                               
                        if object_color[0] == 0 and object_color[1] == 0 and object_color[2] == 0:                                                                                                                                      
                            cv2.putText(frame, text, org_0, font, fontScale_yolo, [255, 255, 255], thickness, cv2.LINE_AA)
                        elif object_color[0] == 255 and object_color[1] == 0 and object_color[2] == 0:                                                                                                         
                            cv2.putText(frame, text, org_0, font, fontScale_yolo, [255, 255, 255], thickness, cv2.LINE_AA)                        
                        else:                                                                                                                                                                                                                                                                                                                                                                                                                                                        
                            cv2.putText(frame, text, org_0, font, fontScale_yolo, [0, 0, 0], thickness, cv2.LINE_AA)                          
                                                                                             
                frame = cv2.resize(frame, up_points, interpolation= cv2.INTER_LINEAR)                                                                                                                 
                                                                                                
                                                                                                                                                                                       
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
                                               

                if frame_num < 3:                                                                                      
                    pass                                                                                                                                                        
                else:                                                                                                          
                    average_inference_time = int(sum(yolo_inference_time) / (len(yolo_inference_time)))
                    average_fps = int(sum(yolo_fps_list) / (len(yolo_fps_list)))                                                           
                                                                                                                                                   
                    print("\n")                                                                    
                    print("per frame at shape (for model): "  + str([imgsz[0], imgsz[1], 3]) +  "    "  + "average_inference_time (for model): " + str(average_inference_time) + "  " + "average_fps (for display): " + str(average_fps) + "  " +  "device: " + device)
                    print("\n")     
                
                                                                                                                            
                frame_num += 1      
                    
             
                if yolo_total > 10:                                                    
                    break                   
                                                                                                                             
                                                                                                    
              
def demo_arg(model, video_path_yolo, imgsz, conf_thres, iou_thres,  detect_obj_name, detect_obj_color, coco_category, object_colors, fontScale_yolo_ori, device, up_points, font, window_name, bgr_classes):
                                                             
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
                start = time.perf_counter()                                               
                                                                                    
                results = model.predict(                                                            
                        source=frame,                                                  
                        imgsz=imgsz,        # the images size detected 
                        rect=False,                                                     
                        conf=conf_thres,                      
                        iou=iou_thres,                                                                                                                                                     
                        device=device,                                                     
                        verbose=False,                                                      
                        retina_masks=True                        
                )                                                                                                                                                                                                       
                                                                     
                                                                                                                                                                                                                                            
                                 
                end = time.perf_counter()                                                                                                                    
                inference_time =  int((end - start) * 1000) 
                                 
                if frame_num < 3:    #warm-up stage
                    pass
                else:
                    yolo_inference_time.append(inference_time)
                    
                                                  
                if len(results[0].boxes) == 0:                                        
                    pass
                else:
                    object_boxes = results[0].boxes.xyxy                                                                    
                    object_names = []                                                                                  
                    id_classes = []          
                                                                                                                                                   
                    object_confs_0 = (results[0].boxes.conf).numpy()
                    object_confs = []                                                             
                    for i in range(len(object_confs_0)):
                        a = round(object_confs_0[i], 2)                    
                        object_confs.append(a)     

                    mask_data = []
                                                                                                                             
                    for i, cls_id in enumerate(results[0].boxes.cls):                                         
                        name = detect_obj_name[int(cls_id.item())]                                                                                                                                           
                        object_names.append(name)                                                                                                                                                                                                     
                        id_object = int(cls_id.numpy())   
                        id_classes.append(id_object)              
                                                                                                                      
                        if name == "helmet":                                                     
                            mask_data.append(None)                                                                                                     
                        else:                                                                                                                                                          
                            mask_data.append(results[0].masks[i].data.cpu().numpy().squeeze())
                                                                                                                                                                                                  
                    thickness = 1                                                                                                                                            

                    for object_box, object_name, object_conf, mask, id_object in zip(object_boxes, object_names, object_confs, mask_data, id_classes):
                                                                                          
                        fontScale_yolo = 0.5                                                                                                                                                                                                  
                                                                                                        
                        if object_name == "car" or object_name ==  "bus" or object_name == "truck" or object_name == "helmet":
                            if object_name == "car" or object_name == "bus" or object_name == "truck" :  
                                ys, xs = np.nonzero(mask)                                                                                      
                                img_pixels = frame[ys, xs]                                                                                                                                                      
                                img_pixels = img_pixels.reshape(-1,1,3)                                                                                             
                                color_name_detected = vehicles_detect_color_family(img_pixels)  
                                
                                if detect_obj_color[0] in ["black", "white", "red", "lime", "blue", "yellow", "aqua", "magenta", "silver", "gray", "maroon", "olive", "green", "purple", "teal", "navy"]:    
                                    if detect_obj_color[0] == color_name_detected:
                                        color_name = color_name_detected                        
                                        text = color_name + " " + object_name +  " " + str(object_conf)              
                                        
                                        (text_width, text_height), baseline = cv2.getTextSize(text, font, fontScale_yolo, thickness)  
                                                                                                                                                                                                                                                                                                                                                                                                                                      
                                        object_box_copy_0 = int(object_box[0].clone().numpy())
                                        object_box_copy_1 = int(object_box[1].clone().numpy())                                                                                              
                                        object_box_copy_2 = int(object_box[2].clone().numpy())   
                                        object_box_copy_3 = int(object_box[3].clone().numpy())   
                                   
                                        top_left = (int(object_box_copy_0), int(object_box_copy_1 - text_height * 1.2) )                   
                                        bottom_right = (int(object_box_copy_0 + text_width), int(object_box_copy_1))                                  
                                                                                                                                                   
                                        if object_name == "car" or object_name ==  "bus" or object_name == "truck" or object_name == "helmet":
                                            if color_name == "red" or color_name == "red_wrap":
                                                 color_name == "red"                 
                                                                                                                                                                                                             
                                            for n in range(len(bgr_classes)):                                                                                                                           
                                                if color_name == bgr_classes[n][0]:                                                                                                                                      
                                                    object_color = bgr_classes[n][1]                                                                                                     
                                        else:                                                                    
                                            object_color = object_colors[id_object]                
                                                                                                                                                    
                                        cv2.rectangle(frame, top_left, bottom_right, object_color, thickness=-1)                                                              
                                        cv2.rectangle(frame, (int(object_box_copy_0), int(object_box_copy_1)), (int(object_box_copy_2), int(object_box_copy_3)), object_color, 4)                                                                                                                                                                     
                                        org_0 = (int(object_box_copy_0), int(object_box_copy_1))                                                                                                          
                                                                                        
                                        if object_color[0] == 0 and object_color[1] == 0 and object_color[2] == 0:                                                                                                                                      
                                            cv2.putText(frame, text, org_0, font, fontScale_yolo, [255, 255, 255], thickness, cv2.LINE_AA)
                                        elif object_color[0] == 255 and object_color[1] == 0 and object_color[2] == 0:                                                                                                         
                                            cv2.putText(frame, text, org_0, font, fontScale_yolo, [255, 255, 255], thickness, cv2.LINE_AA)                        
                                        else:                                                                                                                                                                                                                                                                                                                                                                                                                                                        
                                            cv2.putText(frame, text, org_0, font, fontScale_yolo, [0, 0, 0], thickness, cv2.LINE_AA)                                              

                                                                                      
                            if object_name == "helmet":                                             
                                
                                helmet_crop = frame_copy[int(object_box[1]) : int(object_box[3]), int(object_box[0]) : int(object_box[2])]
                                color_name_detected = detect_color_family(helmet_crop) 
                                if detect_obj_color[0] in ["black", "white", "red", "lime", "blue", "yellow", "aqua", "magenta", "silver", "gray", "maroon", "olive", "green", "purple", "teal", "navy"]:
                                                                                   
                                    if detect_obj_color[0] == color_name_detected:
                                        color_name = color_name_detected                        
                                        text = color_name + " " + object_name +  " " + str(object_conf) 
                                                                                                                                                                                                                                                            
                                        (text_width, text_height), baseline = cv2.getTextSize(text, font, fontScale_yolo, thickness)  
                                                                                                                                                                                                                                                                                                                                                                                                                                      
                                        object_box_copy_0 = int(object_box[0].clone().numpy())
                                        object_box_copy_1 = int(object_box[1].clone().numpy())                                                                                              
                                        object_box_copy_2 = int(object_box[2].clone().numpy())   
                                        object_box_copy_3 = int(object_box[3].clone().numpy())   
                                   
                                        top_left = (int(object_box_copy_0), int(object_box_copy_1 - text_height * 1.2) )                   
                                        bottom_right = (int(object_box_copy_0 + text_width), int(object_box_copy_1))                                  
                                                                                                                                                   
                                        if object_name == "car" or object_name ==  "bus" or object_name == "truck" or object_name == "helmet":
                                            if color_name == "red" or color_name == "red_wrap":
                                                 color_name == "red"                 
                                                                                                                                                                                                             
                                            for n in range(len(bgr_classes)):                                                                                                                           
                                                if color_name == bgr_classes[n][0]:                                                                                                                                      
                                                    object_color = bgr_classes[n][1]                                                                                                     
                                        else:                                                                    
                                            object_color = object_colors[id_object]                
                                                                                                                                                    
                                        cv2.rectangle(frame, top_left, bottom_right, object_color, thickness=-1)                                                              
                                        cv2.rectangle(frame, (int(object_box_copy_0), int(object_box_copy_1)), (int(object_box_copy_2), int(object_box_copy_3)), object_color, 4)                                                                                                                                                                     
                                        org_0 = (int(object_box_copy_0), int(object_box_copy_1))                                                                                                          
                                                                                        
                                        if object_color[0] == 0 and object_color[1] == 0 and object_color[2] == 0:                                                                                                                                      
                                            cv2.putText(frame, text, org_0, font, fontScale_yolo, [255, 255, 255], thickness, cv2.LINE_AA)
                                        elif object_color[0] == 255 and object_color[1] == 0 and object_color[2] == 0:                                                                                                         
                                            cv2.putText(frame, text, org_0, font, fontScale_yolo, [255, 255, 255], thickness, cv2.LINE_AA)                        
                                        else:                                                                                                                                                                                                                                                                                                                                                                                                                                                        
                                            cv2.putText(frame, text, org_0, font, fontScale_yolo, [0, 0, 0], thickness, cv2.LINE_AA)      
                                             
                             
                      
                frame = cv2.resize(frame, up_points, interpolation= cv2.INTER_LINEAR)       
                                                                                                
                                                                                                                                                                                       
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
                                               

                if frame_num < 3:                                                                                      
                    pass                                                                                                                                                        
                else:                                                                                                          
                    average_inference_time = int(sum(yolo_inference_time) / (len(yolo_inference_time)))
                    average_fps = int(sum(yolo_fps_list) / (len(yolo_fps_list)))                                                           
                                                                                                                                                   
                    print("\n")                                                                    
                    print("per frame at shape (for model): "  + str([imgsz[0], imgsz[1], 3]) +  "    "  + "average_inference_time (for model): " + str(average_inference_time) + "  " + "average_fps (for display): " + str(average_fps) + "  " +  "device: " + device)
                    print("\n")     
                
                                                                                                                            
                frame_num += 1      

             
                if yolo_total > 10:                                                    
                    break                                                              
                                                                                                                                                          
def demo_mode_without_arg():
    # the 16 basic colors
    bgr_classes = [["black", (0,0,0)], ["white", (255,255,255)], ["red", (0,0,255)], ["lime", (0,255,0)], ["blue", (255,0,0)], ["yellow", (0,255,255)], ["aqua", (255,255,0)], ["magenta", (255,0,255)], ["silver", (192,192,192)], ["gray", (128,128,128)], ["maroon", (0,0,128)], ["olive", (0,128,128)], ["green", (0,128,0)], ["purple", (128,0,128)], ["teal", (128,128,0)], ["navy", (128,0,0)]]    
                                           
    coco_color = []                                                                                                                                            
    for i in range(81):
        b = random.randint(0, 255)
        g = random.randint(0, 255)
        r = random.randint(0, 255)
        bgr = [b,g,r]                                                        
        while (bgr in bgr_classes):
            b = random.randint(0, 255)
            g = random.randint(0, 255)
            r = random.randint(0, 255)           
            bgr = [b,g,r]
            
        coco_color.append(bgr)                                                    

                                                                                                                  
    with open('config.yml', 'r') as file:                                                                                                                                      
        config = yaml.safe_load(file)                                                                                                                                                              
                                                                                                                                                                 
    # video sources                                                                                                                       
    video_path_yolo = []                                                                                                                                                                                                          
    video_path_yolo.append(config["video_source"]["video_0"])
    video_path_yolo.append(config["video_source"]["video_1"])           
    video_path_yolo.append(config["video_source"]["video_2"])                                                                   
                                                              
    # model                            
    model = YOLOE(config["model"]["weight_path"])                                                                                                                                   
    
    # device                                                                                             
    device = "cpu"   
                                                                                                                                                                                                                      

    # title                                  
    font = cv2.FONT_HERSHEY_SIMPLEX                    
                     
                                                                                                                                               
    #  the features of YOLOE-26n object detection                                                                                                                                                              
    imgsz = config["yolo_features"]["imgsz"]                                                                                                                                           
    conf_thres = config["yolo_features"]["conf_thres"]                                                                                                                                 
    iou_thres = config["yolo_features"]["iou_thres"]
    fontScale_yolo_ori = config["yolo_features"]["fontScale_yolo_ori"]                                                            
    
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
    model.set_classes(all_category, model.get_text_pe(all_category))       
                                                                                                                                                                                                                           
    # the name of window                                                                                                                                                                                          
    window_name = config["window"]["window_name"]                                                                                         
                                                                                                                        
    # demo function                                                                                                                        
    while True:                                                                                                                                                                                                
        demo(model, video_path_yolo, imgsz, conf_thres, iou_thres,  all_category, coco_color, fontScale_yolo_ori, device, up_points, font, window_name, bgr_classes)



def demo_mode_with_arg(detect_obj_name, detect_obj_color):    
    
    # the 16 basic colors
    bgr_classes = [["black", (0,0,0)], ["white", (255,255,255)], ["red", (0,0,255)], ["lime", (0,255,0)], ["blue", (255,0,0)], ["yellow", (0,255,255)], ["aqua", (255,255,0)], ["magenta", (255,0,255)], ["silver", (192,192,192)], ["gray", (128,128,128)], ["maroon", (0,0,128)], ["olive", (0,128,128)], ["green", (0,128,0)], ["purple", (128,0,128)], ["teal", (128,128,0)], ["navy", (128,0,0)]]    
    
                                                                  
    object_colors = []                                                                                                                                            
    for i in range(len(detect_obj_name)):
        b = random.randint(0, 255)
        g = random.randint(0, 255)
        r = random.randint(0, 255)
        bgr = [b,g,r]                                                        
        while (bgr in bgr_classes):
            b = random.randint(0, 255)
            g = random.randint(0, 255)
            r = random.randint(0, 255)           
            bgr = [b,g,r]
            
        object_colors.append(bgr)

                                                                                                                  
    with open('config.yml', 'r') as file:                                                                                                                                      
        config = yaml.safe_load(file)                                                                                                                                                              
                                                                                                                                                                 
    # video sources                                                                                                                       
    video_path_yolo = []                                                                                                                                                                                                          
    video_path_yolo.append(config["video_source"]["video_0"])
    video_path_yolo.append(config["video_source"]["video_1"])           
    video_path_yolo.append(config["video_source"]["video_2"])                                                                   
                                                              
    # model                            
    model = YOLOE(config["model"]["weight_path"])                                                                                                                                   
    
    # device                                                                                             
    device = "cpu"   
                                                                                                                                                                                                                      
                                                               
    # title                                  
    font = cv2.FONT_HERSHEY_SIMPLEX                    
                     
                                                                                                                                               
    #  the features of YOLOE-26n object detection                                                                                                                                                              
    imgsz = config["yolo_features"]["imgsz"]                                                                                                                                           
    conf_thres = config["yolo_features"]["conf_thres"]                                                                                                                                 
    iou_thres = config["yolo_features"]["iou_thres"]
    fontScale_yolo_ori = config["yolo_features"]["fontScale_yolo_ori"]                                                            
    
    # display resolution: 1920 x 1080
    up_width = config["display_resolution"]["up_width"]                                                                                                                                           
    up_height = config["display_resolution"]["up_height"]                                                                                                                               
    up_points = (up_width, up_height)                          
                 
    # the class name of COCO                                                                                                                                                                                              
    coco_path = config["coco"]["coco_path"]                                                                                                     
    with open(coco_path, "r") as f:                                                                                                                                                                                                            
        coco_category = [line.strip() for line in f .readlines()]                

    model.set_classes(detect_obj_name, model.get_text_pe(detect_obj_name))      
                                                            
                                                                                                                                                                                                                
    # the name of window                                                                                                                                                                                          
    window_name = config["window"]["window_name"]                                                                                         
                                                                                                                        
    # demo function                                                                                                                        
    while True:                                                                                                                                                                                                
        demo_arg(model, video_path_yolo, imgsz, conf_thres, iou_thres,  detect_obj_name, detect_obj_color, coco_category, object_colors, fontScale_yolo_ori, device, up_points, font, window_name, bgr_classes)

                                                                                                                   
                                                                                                                                                                                                   
if __name__ == "__main__":            
    
    """                                                                                                                                                                        
    colors supported: ["black", "white", "red", "lime", "blue", "yellow", "aqua", "magenta", "silver", "gray", "maroon", "olive", "green", "purple", "teal", "navy"]
    colored object: helmet, truck, bus, car
    
    demo modes:
    demo with arguments: python3 demo.py yellow-helmet 
    demo without arguments: pyhton3 demo.py    
    """
                                                                            
    try:                                                             
        if len(sys.argv) > 1:                   
            detect_obj_name = []
            detect_obj_color = []                                                        

            obj = sys.argv[1].replace("-", " ")
            

            detect_obj_name .append(obj.split(" ")[1])
            detect_obj_color.append( obj.split(" ")[0])      

            demo_mode_with_arg(detect_obj_name, detect_obj_color)
            
        else:
            demo_mode_without_arg()
    except:
        print("please checking input format:")
        print("\n")
        print("python3 demo.py yellow-helmet ")
        print("or")
        print("pyhton3 demo.py  ")


                       