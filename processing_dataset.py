
import os
import cv2

train_annotation = "/home/gpnpu/Desktop/Drone_Indonisia/drone_object_detection/VisDrone2019-DET-train/annotations/"
val_annotation = "/home/gpnpu/Desktop/Drone_Indonisia/drone_object_detection/VisDrone2019-DET-val/annotations/"
test_annotation = "/home/gpnpu/Desktop/Drone_Indonisia/drone_object_detection/VisDrone2019-DET-test-dev/annotations/"


train_image = "/home/gpnpu/Desktop/Drone_Indonisia/drone_object_detection/VisDrone2019-DET-train/images/"
val_image = "/home/gpnpu/Desktop/Drone_Indonisia/drone_object_detection/VisDrone2019-DET-val/images/"
test_image = "/home/gpnpu/Desktop/Drone_Indonisia/drone_object_detection/VisDrone2019-DET-test-dev/images/"
                                                                                                           
                                                        
# Training                                                                                                                                                                                       
train_images_0 = os.listdir(train_image)                                                                                        
train_images = []

print("len(train_images_0):", len(train_images_0))
                                                                                                                                                                                          
train_annotations = []
                                                                        
for i in range(len(train_images_0)):
    name = train_images_0[i].split(".")[0]
    #print("name:", name)
    train_images.append(train_image + train_images_0[i])
    train_annotations.append(train_annotation + name + ".txt")
              
                                            
for i in range(len(train_images)):
    
    
    img = cv2.imread(train_images[i])
    h, w = img.shape[:2]
    
    labels = []
    with open(train_annotations[i], "r") as f:
        for line in f:              
            label = []
    
            # clean commas if present                                                           
            line = line.replace(",", " ")       
            #print("line_____________:", line)                  
                
            parts = line.strip().split()
            #print("parts____________:", parts)
            
            
            cls = int(parts[5])
            
            if cls == 0 or cls == 11:
                continue
            
            if cls == 1:
                cls = 0
            if cls == 2:
                cls = 1
            if cls == 3:
                cls = 2
            if cls == 4:
                cls = 3
            if cls == 5:
                cls = 4
            if cls == 6:
                cls = 5
            if cls == 7:
                cls = 6
            if cls == 8:
                cls = 7  
            if cls == 9:
                cls = 8
            if cls == 10:
                cls = 9                  
            
            bbox_left = int(parts[0])
            bbox_top = int(parts[1])
            bbox_width = int(parts[2])
            bbox_height = int(parts[3])    
            
            class_id = cls
            x_center = (((bbox_width / 2) + bbox_left )/ w )
            y_center = (((bbox_height / 2) + bbox_top)/ h)
            width = bbox_width / w
            height = bbox_height / h                                                                                                                                          
                                                                                                   
            label.append(class_id)                                                                                                                                                     
            label.append(x_center)
            label.append(y_center)
            label.append(width)
            label.append(height)
            
            labels.append(label)
            
    file_name = train_annotations[i].split("/")[-1]
    #print("file_name:", file_name)
    

    with open("/home/gpnpu/Desktop/Drone_Indonisia/drone_object_detection/yolo_format/train/" + file_name, "w") as f:
            for label in labels:
                line = " ".join(map(str, label))
                f.write(line + "\n")                                                                                                                 
       

                 
# val                                     
val_images_0 = os.listdir(val_image)
val_images = []


print("len(val_images_0):", len(val_images_0))

val_annotations = []
                                                                        
for i in range(len(val_images_0)):
    name = val_images_0[i].split(".")[0]
    #print("name:", name)
    val_images.append(val_image + val_images_0[i])
    val_annotations.append(val_annotation + name + ".txt")
              
                                            
for i in range(len(val_images)):
    
    
    img = cv2.imread(val_images[i])
    h, w = img.shape[:2]
    
    labels = []
    with open(val_annotations[i], "r") as f:
        for line in f:              
            label = []
    
            # clean commas if present                                                           
            line = line.replace(",", " ")       
            #print("line_____________:", line)                  
                
            parts = line.strip().split()
            #print("parts____________:", parts)
            
            
            cls = int(parts[5])

            if cls == 0 or cls == 11:
                continue
            
            if cls == 1:
                cls = 0
            if cls == 2:
                cls = 1
            if cls == 3:
                cls = 2
            if cls == 4:
                cls = 3
            if cls == 5:
                cls = 4
            if cls == 6:
                cls = 5
            if cls == 7:
                cls = 6
            if cls == 8:
                cls = 7  
            if cls == 9:
                cls = 8
            if cls == 10:
                cls = 9              
            
            
            
            
            bbox_left = int(parts[0])
            bbox_top = int(parts[1])
            bbox_width = int(parts[2])                                                                                                            
            bbox_height = int(parts[3])    
            
            class_id = cls
            x_center = (((bbox_width / 2) + bbox_left )/ w )
            y_center = (((bbox_height / 2) + bbox_top)/ h)
            width = bbox_width / w
            height = bbox_height / h
            
            label.append(class_id)
            label.append(x_center)
            label.append(y_center)
            label.append(width)
            label.append(height)
            
            labels.append(label)
            
    file_name = val_annotations[i].split("/")[-1]
    #print("file_name:", file_name)
                                                                                                                     

    with open("/home/gpnpu/Desktop/Drone_Indonisia/drone_object_detection/yolo_format/val/" + file_name, "w") as f:
            for label in labels:
                line = " ".join(map(str, label))
                f.write(line + "\n")              
 
                                                   
# test                                     
test_images_0 = os.listdir(test_image)
test_images = []

print("len(test_images_0):", len(test_images_0))

test_annotations = []
                                                                        
for i in range(len(test_images_0)):
    name = test_images_0[i].split(".")[0]
    #print("name:", name)
    test_images.append(test_image + test_images_0[i])
    test_annotations.append(test_annotation + name + ".txt")
              
                                            
for i in range(len(test_images)):
    
    
    img = cv2.imread(test_images[i])
    h, w = img.shape[:2]
    
    labels = []
    with open(test_annotations[i], "r") as f:
        for line in f:              
            label = []
    
            # clean commas if present                                                           
            line = line.replace(",", " ")       
            #print("line_____________:", line)                  
                
            parts = line.strip().split()
            #print("parts____________:", parts)
            
            
            cls = int(parts[5])
            
            if cls == 0 or cls == 11:
                continue
            
            if cls == 1:
                cls = 0
            if cls == 2:
                cls = 1
            if cls == 3:
                cls = 2
            if cls == 4:
                cls = 3
            if cls == 5:
                cls = 4
            if cls == 6:
                cls = 5
            if cls == 7:
                cls = 6
            if cls == 8:
                cls = 7  
            if cls == 9:
                cls = 8
            if cls == 10:
                cls = 9   
            
            
            bbox_left = int(parts[0])
            bbox_top = int(parts[1])
            bbox_width = int(parts[2])
            bbox_height = int(parts[3])    
            
            class_id = cls
            x_center = (((bbox_width / 2) + bbox_left )/ w )
            y_center = (((bbox_height / 2) + bbox_top)/ h)
            width = bbox_width / w
            height = bbox_height / h
            
            label.append(class_id)
            label.append(x_center)
            label.append(y_center)
            label.append(width)
            label.append(height)
            
            labels.append(label)
            
    file_name = test_annotations[i].split("/")[-1]
    #print("file_name:", file_name)
    

    with open("/home/gpnpu/Desktop/Drone_Indonisia/drone_object_detection/yolo_format/test/" + file_name, "w") as f:
            for label in labels:
                line = " ".join(map(str, label))                                                                
                f.write(line + "\n")                                           

                       

