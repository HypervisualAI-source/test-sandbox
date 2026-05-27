

""" 
################################  extract frame from video
import cv2                                                                                                                                                                                                     
import os                                                                                             
                                                                                                          
# Input video path
video_path = '/home/gpnpu/Desktop/AI_development/v0.0.rc6/demos/YOLOv13-S/source/output_video.mp4'
# Output folder for frames                        
output_dir = 'yolov13_s_frames'                      
                                                                                                                                                                                                        
# Create output directory if it doesn't exist                                         
os.makedirs(output_dir, exist_ok=True)                                

# Load the video                                                  
cap = cv2.VideoCapture(video_path)
                                                                           
# Frame counter                                   
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()                                                              
    if not ret:
        break                                                                                                             
                                                                                         
    # Save frame as image
    frame_filename = os.path.join(output_dir, f'frame_{frame_number:04d}.png')                                                                              
                                                                                   
    down_width = 640                                       
    down_height = 400
    down_points = (down_width, down_height)
    frame = cv2.resize(frame, down_points, interpolation= cv2.INTER_LINEAR)

    cv2.imwrite(frame_filename, frame)
    frame_number += 1                           
                                
cap.release()                                                                    
print(f"Extracted {frame_number} frames to '{output_dir}'")                                                                        
                    
                                                                                                                                           
"""         
"""                                                                                                   
#################################      image to .gif
from PIL import Image                                                                                                                                                                                                                                                                                                                                                                                                  
import glob
                                                 
# Get all images from the folder (adjust path and extension if needed)
image_files = sorted(glob.glob("yolov13_s_frames/*.png"))
print("image_files:", image_files)                                          
                                                                                                                                                
# Load all images                                                                                            
frames = [Image.open(image) for image in image_files]
                                                                                                                                                                                         
# Save as GIF                                                                                            
frames[0].save(                                                                                       
    'yolov13_s_frames.gif',                                                                       
    save_all=True,
    append_images=frames[1:],  # all but first image
    duration=30,              # duration of each frame in milliseconds
    loop=0                    # 0 = infinite loop, use any other number for count
)
                                                                                                                                                                                                                 
print("GIF saved as output.gif")           
"""

                                                                                                             

##### combine images 

import cv2                                                                                                 


img_n = cv2.imread('yolov13n.png')
img_s = cv2.imread('yolov13s.png')

target_size = (840, 680)  # (width, height)

img_n = cv2.resize(img_n, target_size)
img_s = cv2.resize(img_s, target_size)


# Horizontal stack
combined_n_s = cv2.hconcat([img_n, img_s])

cv2.imwrite("combined_n_s_840_680.jpg", combined_n_s)

                                                                                                                                                                 
                                                                                                                            
                                                                                                 
