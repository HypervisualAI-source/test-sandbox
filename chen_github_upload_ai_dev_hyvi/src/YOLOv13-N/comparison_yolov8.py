                   
import subprocess                                          
import re                                                                                          
import json                                                                                           
                                                                                
# Call script in venv_benchmark                                                                                                    
features_benchmark = subprocess.run(["/home/gpnpu/Desktop/AI_development/v0.0.rc4/venv_benchmark/bin/python", "benchmark_yolov8.py"], capture_output=True, text=True)
                                                                                                                                                                                                              
# Call script in venv                                                                                  
features_v13 = subprocess.run(["/home/gpnpu/Desktop/AI_development/v0.0.rc4/venv/bin/python", "YOLOv13_N.py"], capture_output=True, text=True)
                                                                                
match_benchmark = re.search(r'features:\s*(.+)', features_benchmark.stdout)
match = re.search(r'features:\s*(.+)', features_v13.stdout)                                                                           
                                                                                                                                                                     
data_v13 = match.group(1)                                                                                                                          
data_benchmark = match_benchmark.group(1)  
                                                                    
                                                                                       
# title
title = ["Model", "parameters(M)", "GFLOPs", "latency(ms)", "mAP50_95"]
                                                                                                                                                   
# features from YOlOv13
yolov13n = []                                                                                                                                         
text = data_v13                                       
result = text.split(' ')                          
for i in range(len(result)):
    if not result[i]:
        pass
    else:
        yolov13n.append(result[i])
                                                    
                                                                                                                                                         
# features from benchmarks                                                                                                       
benchmark = []  
text = data_benchmark
result = text.split(' ')                                                       

for i in range(len(result)):
    if not result[i]:
        pass
    else:
        benchmark.append(result[i])

benchmark_list = []
benchmark_list.append(benchmark[:5]) 
benchmark_list.append(benchmark[5:10]) 
benchmark_list.append(benchmark[10:15])       
benchmark_list.append(benchmark[15:20]) 
benchmark_list.append(benchmark[20:25])    
                                                                       
                                                                                                                            
data = []                                                                             
data.append(title)                             
                                                                                      
for i in range(len(benchmark_list)):
    data.append(benchmark_list[i])
                                                                                                                 
data.append(yolov13n)                                            
                                                                                                  
# comparison table                                               
for row in data:                                                                  
    print(row)  
