#!/bin/bash
echo "Running Demo Inference..."                                  

# Activate virtual environment                                                                           
source ../../venv/bin/activate                                  
                                  
# Run the script                                                                                                                                                        
python3 demo.py                      
                                                    
echo "Shutting Down..."

# Deactivate environment (optional)
deactivate               
                                                                                                       