# app.py

"""
import streamlit as st
                        
# check 0: file_uploader, selectbox, differen pages

st.title("🌟 My First Streamlit App")
st.write("Hello! This is a simple web app built with Streamlit.")

name = st.text_input("What's your name?")
if name:
    st.success(f"Nice to meet you, {name}!")

if st.button("Click me"):
    st.balloons()


st.slider("Pick a number", 0, 100)                                                                
st.selectbox("Choose a language", ["Python", "C++", "JavaScript"])
st.file_uploader("Upload a CSV file")
"""
                                                                                                 
"""
# check 1: Pages by Clicking a Button                                 

import streamlit as st

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = 1

# Function to go to next page
def next_page():
    st.session_state.page += 1

# Function to go to previous page (optional)
def prev_page():
    st.session_state.page -= 1

# Display page based on state
if st.session_state.page == 1:
    st.title("📄 Page 1")
    st.write("Welcome to page 1.")
    
    if st.button("Next"):
        next_page()
                                                                                                                                          
elif st.session_state.page == 2:
    st.title("📄 Page 2")                   
    st.write("Now you're on page 2.")
    st.file_uploader("eeeeeejkjkjkUpload a CSV file")
    col1, col2 = st.columns(2)
    if col1.button("Back"):
        prev_page()
    if col2.button("Next"):
        next_page()

elif st.session_state.page == 3:
    st.title("📄 Page 3")
    st.write("This is the last page.")
    st.file_uploader("jkjkjkUpload a CSV file")
    st.selectbox("Choose a language", ["ytytPython", "C++", "JavaScript"])
    if st.button("Back"):
        prev_page()

"""                                                                                    


"""
# check 2: display videos               
                
import streamlit as st

st.title("🎥 Video Player Example")                                                                     
                       
# Load and display a local video file
video_file = open('/home/gpnpu/Desktop/virtual_environment_gpnpu/pytorch_dlprim_cpu_gpu/pytorch_cpu/source/jellyfish.mp4', 'rb')
video_bytes = video_file.read()

print("video_bytes:", video_bytes, type(video_bytes))

st.video(video_bytes)
"""                                                                                                                     


"""     
# check 3: display image                
                                                                                       
# 3-1: Image 
import streamlit as st
from PIL import Image                   
                             
image = Image.open("/home/gpnpu/Desktop/virtual_environment_gpnpu/pytorch_dlprim_cpu_gpu/pytorch_cpu/224_yolo.jpg")
print("000000000000000000000000000")
print("Image:", Image, type(image))
st.image(image, caption="This is an image", use_column_width=True)
print("11111111111111111111111")

                     

                                                                  
# 3-2: numpy + opencv                                                                                                                                                                                                                               
import cv2
                                                                        

import streamlit as st
import time
import numpy as np
                                                                                             
st.title("📷 Continuous Image Display")

# Create a placeholder (this is where images will be updated)
image_slot = st.empty()

# Simulate a list of images (you can load real images here)
cout = 0                                                                             
                                                                                                                                       
                                                                                                                            
for i in range(500):                                                                                                                                           
    cout += 1
    # Create a dummy image (random colors)                                                                      
    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    # Display image in the placeholder
    image_slot.image(img, caption=f"Frame {i+1}", use_container_width=True)
    
    # Pause before showing the next one
    time.sleep(0.01)
    print("cout:", cout)              
                         
"""

"""
# check 4: usage of cpu

import psutil
import os
import time

pid = os.getpid()  # Current process ID
proc = psutil.Process(pid)

for i in range(500):
    print(f"CPU usage of this process: {proc.cpu_percent(interval=1)}%")
"""

# check 5: layout of page         
        
import streamlit as st

"""
# 5-1: side by side    
                                                                                                          
col1, col2 = st.columns(2)                       

with col1:
    st.header("Left Column")
    st.button("Click me")           

with col2:
    st.header("Right Column")
    st.slider("Slide me")

                                                                         

# 5-2: separate config, filter, or navigation from main page
st.sidebar.title("Sidebar")
option = st.sidebar.selectbox("Choose a value", [1, 2, 3])

st.write("Main content area shows:", option)


# 5-3: group related elements together (update later)

container = st.container()

with container:
    st.write("Grouped content")
    st.button("Do something")
    
    
# 5-4: hide/show sections on click
                                                       
with st.expander("See more details"):
    st.write("Here’s some extra info...")


# 5-5: control overal  layout, wide or centered
st.set_page_config(
    page_title="My App",
    layout="wide",  # Use "wide" or "centered"
)
"""
"""
# 5-6: Three sections with custom widths: 1:2:1

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.write("📋 Sidebar Area")

with col2:
    st.write("📊 Main Chart Area")

with col3:
    st.write("🛠️ Settings or Summary")
"""
                                 
                                                                                                                                
"""                                  
# 5-7: custom vertical sections(rows)  with inside layout

import streamlit as st

st.set_page_config(layout="wide")

# Apply global CSS styling
### important: adding """ """ behind  st.markdown( and in fornt of , unsafe_allow_html=True)  

st.markdown(
<style>
.row-box {
    padding: 20px;
    margin-bottom: 10px;
    border-radius: 10px;
    border: 1px solid #ddd;
}

/* Set heights using vh (viewport height) to mimic 1:2:1 ratio */
.row-1 { height: 25vh; background-color: #f9f9f9; }
.row-2 { height: 50vh; background-color: #e0f7fa; }
.row-3 { height: 25vh; background-color: #fce4ec; }
</style>
, unsafe_allow_html=True)


# Now, use st.container() to hold real content
with st.container():
    st.markdown('<div class="row-box row-1">', unsafe_allow_html=True)
    st.subheader("🔝 Top Section")
    st.slider("Top slider", 0, 100)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="row-box row-2">', unsafe_allow_html=True)
    st.subheader("📊 Middle Section")
    st.line_chart([1, 3, 2, 4])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="row-box row-3">', unsafe_allow_html=True)
    st.subheader("🔻 Bottom Section")
    st.text_input("Bottom input")
    st.markdown("</div>", unsafe_allow_html=True)                                                                        
"""                                                                                       


"""                                                               
# 5-8: layout example -> full dashboard page
import streamlit as st

st.set_page_config(page_title="Dashboard", layout="wide")

st.title("📊 My Dashboard")

# Sidebar
st.sidebar.header("Controls")
date = st.sidebar.date_input("Select a date")
option = st.sidebar.selectbox("Choose metric", ["Revenue", "Users", "Errors"])

# Two columns in main area
col1, col2 = st.columns(2)                         
                                                                                                                           
with col1:
    st.metric("Total Revenue", "$12,000")
    st.line_chart([1, 2, 3, 4])

with col2:
    st.metric("Active Users", "340")
    st.bar_chart([4, 2, 5, 3])

           

st.title("🎥 Video Player Example")                                                                     
                       
# Load and display a local video file
video_file = open('/home/gpnpu/Desktop/virtual_environment_gpnpu/pytorch_dlprim_cpu_gpu/pytorch_cpu/source/jellyfish.mp4', 'rb')
video_bytes = video_file.read()

print("video_bytes:", video_bytes, type(video_bytes))

#st.video(video_bytes)                                                             


col1, col2 = st.columns(2)                       

with col1:
    st.header("Left Column")                                                                                                                                                                
    st.button("Click me")           

with col2:
    st.header("Right Column")
    st.video(video_bytes)        
    
"""         


                                                                                                                                                                                  

