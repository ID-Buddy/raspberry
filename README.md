# raspberry
안면실인증을 위한 서비스: 라즈베리파이 서버



## Operating System

- Raspberry Pi OS (64-bit), latest version recommended.

## Python Environment

- **Python Version:** 3.9 or higher
- **Virtual Environment:** Python venv or virtualenv is recommended.

## Required Libraries

flask==2.1.3
flask-socketio==5.2.0
picamera2
opencv-python-headless==4.5.3.56
numpy==1.21.2
face_recognition==1.3.0
dlib==19.22.99
mtcnn==0.1.0
eventlet==0.31.0
gevent==21.8.0




Learn how to install Dlib and Face_Recognition on your Raspberry Pi for efficient face detection and recognition

## Installing Dlib on Raspberry Pi
To install Dlib on your Raspberry Pi, follow these detailed steps to ensure a smooth setup. This guide assumes you have a Raspberry Pi running a compatible version of Raspbian OS.

## Step 1: Update Your System
Before starting the installation, it's crucial to update your system packages. Open a terminal and run the following commands:
```bash
sudo apt update
sudo apt upgrade
```
## Step 2: Install Required Dependencies
Dlib requires several libraries to function correctly. Install the necessary dependencies with the following command:
```bash
sudo apt install build-essential cmake python3-dev python3-pip libatlas-base-dev
```
## Step 3: Install Dlib
Now, you can install Dlib using pip. It's recommended to use a virtual environment to avoid conflicts with other packages. First, install the virtual environment package:
```bash
sudo pip3 install virtualenv
```
Create a new virtual environment:
```bash
virtualenv dlib_env
```
Activate the virtual environment:
```bash
source dlib_env/bin/activate
```
Now, install Dlib:
```bash
pip install dlib
```
## Step 4: Install face_recognition
To use Dlib for face recognition, you can install the face_recognition library, which simplifies the process. Run the following command in your activated virtual environment:
```bash
pip install face_recognition
```
## Step 5: Verify the Installation
To ensure that Dlib and face_recognition are installed correctly, you can run a simple Python script. Create a new Python file and add the following code:
```bash
import dlib
import face_recognition

print("Dlib and face_recognition installed successfully!")
```
Run the script:
```bash
python your_script_name.py
```
If you see the success message, your installation is complete!

## Additional Resources
For more detailed information, you can refer to the official Dlib documentation at Dlib Documentation.

By following these steps, you should have Dlib and face_recognition installed on your Raspberry Pi, ready for your projects.

#### Related answers

*Open-source Face Recognition System Install*
Learn how to install the Open-source Face Recognition System in Python with step-by-step instructions and essential tips.

*Dlib Face Recognition GitHub*
Explore the dlib face recognition library on GitHub, featuring open-source code for advanced facial recognition systems.

*Dlib Face Recognition Download*
Download the Open-source Face Recognition System Code using dlib for efficient and accurate face detection and recognition.
