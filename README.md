# Smart Waste Detection

This project contains `garbage_detection4.py`, a Python script for **real-time garbage detection** using **Faster R-CNN with ResNet backbone**.  
The system is designed for smart waste management applications and demonstrates detection with bounding boxes, confidence scores, and email alerts.

---

## Features
- Detects garbage and draws **bounding boxes** around detected objects.  
- Displays **confidence scores** for each detected object.  
- Sends **email alerts** with a screenshot of the detected garbage.  
- Accepts input in **three modes**:
  1. **Image** (single picture)  
  2. **Video** (pre-recorded)  
  3. **Live feed** (webcam)

---

## Input and Output Examples

### 1. Image Input
![Image Input with Bounding Boxes](image%20(7).png)  
*Bounding boxes drawn around garbage in a picture.*

### 2. Email Notification
![Email with Screenshot](image%20(8).png)  
*The system automatically sends an email with the screenshot of detected garbage.*

### 3. Video Input
![Video Frame Detection](image%20(9).png)  
*Bounding boxes drawn around garbage in a video frame.*

### 4. Live Feed
![Live Webcam Detection](image%20(10).png)  
*Real-time detection from webcam input, drawing bounding boxes around garbage held in front of the camera.*

---

