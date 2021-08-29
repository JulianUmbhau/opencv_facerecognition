# Easy OpenCV Face Recognition
Uses OpenCV facerecognition to train and predict.

2. Create folders images/person1, images/person2, etc. with training images
3. (optional) create images from webcam of faces, by running create_training_data.py
4. Save frontalcascade.xml from OpenCV module in cascades/data folder
4. Run faces_train.py to label, train and save trained model
5. Run faces.py to open webcam and make realtime face recognition based on trained model

Based on:
https://www.youtube.com/watch?v=PmZ29Vta7Vc

### Optional - Use face_recognition module for realtime prediction
Slower but much better performance
1. Correct picture location and labels in script face_recognition_package_stream.py
2. Run script for realtime webcam streaming with recognition 


## TODO
- Test different settings for training
- Create more diverse training images
- Different labels? eg. make every non-putin face correctly labeled and turn up conf limit on putin-images