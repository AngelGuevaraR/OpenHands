import cv2
import mediapipe as mp

dispositivocam = cv2.VideoCapture(0)

mpManos = mp.solutions.hands

manos = mpManos.Hands(static_image_mode = False, 
                      max_num_hands = 2, 
                      min_detection_confidence = 0.9, 
                      min_tracking_confidence = 0.8)

mpMostrar = mp.solutions.drawing_utils

while True:
    success,img = dispositivocam.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    resultadocapturado = manos.process(imgRGB)
    if resultadocapturado.multi_hand_landmarks:
        for handLms in resultadocapturado.multi_hand_landmarks:
            mpMostrar.draw_landmarks(img, handLms, mpManos.HAND_CONNECTIONS)

    cv2.imshow("image", img)
    cv2.waitKey(1) #milisegundos

