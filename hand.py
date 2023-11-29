import cv2
import mediapipe as mp
import time

# open webcam
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

# using drawing_utils to draw
mpDraw = mp.solutions.drawing_utils

# set the color of the landmarks
handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=5)
# set the color of the connection line of the landmarks
handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=10)

currentTime = 0
endTime = 0

while True:
    ret, img = cap.read()
    if ret:
        # turn img into RGB
        imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # get the height and width of the img
        imgHeight = img.shape[0]
        imgWidth = img.shape[1]

        # through the " hands.process " of Mediapipe to analyze hands info
        result = hands.process(imgRGB)
        # print the landmarks of hands
        print(result.multi_hand_landmarks)

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                # draw landmarks only
                # mpDraw.draw_landmarks(img, handLms)
                
                # draw landmarks and connection lines
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)

                # print all landmarks of hands (x, y)
                for i, lm in enumerate(handLms.landmark):
                    # get actual coordinate
                    xPos = int(lm.x * imgWidth)
                    yPos = int(lm.y * imgHeight)
                    
                    # print on screen
                    cv2.putText(img, str(i), (xPos-25, yPos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255, 2))

                    # print the thumb
                    if i == 4:
                        cv2.circle(img, (xPos, yPos), 20, (255, 0, 0), cv2.FILLED)

                    # print the coordinates
                    print(i, xPos, yPos)
        
        currentTime = time.time()
        fps = 1/(currentTime-endTime)
        endTime = currentTime
        cv2.putText(img, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow('img', img)

    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break