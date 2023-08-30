import cv2
import numpy as np
import imutils
import easyocr

# Load the video capture object
cap = cv2.VideoCapture('video4.mp4')  # Replace 'video.mp4' with your video file

reader = easyocr.Reader(['en'])

detected_texts = []  # Store detected texts

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize the frame to a smaller size
    resized_frame = imutils.resize(frame, width=800)  # Adjust the width as needed
    
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is not None:
        mask = np.zeros(gray.shape, np.uint8)
        new_img = cv2.drawContours(mask, [location], 0, 255, -1)
        new_img = cv2.bitwise_and(resized_frame, resized_frame, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_img = gray[x1:x2 + 1, y1:y2 + 1]

        result = reader.readtext(cropped_img)
        if result:
            text = result[0][-2]
            detected_texts.append(text)  # Store the detected text
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(resized_frame, text=text, org=(location[0][0][0], location[1][0][1] + 60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            cv2.rectangle(resized_frame, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

    cv2.imshow('Number Plate Detection', resized_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    for i, text in enumerate(detected_texts):
        print(f"Frame {i+1}: {text}")

cap.release()
cv2.destroyAllWindows()
