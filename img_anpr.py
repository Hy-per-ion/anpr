import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import easyocr

img = cv2.imread("image4.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

mask = np.zeros(gray.shape, np.uint8)
new_img = cv2.drawContours(mask, [location], 0, 255, -1)
new_img = cv2.bitwise_and(img, img, mask=mask)

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_img = gray[x1:x2 + 1, y1:y2 + 1]

reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_img)
text = result[0][-2]

font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(location[0][0][0], location[1][0][1] + 30), fontFace=font,fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 3)

#plt.figure(figsize=(12, 6))

# plt.subplot(2, 2, 1)
# plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
# plt.title('Grayscale Image')

# plt.subplot(2, 2, 2)
# plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
# plt.title('Edged Image')

# plt.subplot(2, 2, 3)
# plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
# plt.title('Contour Mask')

#plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.title('Text Detection Result')

plt.show()
