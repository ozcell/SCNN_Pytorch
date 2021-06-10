import cv2

cap = cv2.VideoCapture('/home/ok21/dataset/forddata/V1.mp4')
assert cap.isOpened(), "Video file cannot be opened"

while cap.isOpened():
    ret, img = cap.read()
    if ret:
        cv2.imshow("da window", img)
        cv2.waitKey(1)
    else:
        cap.release()
print("done")
    
cv2.destroyAllWindows()