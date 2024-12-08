import threading
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

reference_img = cv2.imread("new.jpg")
face_match = False
lock = threading.Lock()

def check_face(frame):
    global face_match
    with lock:
        try:
            result = DeepFace.verify(frame, reference_img, enforce_detection=False)
            face_match = result['verified']  
        except Exception as e:
            print(f"An error occurred: {e}")
            face_match = False

counter = 0
while True:
    ret, frame = cap.read()
    if ret:
        if counter % 30 == 0:
            threading.Thread(target=check_face, args=(frame.copy(),)).start()
        counter += 1
        with lock:
            if face_match:
                cv2.putText(frame, "MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow("video", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cv2.destroyAllWindows()