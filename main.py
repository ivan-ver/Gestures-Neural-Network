from tensorflow import keras
import cv2
import numpy as np

if __name__ == '__main__':
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out_stream = cv2.VideoWriter('output.avi', fourcc, 30, (640, 480))
    model = keras.models.load_model('gestures_model.h5')
    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()
        img2 = cv2.resize(img, None, fx=0.3, fy=0.3)
        result = model.predict(np.array([img2]))
        result = list(result[0]).index(max(list(result[0])))
        text = ""
        if result == 0:
            text = "CLASS"
        elif result == 1:
            text = "OK"
        elif result == 2:
            text = "BAD"
        else:
            text = ""

        cv2.rectangle(img, (0, 0), (230, 60), (0, 255, 0), - 1)
        cv2.putText(img, f"Result: {text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow("camera", img)
        out_stream.write(img)

        if cv2.waitKey(10) == 27:
            break

    cam.release()
    out_stream.release()
    cv2.destroyAllWindows()
