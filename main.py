import cv2
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression
st.urlretrieve(url, filename)

def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

def main():
    # Load the pre-trained EAST text detector
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    # Start the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        orig = frame.copy()

        (height, width) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (width, height), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"])
        (rects, confidences) = decode_predictions(scores, geometry)

        boxes = non_max_suppression(np.array(rects), probs=confidences)

        for (startX, startY, endX, endY) in boxes:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            roi = orig[startY:endY, startX:endX]
            text = pytesseract.image_to_string
            # Run Tesseract OCR on the region of interest to extract the text
            text = pytesseract.image_to_string(roi)

            # Check if the extracted text is a chain of six numbers
            if text.isdigit() and len(text) == 6:
                print("Found chain of six numbers:", text)

        # Display the processed frame
        cv2.imshow("Text Detection", frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
