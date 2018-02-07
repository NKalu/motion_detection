import cv2
import pandas
from datetime import datetime as dt


class Motion_Detector(object):
    #Uses cv2 to detect motion using webcam and writes when motion was on screen to .csv and .json files

    def __init__(self):
        self.initial_frame = None
        self.movement_list = [None, None]
        self.movement_times = []
        self.movement_data_frame = pandas.DataFrame(columns=["Start Time","End Time"])
        self.video_capture()

    def video_capture(self):
        video = cv2.VideoCapture(0)

        while True:
            movement_in_frame = 0
            check, frame = video.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

            if self.initial_frame is None:
                self.initial_frame = gray_frame
                continue

            delta_frame = cv2.absdiff(self.initial_frame, gray_frame)

            threshold_delta_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
            threshold_delta_frame = cv2.dilate(threshold_delta_frame, None, iterations=2)

            # finding external contours
            (_, cnts, _) = cv2.findContours(threshold_delta_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in cnts:
                # filters out objects larger than 10000 pixels and wraps in rectangle
                if cv2.contourArea(contour) < 10000:
                    continue
                movement_in_frame = 1
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            self.movement_list.append(movement_in_frame)

            if self.movement_list[-1] == 1 and self.movement_list[-2] == 0:
                self.movement_times.append(dt.now())
            if self.movement_list[-1] == 0  and self.movement_list[-2] == 1:
                self.movement_times.append(dt.now())

            cv2.imshow("Video", frame)
            key = cv2.waitKey(1)

            if key == ord('q'):
                if movement_in_frame == 1:
                    self.movement_times.append(dt.now())
                break

        video.release()
        cv2.destroyAllWindows()
        self.process_times()

    def process_times(self):
        for x in range(0, len(self.movement_times), 2):
            self.movement_data_frame = self.movement_data_frame.append({"Start Time":self.movement_times[x],
                                                    "End Time":self.movement_times[x+1]}, ignore_index=True)
        self.movement_data_frame.to_csv("Movement_Times.csv")
        self.movement_data_frame.to_json("Movement_Times.json")

    def plot_times(self):
        #TODO: implement plotting or some other visual display of data
        pass

if __name__ == '__main__':
    md = Motion_Detector()
    md.plot_times()