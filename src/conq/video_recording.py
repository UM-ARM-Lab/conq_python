#! /usr/bin/env python
from __future__ import print_function

import pathlib
import threading

import cv2

DEFAULT_CAMERA_NAME = '/dev/v4l/by-id/usb-AVerMedia_Technologies__Inc._Live_Gamer_Portable_2_Plus_1311774402370-video-index0'


class VideoRecorder:
    def __init__(self, device_num, path_prefix='video/'):
        self.device_num = device_num
        self.path_prefix = path_prefix
        self.cap = cv2.VideoCapture(device_num)
        self.thread = None
        self.exit = None
        self.writer = None

    def stop_current_recording(self):
        if self.writer is not None:
            self.writer.release()
            print("Stopping current recording")
        self.is_recording = False

    def start_new_recording(self, filename, fps=30):
        if filename[-4:] == ".mp4":
            fourcc_code = cv2.VideoWriter_fourcc(*"mp4v")  # Previously used hardcoded fourcc_code = 0x00000021
        elif filename[-4:] == ".avi":
            fourcc_code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        else:
            print("Invalid file type " + filename[-4:])
            return False

        frame_dims = (int(self.cap.get(3)), int(self.cap.get(4)))
        print(f"{frame_dims=}")

        print(f"{fps=}")

        path = pathlib.Path(self.path_prefix) / filename
        path.parent.mkdir(exist_ok=True, parents=True)
        print(f"Starting recording for {str(path)}")

        self.is_recording = True

        self.writer = cv2.VideoWriter(str(path), fourcc_code, fps, frame_dims)
        return True

    def start_in_thread(self):
        def _target():
            self.exit = False

            while not self.exit:
                ret, frame = self.cap.read()
                self.writer.write(frame)

        self.thread = threading.Thread(target=_target)
        self.thread.start()

    def stop_in_thread(self):
        self.stop_current_recording()
        self.exit = True
        self.thread.join()
        self.cap.release()


def live_view(cap):
    print("Frame dims: ", cap.get(3), ", ", cap.get(4))
    print("FourCC code: ", cap.get(cv2.CAP_PROP_FOURCC))
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()
