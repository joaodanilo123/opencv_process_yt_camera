# https://stackoverflow.com/questions/44752240/how-to-remove-shadow-from-scanned-images-using-opencv
# https://stackoverflow.com/questions/44047819/increase-image-brightness-without-overflow/44054699#44054699

import cv2
import youtube_dl
import time
import threading
import numpy as np


class Buffer:
    def __init__(self, max_size):
        self.buffer = []
        self.lock = threading.Lock()
        self.max_size = max_size

    def write_to_buffer(self, data):
        with self.lock:
            if len(self.buffer) < self.max_size:
                self.buffer.append(data)
            else:
                frame_buffer.buffer = []
                print("buffer cheio, reiniciando buffer")

    def read_from_buffer(self):
        with self.lock:
            if self.buffer:
                data = self.buffer[0]
                self.buffer = self.buffer[1:]
                return data
            else:
                print("Buffer is empty.")
                return None


def read_frames(buffer):
    livestream_url = "https://www.youtube.com/watch?v=5_XSYlAfJZM"
    ydl = youtube_dl.YoutubeDL({'quiet': True, 'nocheckcertificate': True})
    info = ydl.extract_info(livestream_url, download=False)
    video_url = info['url']
    cap = cv2.VideoCapture(video_url)

    counter = 1
    while len(buffer.buffer) < 1000:
        ret, frame = cap.read()
        buffer.write_to_buffer(frame)
        print(f"Saving frame {counter}")
        counter += 1

    while cap.isOpened():
        ret, frame = cap.read()
        buffer.write_to_buffer(frame)

    cap.release()


frame_buffer = Buffer(max_size=10000)

# Start a separate thread to read frames from the livestream
reader_thread = threading.Thread(target=read_frames, args=(frame_buffer,))
reader_thread.start()

FPS = 30
FRAMETIME = 1.0/FPS


bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    time.sleep(FRAMETIME)
    stored_frame = frame_buffer.read_from_buffer()

    if stored_frame is not None:

        stored_frame = cv2.cvtColor(stored_frame, cv2.COLOR_BGR2GRAY)

        ee = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        stored_frame = cv2.dilate(stored_frame, ee, iterations=3)

        stored_frame = cv2.GaussianBlur(stored_frame, (3, 3), 0)
        dilated_img = cv2.dilate(stored_frame, np.ones((7, 7), np.uint8))
        # bg_frame = cv2.medianBlur(dilated_img, 21)
        # stored_frame = 255 - cv2.absdiff(stored_frame, bg_frame)

        normalized = stored_frame.copy()  # Needed for 3.x compatibility
        cv2.normalize(stored_frame, normalized, alpha=0, beta=255,
                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        cv2.imshow("sem sombra", stored_frame)

        fg_mask = bg_subtractor.apply(stored_frame)
        limiar, fg_mask = cv2.threshold(
            fg_mask, None, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_mask = cv2.erode(fg_mask, ee, iterations=3)

        contours, h = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        str_c = f"a imagem possui {len(contours)} contornos"
        str_bf = f"Tamanho do Buffer: {len(frame_buffer.buffer)}",

        stored_frame = cv2.bitwise_and(
            stored_frame, stored_frame, mask=fg_mask)

        display_str = f"{str_c}\n{str_bf}"
        cv2.putText(stored_frame, display_str, (
            100, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Video Processado", stored_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

reader_thread.join()
cv2.destroyAllWindows()
