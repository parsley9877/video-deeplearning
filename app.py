import os
import wave

import numpy as np
from absl import app
import subprocess
import cv2
from scipy.io import wavfile

"""
Extracting Video Frames as numpy [T, H, W, C]:
1) Use kinetics downloader and download clips as mp4
2) Use opencv to read frames and put them in a numpy

Extracting Audio as numpy
ffmpeg is used for extracting audio as wav file
reading wav file using scipy.io

"""

def main(argv):
    path_to_data = './datasets/kinetics400'
    path_to_mp4 = 'datasets/kinetics400/video.mp4'

    # Extracting Frames
    del argv
    cap = cv2.VideoCapture(path_to_mp4)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    channels = 3
    video_numpy = np.zeros([num_frames, height, width, channels], dtype=np.uint8)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        video_numpy[i, :, :, :] = frame
        i += 1
    for i in range(video_numpy.shape[0]):
        cv2.imshow('video sample', video_numpy[i])
        cv2.waitKey(int((1/fps)*1000))
    cap.release()
    cv2.destroyAllWindows()

    # Extracting Audio
    subprocess.run(['sh', './datasets/kinetics400/audio_extractor.sh'])
    sample_rate, data = wavfile.read('./datasets/kinetics400/audio.wav')
    print(sample_rate)
    print(data.dtype)


if __name__ == '__main__':
    app.run(main)
