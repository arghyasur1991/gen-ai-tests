import os
import cv2
import numpy as np
from moviepy.editor import *
frame_size = 512

output_dir = "/Users/sur/Downloads/test/"


def error_str(error, title="Error"):
    return (
        f"""#### {title}
            {error}"""
        if error
        else ""
    )


def get_frames(video_in):
    frames = []
    # resize the video
    clip = VideoFileClip(video_in)

    print("video rate is OK")
    clip_resized = clip.resize(height=frame_size)
    clip_resized.write_videofile(output_dir + "video_resized.mp4", fps=clip.fps)

    # Opens the Video file with CV2
    cap = cv2.VideoCapture(output_dir + "video_resized.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("video fps: " + str(fps))
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        fi_name = output_dir + 'kang' + str(i) + '.jpg'
        if True:
            cv2.imwrite(fi_name, frame)
        frames.append(fi_name)
        i += 1

    cap.release()
    cv2.destroyAllWindows()
    print("broke the video into frames")

    return frames, fps


def create_video(frames, fps):
    print("building video result")
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_dir + file_name + "_processed.mp4", fps=fps)

    return 'movie.mp4'


get_frames("/var/folders/3x/sl713qb96wzgh323dx4qx6tr0000gn/T/testwzd31c6s.mp4")
