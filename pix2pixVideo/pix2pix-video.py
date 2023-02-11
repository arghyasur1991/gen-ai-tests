import os
import cv2
import numpy as np
from moviepy.editor import *
# from share_btn import community_icon_html, loading_icon_html, share_js

from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
from PIL import Image
import time
import psutil
import random

pipe = DiffusionPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.unet.to(memory_format=torch.channels_last)

device = "GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"
frame_size = 1024

output_dir = "G:\\My Drive\\My Private\\Projects\\ML\\test_videos\\pix2PixResults\\"
output_dir_int = output_dir + "intermediates\\"

prompt = "Make it a dragon"
file_name = "girgit_2_st"

trim_in = 4
video_inp = "G:\\My Drive\\My Private\\Projects\\ML\\test_videos\\" + file_name + ".mp4"
# seed_inp = gr.Slider(label="Seed", minimum=0, maximum=2147483647, step=1, value=123456)
seed_inp = 123456
generate_intermediates = True

if torch.cuda.is_available():
    pipe = pipe.to("cuda")


def pix2pix(
        prompt,
        text_guidance_scale,
        image_guidance_scale,
        image,
        steps,
        neg_prompt="",
        width=512,
        height=512,
        seed=0,
):
    print(psutil.virtual_memory())  # print memory usage

    if seed == 0:
        seed = random.randint(0, 2147483647)

    generator = torch.Generator("cuda").manual_seed(seed)

    try:
        image = Image.open(image)
        ratio = min(height / image.height, width / image.width)
        image = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.LANCZOS)

        result = pipe(
            prompt,
            negative_prompt=neg_prompt,
            image=image,
            num_inference_steps=int(steps),
            image_guidance_scale=image_guidance_scale,
            guidance_scale=text_guidance_scale,
            generator=generator,
        )

        # return replace_nsfw_images(result)
        return result.images, result.nsfw_content_detected, seed
    except Exception as e:
        return None, None, error_str(e)


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

    # check fps
    if False: #clip.fps > 30:
        print("vide rate is over 30, resetting to 30")
        clip_resized = clip.resize(height=frame_size)
        clip_resized.write_videofile(output_dir_int + "video_resized.mp4", fps=30)
    elif generate_intermediates:
        print("video rate is OK")
        clip_resized = clip.resize(height=frame_size)
        clip_resized.write_videofile(output_dir_int + "video_resized.mp4", fps=clip.fps)

    print("video resized to 512 height")

    # Opens the Video file with CV2
    cap = cv2.VideoCapture(output_dir_int + "video_resized.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("video fps: " + str(fps))
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if generate_intermediates:
            cv2.imwrite(output_dir_int + 'kang' + str(i) + '.jpg', frame)
        frames.append(output_dir_int + 'kang' + str(i) + '.jpg')
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


def infer(prompt, video_in, seed_in, trim_value):
    print(prompt)
    break_vid = get_frames(video_in)

    frames_list = break_vid[0]
    fps = break_vid[1]
    n_frame = int(trim_value * fps)

    if n_frame >= len(frames_list):
        print("video is shorter than the cut value")
        n_frame = len(frames_list)

    result_frames = []
    print("set stop frames to: " + str(n_frame))

    for i in frames_list[0:int(n_frame)]:
        pix2pix_img = pix2pix(prompt, 7.5, 1.5, i, 30, "", frame_size, frame_size, seed_in)
        images = pix2pix_img[0]
        rgb_im = images[0].convert("RGB")

        # exporting the image
        frame_file = i.split("\\")
        frame_file = frame_file[len(frame_file)-1]
        print(frame_file)
        rgb_im.save(f"result_img-{frame_file}")
        result_frames.append(f"result_img-{frame_file}")
        print("frame " + i + "/" + str(n_frame) + ": done;")

    final_vid = create_video(result_frames, fps)
    print("finished !")

    return final_vid


infer(prompt, video_inp, seed_inp, trim_in)
