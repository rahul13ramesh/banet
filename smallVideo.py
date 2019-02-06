# coding: utf-8
''' Load image features and extract with VGG '''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import cv2
import numpy as np
import skimage
import pickle
from args import video_root
from args import max_frames, smallvid_path


def sample_frames(video_path, train=True):
    '''
    对视频帧进行采样，减少计算量。等间隔地取max_frames帧
    '''
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        # 把BGR的图片转换成RGB的图片，因为之后的模型用的是RGB格式
        if ret is False:
            break
        frame = frame[:, :, ::-1]
        frames.append(frame)
        frame_count += 1

    indices = np.linspace(8, frame_count - 7, max_frames, endpoint=False, dtype=int)

    frames = np.array(frames)
    frame_list = frames[indices]
    clip_list = []
    for index in indices:
        clip_list.append(frames[index - 8: index + 8])
    clip_list = np.array(clip_list)
    return frame_list, clip_list, frame_count


def blackWhite_resize_frame(image, target_height=56, target_width=56):
    assert(len(image.shape) == 3)

    height, width, channels = image.shape
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if height == width:
        resized_image = cv2.resize(img_gray, (target_height, target_width))
    elif height < width:
        resized_image = cv2.resize(img_gray, (int(width * target_height / height),
                                              target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,
                                      cropping_length:resized_image.shape[1] - cropping_length]
    else:
        resized_image = cv2.resize(img_gray, (target_height,
                                              int(height * target_width / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:
                                      resized_image.shape[0] - cropping_length]
    return cv2.resize(resized_image, (target_height, target_width))


def preprocess_frame(image, target_height=56, target_width=56):
    image = blackWhite_resize_frame(image, target_height, target_width)
    image = skimage.img_as_float(image).astype(np.float32)
    return image


def extract_smallVids():
    # Read videos list and let  the videos sort by id
    # This should be same as youtube mapping
    videos = sorted(os.listdir(video_root))

    allVid = []
    for i, video in enumerate(videos):
        print(video, end=' ')
        video_path = os.path.join(video_root, video)
        # Extract video frames and video tiles
        frame_list, clip_list, frame_count = sample_frames(video_path, train=True)
        print(frame_count)

        # Image processed and converted to (B, C, H, W) format
        frame_list = np.array([preprocess_frame(x) for x in frame_list])
        allVid.append(frame_list)

    with open(smallvid_path, "wb") as f:
        pickle.dump(f, allVid)


def main():
    extract_smallVids()


if __name__ == '__main__':
    main()
