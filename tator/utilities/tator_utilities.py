""" Module containing Tator/Algorithm helper functions and classes

Expected to be used between multiple algorithm versions.

"""
import base64
import datetime
import json
import os
import subprocess

import cv2
import numpy as np
import tator

class FrameBuffer():
    """ Helper class used to download video clips/frames of a particular Tator video media

    Use this class to speed up processing a video locally instead of processing individual frames
    using the GetFrame endpoint. This also supports av1 by re-encoding the video locally

    """

    def __init__(
            self,
            tator_api,
            media,
            work_folder: str,
            moving_forward: bool=True,
            buffer_size: int=1000,
            presigned_seconds: int=86400) -> None:
        """ Constructor
        """

        self.tator_api = tator_api
        self.media = tator_api.get_media(id=media.id, presigned=presigned_seconds)
        self.moving_forward = moving_forward
        self.buffer_size = buffer_size
        self.work_folder = work_folder
        self.quality = None
        self.util = tator.util.MediaUtil(work_folder)

        # Default to the highest quality
        for info in self.media.media_files.streaming:
            if self.quality is None:
                self.quality = info.resolution[0]

            elif self.quality < info.resolution[0]:
                self.quality = info.resolution[0]

        # Setup the media utility
        self.util.load_from_media_object(self.media, quality=self.quality)

        # Frames will be indexed by frame number. Each entry will be the 3 channel np matrix
        # that can be directly used by opencv, etc.
        self.frame_buffer = {}

    def get_single_frame(self, frame: int) -> np.ndarray:
        """Doesn't update buffer. Calls equivalent of tator_api.get_frame()
        """

        tile_path = self.util.get_tile_image([frame])
        tile = cv2.imread(tile_path)
        os.remove(tile_path)
        return tile

    def get_frame(self, frame: int) -> np.ndarray:
        """ Returns image to process from cv2.imread
        """

        # Have we already read the frame we care about?
        if frame not in self.frame_buffer:

            # Nope, looks like we need to refresh the buffer.
            # If we are moving backwards in the media, then we should jump further back.
            start_frame = frame
            if not self.moving_forward:
                start_frame = frame - self.buffer_size
                start_frame = 0 if start_frame < 0 else start_frame

            self._refresh_frame_buffer(start_frame=start_frame)

            # Check again, if frame is still not in the frame buffer after refreshing,
            # we've got a problem. And bounce out.
            if frame not in self.frame_buffer:
                raise ValueError("Problem refreshing frame buffer")

        return self.frame_buffer[frame]

    def _refresh_frame_buffer(
            self,
            start_frame: int) -> None:
        """ Refreshes the frame buffer by getting a video clip and reading the video's frames

        Postcondition(s):
            self.frame_buffer is set with numpy arrays, indexed by frame number
        """

        # Request the video clip and download it
        last_frame = start_frame + self.buffer_size
        last_frame = self.media.num_frames - 2 if last_frame > self.media.num_frames - 2 else last_frame

        clip = self.util.get_clip(frame_ranges=[(start_frame,last_frame)])
        orig_clip_path = clip[0]
        clip_path = os.path.join(self.work_folder, "this_clip.mp4")
        segment_list = clip[1]

        print(f"[{datetime.datetime.now()}] libx264: {datetime.datetime.now()}")
        args = ["ffmpeg",
                "-y",
                "-i", clip[0],
                "-vcodec", "libx264",
                "-preset", "veryfast",
                clip_path]
        proc = subprocess.run(args, check=True, capture_output=True)
        print(f"[{datetime.datetime.now()}] libx264 - done: {datetime.datetime.now()}")

        # Create a list of frame numbers associated with the video clip
        frame_list = []
        for segment in segment_list:
            frame_list.extend(list(range(segment["frame_start"], segment["frame_start"] + segment["num_frames"])))

        # With the video downloaded, process the video and save the imagery into the buffer
        self.frame_buffer = {}
        reader = cv2.VideoCapture(clip_path)
        while reader.isOpened():
            ok, frame = reader.read()
            if not ok:
                break
            self.frame_buffer[frame_list.pop(0)] = frame.copy()
        reader.release()
        os.remove(orig_clip_path)
        os.remove(clip_path)

def create_image_folder(
        tator_api: tator.openapi.tator_openapi.api.tator_api.TatorApi,
        media: tator.openapi.tator_openapi.models.media.Media,
        frame_segments: list,
        output_folder: str,
        work_folder: str) -> None:
    """ Create a folder of images for the detection algorithm to process

    :param tator_api: Connected Tator API interface
    :param media: Media to process
    :param frame_segments: Tuples of [[start_frame, end_frame)] (e.g. [(0,1000)] processes frames 1000 frames from 0 to 999)
                           If the start_frame is equal to the end_frame, then only that frame is processed.
    :param output_folder: Folder to put the images in
    :param work_folder: Folder to put temporary files into

    :postcondition: Images will be placed in output_folder with the format {media_id}_{frame_id}.png
    """

    media = tator_api.get_media(id=media.id)
    frame_buffer = FrameBuffer(
        tator_api=tator_api,
        media=media,
        work_folder=work_folder)

    num_frames = 0
    for segment in frame_segments:
        delta = segment[1] - segment[0]
        if delta == 0:
            num_frames += 1
        else:
            num_frames += delta

    print(f"[{datetime.datetime.now()}] Downloading {num_frames} frames for {media.name} (ID: {media.id})...")

    num_frames_created = 0
    for segment in frame_segments:

        if segment[0] == segment[1]:
            frame = segment[0]
            image = frame_buffer.get_single_frame(frame=frame)
            image_path = os.path.join(output_folder, f"{media.id}_{frame}.png")
            cv2.imwrite(image_path, image)

        else:
            for frame in range(segment[0], segment[1]):
                image = frame_buffer.get_frame(frame=frame)
                image_path = os.path.join(output_folder, f"{media.id}_{frame}.png")
                cv2.imwrite(image_path, image)

                num_frames_created += 1
                if (num_frames_created % 100) == 0:
                    print(f"[{datetime.datetime.now()}] Downloaded {num_frames_created} frames of {num_frames}")
