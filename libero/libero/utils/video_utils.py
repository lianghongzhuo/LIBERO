import os
import imageio
import numpy as np
import cv2


def save_rollout_video(rollout_images, mp4_path, rotate, fps, title=None):
    """
    Saves a sequence of images as an MP4 video file.

    Args:
        rollout_images (list or iterable): A sequence of images (numpy arrays) to be written to the video.
        mp4_path (str): The file path where the MP4 video will be saved.
        fps (int): Frames per second for the output video.
        rotate (int, 90, 180, 270): The angle by which to rotate the images before saving.
        title (str, optional): A title to be displayed on the video. Defaults to None.

    Side Effects:
        Writes the video file to the specified path.
        Prints a message indicating the location of the saved video.
    """
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        if rotate != 0:
            assert rotate in [0, 90, 180, 270], "Rotate must be one of [0, 90, 180, 270]"
            img = np.rot90(img, k=rotate // 90)
            img = img.astype(np.uint8)
        if title is not None:
            # Add title to the image
            img = cv2.putText(img, title, (60, 60), cv2.FONT_HERSHEY_SIMPLEX,
                              1.5, (255, 255, 255), 2, cv2.LINE_AA)
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")


class VideoWriter:
    def __init__(self, video_path, save_video=False, fps=30, single_video=True):
        self.video_path = video_path
        self.save_video = save_video
        self.fps = fps
        self.image_buffer = {}
        self.last_images = {}
        self.single_video = single_video

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()

    def append_image(self, img, idx=0):
        """Directly append an image to the video."""
        if self.save_video:
            if idx not in self.image_buffer:
                self.image_buffer[idx] = []
            self.image_buffer[idx].append(img)

    def append_obs(self, obs, done, idx=0, camera_name="agentview_image"):
        """Append a camera observation to the video."""
        if self.save_video:
            if idx not in self.image_buffer:
                self.image_buffer[idx] = []
            if idx not in self.last_images:
                self.last_images[idx] = None
            if not done:
                self.image_buffer[idx].append(obs[camera_name][::-1])
            else:
                if self.last_images[idx] is None:
                    self.last_images[idx] = obs[camera_name][::-1]
                original_image = np.copy(self.last_images[idx])
                blank_image = np.ones_like(original_image) * 128
                blank_image[:, :, 0] = 0
                blank_image[:, :, -1] = 0
                transparency = 0.7
                original_image = (
                    original_image * (1 - transparency) + blank_image * transparency
                )

                self.image_buffer[idx].append(original_image.astype(np.uint8))

    def reset(self):
        if self.save_video:
            self.last_images = {}

    def append_vector_obs(self, obs, dones, camera_name="agentview_image"):
        if self.save_video:
            for i in range(len(obs)):
                self.append_obs(obs[i], dones[i], i, camera_name)

    def save(self):
        if self.save_video:
            os.makedirs(self.video_path, exist_ok=True)
            if self.single_video:
                video_name = os.path.join(self.video_path, f"video.mp4")
                video_writer = imageio.get_writer(video_name, fps=self.fps)
                for idx in self.image_buffer.keys():
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im)
                video_writer.close()
            else:
                for idx in self.image_buffer.keys():
                    video_name = os.path.join(self.video_path, f"{idx}.mp4")
                    video_writer = imageio.get_writer(video_name, fps=self.fps)
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im)
                    video_writer.close()
            print(f"Saved videos to {self.video_path}.")
