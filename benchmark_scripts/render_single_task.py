import os
from termcolor import colored
import cv2
import h5py
import argparse
import numpy as np

from libero.libero.envs import OffScreenRenderEnv, MjViewerRenderEnv
from libero.libero import benchmark, get_libero_path
from libero.libero.utils.video_utils import save_rollout_video
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_name", type=str)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--bddl_file", type=str)
    parser.add_argument("--demo_file", type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    benchmark_name = args.benchmark_name
    task_id = args.task_id
    bddl_file = args.bddl_file
    demo_file = args.demo_file

    benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 1280,
        "camera_widths": 1280,
        "has_renderer": True if args.debug else False,
    }

    os.makedirs("benchmark_tasks", exist_ok=True)

    task = benchmark_instance.get_task(task_id)
    init_states = benchmark_instance.get_task_init_states(task_id)
    if args.debug:
        env = MjViewerRenderEnv(**env_args)
    else:
        env = OffScreenRenderEnv(**env_args)
    env.reset()
    obs = env.set_init_state(init_states[0])
    for _ in range(5):
        obs, _, _, _ = env.step([0.0] * 7)
    init_image = obs["agentview_image"]
    images = [init_image]

    with h5py.File(demo_file, "r") as f:
        states = f["data/demo_0/states"][()]
        obs = env.set_init_state(states[-1])

    images.append(obs["agentview_image"])
    images = np.concatenate(images, axis=1)
    image_name = demo_file.split("/")[-1].replace(".hdf5", ".png")
    cv2.imwrite(f"benchmark_tasks/{image_name}", images[::-1, :, ::-1])
    images = [init_image]
    for s in states:
        obs = env.set_init_state(s)
        images.append(obs["agentview_image"])
        if args.debug:
            obs, _, _, _ = env.step([0.0] * 7)
            time.sleep(0.05)
    video_name = demo_file.split("/")[-1].replace(".hdf5", ".mp4")
    save_rollout_video(images, f"benchmark_tasks/{video_name}", 180, 30)
    env.close()


if __name__ == "__main__":
    main()
