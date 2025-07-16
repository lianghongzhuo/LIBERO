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
    root_path = os.path.join(get_libero_path("benchmark_root"), "../../")
    controller_path = os.path.join(root_path, "config/default_panda.json")
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 1088,
        "camera_widths": 1920,
        "has_renderer": True if args.debug else False,
        "controller": controller_path,
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
    # sim = env.env.sim
    # for i, name in enumerate(sim.model.joint_names):
    #     joint_id = sim.model.joint_name2id(name)
    #     qpos_addr = sim.model.jnt_qposadr[joint_id]
    #     qvel_addr = sim.model.jnt_dofadr[joint_id]
    #     joint_type = sim.model.jnt_type[joint_id]
    #     print(f"[{i}] Joint: {name}, Type: {joint_type}, qpos[{qpos_addr}], qvel[{qvel_addr}]")
    images.append(obs["agentview_image"])
    images = np.concatenate(images, axis=1)
    image_name = demo_file.split("/")[-1].replace(".hdf5", ".png")
    cv2.imwrite(f"benchmark_tasks/{image_name}", images[::-1, :, ::-1])
    images = [init_image]
    for item, s in enumerate(states):
        if task.name == "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket":
            s[52]+= 0.15  # Adjust the butter position
            s[59]+= 0.15  # Adjust the basket position
        elif task.name == "pick_up_the_chocolate_pudding_and_place_it_in_the_basket":
            s[10] -= 0.03  # Adjust the chocolate_pudding position
            s[11] -= 0.005  # Adjust the chocolate_pudding position
            s[17] -= 0.03  # Adjust the basket position
        elif task.name == "pick_up_the_cream_cheese_and_place_it_in_the_basket":
            s[10] -= 0.05
            s[13] += 0.15
            s[17] -= 0.05
        obs = env.set_init_state(s)
        images.append(obs["agentview_image"])
        if args.debug:
            obs, _, _, _ = env.step([0.0] * 7)
            time.sleep(0.02)
    video_name = demo_file.split("/")[-1].replace(".hdf5", ".mp4")
    save_rollout_video(images, f"benchmark_tasks/{video_name}", 180, 60, task.language)
    env.close()


if __name__ == "__main__":
    main()
