from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv, MjViewerRenderEnv
import os
from libero.libero.utils import get_libero_path
import imageio


def save_rollout_video(rollout_images, image_id):
    rollout_dir = f"./rollouts"
    os.makedirs(rollout_dir, exist_ok=True)
    mp4_path = f"{rollout_dir}/demo_{image_id}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=10)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")


benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10"  # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(
    get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
)
print(
    f"[info] retrieving task {task_id} from suite {task_suite_name}, the "
    + f"language instruction is {task_description}, and the bddl file is {task_bddl_file}"
)

# step over the environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 1280,
    "camera_widths": 1280,
}
# env = OffScreenRenderEnv(**env_args)
env = MjViewerRenderEnv(**env_args)
env.reset()
init_states = task_suite.get_task_init_states(
    task_id
)  # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
env.set_init_state(init_states[init_state_id])
imgs1 = []
imgs2 = []
dummy_action = [0.0] * 7
for step in range(100):
    obs, reward, done, info = env.step(dummy_action)
    imgs1.append(obs["agentview_image"])
    imgs2.append(obs["robot0_eye_in_hand_image"])
env.close()
save_rollout_video(imgs1, 1)
save_rollout_video(imgs2, 2)
