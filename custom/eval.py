from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv, MjViewerRenderEnv
import os
from libero.libero.utils import get_libero_path
from libero.libero.utils.video_utils import save_rollout_video


benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10"  # can also choose libero_goal libero_10 libero_90 libero_spatial, libero_object, etc.
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
    "has_renderer": True,
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
for step in range(1000):
    obs, reward, done, info = env.step(dummy_action)
    imgs1.append(obs["agentview_image"])
    imgs2.append(obs["robot0_eye_in_hand_image"])
env.close()
save_rollout_video(imgs1, "video1.mp4", 180, 30)
save_rollout_video(imgs2, "video2.mp4", 180, 30)
