"""Panda MP demo generator with DART noise + pd_ee_delta_pose labels + state obs.

Companion to ``run.py``. Env runs in pd_joint_pos (required by mplib-based MP
solver), but the h5 is rewritten with ``pd_ee_delta_pose`` action labels
computed inline via pinocchio FK. This preserves the DART property: the clean
label at step t is the expert's delta-pose response to the *actual* (noised)
state visited at t.

Supports GPU sim backend (``physx_cuda``) for native GPU-side state obs — MP
solver still runs CPU-side on mplib, GPU/CPU qpos bridging is handled via
``.cpu().numpy()``. Multi-env GPU simulation is not supported (MP is serial).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import os.path as osp
import time
from copy import deepcopy

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from mani_skill.examples.motionplanning.dart_noise import (
    PandaDeltaPoseDartNoise,
    rewrite_actions_to_delta_pose,
    wrap_env_step,
)
from mani_skill.examples.motionplanning.panda.solutions import (
    solveDrawSVG,
    solveDrawTriangle,
    solveLiftPegUpright,
    solvePegInsertionSide,
    solvePickCube,
    solvePlaceSphere,
    solvePlugCharger,
    solvePullCube,
    solvePullCubeTool,
    solvePushCube,
    solveStackCube,
    solveStackPyramid,
)
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.utils.wrappers.record import RecordEpisode

MP_SOLUTIONS = {
    "DrawTriangle-v1": solveDrawTriangle,
    "PickCube-v1": solvePickCube,
    "StackCube-v1": solveStackCube,
    "PegInsertionSide-v1": solvePegInsertionSide,
    "PlugCharger-v1": solvePlugCharger,
    "PlaceSphere-v1": solvePlaceSphere,
    "PushCube-v1": solvePushCube,
    "PullCubeTool-v1": solvePullCubeTool,
    "LiftPegUpright-v1": solveLiftPegUpright,
    "PullCube-v1": solvePullCube,
    "DrawSVG-v1": solveDrawSVG,
    "StackPyramid-v1": solveStackPyramid,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Record Panda MP demos with DART noise + pd_ee_delta_pose labels."
    )
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1")
    parser.add_argument("-o", "--obs-mode", type=str, default="state",
                        help="Obs mode to record. Default 'state'.")
    parser.add_argument("-n", "--num-traj", type=int, default=10)
    parser.add_argument("--only-count-success", action="store_true")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-b", "--sim-backend", type=str, default="auto",
                        help="'auto', 'physx_cpu', or 'physx_cuda'. GPU supported.")
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--traj-name", type=str)
    parser.add_argument("--shader", default="default", type=str)
    parser.add_argument("--record-dir", type=str, default="demos")
    parser.add_argument("--num-procs", type=int, default=1,
                        help="CPU multiprocessing. Only valid with --sim-backend physx_cpu.")
    parser.add_argument("--action-noise-scale", type=float, default=0.1,
                        help="DART action noise std as fraction of per-step action delta. 0 disables.")
    parser.add_argument("--max-joint-noise", type=float, default=0.05)
    parser.add_argument("--noise-seed", type=int, default=None)
    parser.add_argument("--robot-uids", type=str, default=None,
                        help="Override env default robot (e.g. 'panda' vs 'panda_wristcam'). "
                             "Both share the panda_hand_tcp ee link.")
    return parser.parse_args()


def _main(args, proc_id: int = 0, start_seed: int = 0) -> str:
    env_id = args.env_id
    if env_id not in MP_SOLUTIONS:
        raise RuntimeError(
            f"No MP solution for {env_id}. Available: {list(MP_SOLUTIONS.keys())}"
        )

    env_kwargs = dict(
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        sim_backend=args.sim_backend,
    )
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = args.robot_uids
    env = gym.make(env_id, **env_kwargs)

    if not args.traj_name:
        new_traj_name = time.strftime("%Y%m%d_%H%M%S")
    else:
        new_traj_name = args.traj_name
    if args.num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)

    env = RecordEpisode(
        env,
        output_dir=osp.join(args.record_dir, env_id, "motionplanning"),
        trajectory_name=new_traj_name,
        save_video=args.save_video,
        source_type="motionplanning",
        source_desc="Panda MP + DART + pd_ee_delta_pose labels (run_bc.py)",
        video_fps=30,
        record_reward=False,
        save_on_reset=False,
    )
    output_h5_path = env._h5_file.filename

    noise = PandaDeltaPoseDartNoise(
        scale=args.action_noise_scale,
        max_noise=args.max_joint_noise,
        gripper_dims=1,
        seed=args.noise_seed,
    )
    noise.bind(env)
    wrap_env_step(env, noise)
    print(
        f"[run_bc] DART noise scale={noise.scale} max={noise.max_noise} "
        f"seed={args.noise_seed} backend={args.sim_backend}"
    )

    solve = MP_SOLUTIONS[env_id]
    print(f"[run_bc] Motion Planning Running on {env_id}")
    pbar = tqdm(range(args.num_traj), desc=f"proc_id: {proc_id}")
    seed = start_seed
    successes: list[bool] = []
    solution_episode_lengths: list[int] = []
    failed_motion_plans = 0
    passed = 0

    while True:
        if args.noise_seed is None:
            noise.rng = np.random.default_rng(seed)
        noise.reset()

        try:
            res = solve(env, seed=seed, debug=False, vis=bool(args.vis))
        except Exception as e:
            print(f"[run_bc] MP error: {e}")
            res = -1

        if res == -1:
            success = False
            failed_motion_plans += 1
        else:
            success = res[-1]["success"].item()
            elapsed_steps = res[-1]["elapsed_steps"].item()
            solution_episode_lengths.append(elapsed_steps)
        successes.append(success)

        if args.only_count_success and not success:
            seed += 1
            env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
            continue

        env.flush_trajectory()
        rewrite_actions_to_delta_pose(env, noise)
        if args.save_video:
            env.flush_video()
        pbar.update(1)
        pbar.set_postfix(
            dict(
                success_rate=np.mean(successes),
                failed_motion_plan_rate=failed_motion_plans / (seed + 1),
                avg_episode_length=np.mean(solution_episode_lengths) if solution_episode_lengths else 0,
                max_episode_length=np.max(solution_episode_lengths) if solution_episode_lengths else 0,
            )
        )
        seed += 1
        passed += 1
        if passed == args.num_traj:
            break

    env.close()
    return output_h5_path


def main(args):
    if args.num_procs > 1 and args.sim_backend == "physx_cuda":
        raise ValueError("--num-procs > 1 is CPU multiprocessing; use --sim-backend physx_cpu.")
    if args.num_procs > 1 and args.num_procs < args.num_traj:
        if args.num_traj < args.num_procs:
            raise ValueError("num_traj must be >= num_procs")
        args.num_traj = args.num_traj // args.num_procs
        seeds = [*range(0, args.num_procs * args.num_traj, args.num_traj)]
        pool = mp.Pool(args.num_procs)
        proc_args = [(deepcopy(args), i, seeds[i]) for i in range(args.num_procs)]
        res = pool.starmap(_main, proc_args)
        pool.close()
        output_path = res[0][: -len("0.h5")] + "h5"
        merge_trajectories(output_path, res)
        for h5_path in res:
            tqdm.write(f"Remove {h5_path}")
            os.remove(h5_path)
            json_path = h5_path.replace(".h5", ".json")
            tqdm.write(f"Remove {json_path}")
            os.remove(json_path)
    else:
        _main(args)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main(parse_args())
