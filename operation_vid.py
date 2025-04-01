import os
import math
import threading
import time
from datetime import datetime
import json
import requests  # To fetch HRI data from localhost:8008
import av        # For video encoding
import mss       # For screen capturing


from legged_gym import LEGGED_GYM_ROOT_DIR
from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger


import numpy as np
import torch
import pygame

# -------------------------------------------------
# Screen Recording Thread Function
# -------------------------------------------------
def record_screen_thread(stop_event, monitor, frame_interval):
    """
    Continuously captures the screen and encodes frames to a video file until stop_event is set.
    """
    sct = mss.mss()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screen_filename = os.path.join("./vids/go2amp_op/", "screen_record_" + timestamp + ".mp4")
    os.makedirs(os.path.dirname(screen_filename), exist_ok=True)
    
    output = av.open(screen_filename, mode="w")
    stream = output.add_stream("libx264", rate=24)
    stream.width = monitor['width']
    stream.height = monitor['height']
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "ultrafast"}
    
    last_time = time.time()
    while not stop_event.is_set():
        now = time.time()
        elapsed = now - last_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)
        last_time = time.time()
        raw_frame = np.array(sct.grab(monitor))[:, :, :3]  # Capture screen (BGR)
        frame = raw_frame[:, :, ::-1]  # Convert BGR to RGB
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(video_frame):
            output.mux(packet)
    
    # Flush any remaining packets and close file.
    for packet in stream.encode():
        output.mux(packet)
    output.close()
    print("Screen recording finished. Video saved to:", screen_filename)

# -------------------------------------------------
# SpaceMouse Controller Class
# -------------------------------------------------
class SpaceMouseController:
    def __init__(self):
        # Initialize Pygame and the joystick module.
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("No joystick detected! Please connect a SpaceMouse.")
            self.joystick = None
            self.num_axes = 0
        else:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Joystick 0: {self.joystick.get_name()} initialized")
            self.num_axes = self.joystick.get_numaxes()
            print(f"Number of axes: {self.num_axes}")
            print(f"Number of buttons: {self.joystick.get_numbuttons()}")

        # Initialize axes and button states (assuming at least 6 axes).
        # Mapping:
        #   axis 0: robot X speed (lateral)
        #   axis 3: robot Y speed (forward/backward)
        #   axis 4: robot angular velocity (yaw rate)
        #   axis 2: camera pitch adjustment
        #   axis 5: camera yaw adjustment
        self.axis = [0.0] * 6
        self.button_reset = False  # Button 0 for reset.
        self.button_exit = False   # Button 1 used for data-record toggle.
        self.deadzone = 0.1

        # Start a thread to continuously monitor SpaceMouse events.
        self._monitor_thread = threading.Thread(target=self._monitor_controller)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def read(self):
        """
        Returns a list with the following mapped values:
         - Index 0: Robot X speed command (from axis 0).
         - Index 1: Robot Y speed command (from axis 3).
         - Index 2: Robot angular velocity command (from axis 4).
         - Index 3: Camera pitch adjustment (from axis 2).
         - Index 4: Camera yaw adjustment (from axis 5).
         - Index 5: Reset command (button 0 state).
         - Index 6: Data-record toggle (button 1 state).
        """
        if self.joystick is None:
            return [0, 0, 0, 0, 0, 0, 0]
        robot_y_speed = self.axis[0] if len(self.axis) > 0 else 0.0
        robot_x_speed = self.axis[3] if len(self.axis) > 3 else 0.0
        robot_angular = self.axis[4] if len(self.axis) > 4 else 0.0
        cam_pitch = self.axis[2] if len(self.axis) > 2 else 0.0
        cam_yaw = self.axis[5] if len(self.axis) > 5 else 0.0
        return [robot_x_speed, robot_y_speed, robot_angular, cam_pitch, cam_yaw,
                int(self.button_reset), int(self.button_exit)]
    
    def _monitor_controller(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    axis_index = event.axis
                    val = event.value
                    # Apply deadzone filtering.
                    if abs(val) < self.deadzone:
                        val = 0.0
                    if axis_index < len(self.axis):
                        self.axis[axis_index] = val
                elif event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 0:
                        self.button_reset = True
                    elif event.button == 1:
                        self.button_exit = True
                elif event.type == pygame.JOYBUTTONUP:
                    if event.button == 0:
                        self.button_reset = False
                    elif event.button == 1:
                        self.button_exit = False
            time.sleep(0.01)

# Create a global instance of the SpaceMouse controller.
sm = SpaceMouseController()

# -------------------------------------------------
# Main Simulation, Data & Video Recording Function
# -------------------------------------------------
def play(args):
    # Load environment and training configurations.
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Override parameters for testing.
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.env.episode_length_s = 200
    env_cfg.env.show_goal = True
    env_cfg.env.show_dir = True

    train_cfg.runner.amp_num_preload_transitions = 1

    # Prepare the environment.
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    _, _ = env.reset()
    
    obs = env.get_observations()
    # Load policy.
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    EXPORT_POLICY = False  # Change to True to export the policy.
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 20000
    stop_rew_log = env.max_episode_length + 1
    camera_rot = 0
    camera_rot_per_sec = np.pi / 360

    # Scaling factors for joystick-to-command mapping.
    robot_x_speed_scale = 2    # Scaling factor for X speed.
    robot_y_speed_scale = 0.6    # Scaling factor for Y speed.
    robot_angular_scale = 2     # Scaling factor for angular velocity.
    cam_pitch_scale = 0.1       # Scaling factor for camera pitch.
    cam_yaw_scale = 0.03        # Scaling factor for camera yaw.

    # Data and video recording variables.
    recording_active = False
    episode_data = []
    screen_recorder_thread = None
    screen_recorder_stop_event = None

    # Main control loop.
    for i in range(100 * int(env.max_episode_length)):
        actions = policy(obs.detach())
        current_sm_command = sm.read()

        # --- Check for environment reset (using button 0) ---
        if current_sm_command[5] == 1:
            if recording_active and len(episode_data) > 0:
                os.makedirs("./data/go2amp_op/", exist_ok=True)
                data_filename = os.path.join("./data/go2amp_op/", "{}.json".format(
                    datetime.now().strftime("%Y%m%d_%H%M%S")))
                with open(data_filename, "w") as f:
                    json.dump({"flag": "fail", "data": episode_data}, f)
                print("Reset triggered. Saved data (fail):", data_filename)
            # Stop video recording if active.
            if screen_recorder_stop_event is not None:
                screen_recorder_stop_event.set()
            if screen_recorder_thread is not None:
                screen_recorder_thread.join()
                screen_recorder_thread = None
                screen_recorder_stop_event = None
            episode_data = []
            recording_active = False
            # Reset the environment.
            random_x = np.random.uniform(-1.0, 1.0)
            random_y = np.random.uniform(-1.0, 1.0)
            random_z = env.root_states[0, 2].item()
            random_yaw = np.random.uniform(-np.pi, np.pi)
            env.reset()
            new_pos = torch.tensor([random_x, random_y, random_z], device=env.root_states.device)
            new_orient = torch.tensor([np.cos(random_yaw/2), 0.0, 0.0, np.sin(random_yaw/2)], device=env.root_states.device)
            env.root_states[0, :3] = new_pos
            env.root_states[0, 3:7] = new_orient

        # --- Start data (and video) recording when the X axis command exceeds threshold ---
        if (not recording_active) and (abs(current_sm_command[0]) > 0.3):
            recording_active = True
            episode_data = []
            print("Data recording started.")
            # Start video recording.
            screen_recorder_stop_event = threading.Event()
            sct = mss.mss()
            monitor = sct.monitors[1]  # Primary monitor; adjust if necessary.
            frame_interval = 1.0 / 24  # Target 30 fps.
            screen_recorder_thread = threading.Thread(
                target=record_screen_thread,
                args=(screen_recorder_stop_event, monitor, frame_interval)
            )
            screen_recorder_thread.start()

        # Map joystick inputs to robot commands.
        env.commands[0, 0] = - current_sm_command[0] * robot_x_speed_scale
        env.commands[0, 1] = - current_sm_command[1] * robot_y_speed_scale
        env.commands[0, 2] = current_sm_command[2] * robot_angular_scale
        
        # Map joystick inputs for camera control.
        cam_pitch = current_sm_command[3] * cam_pitch_scale
        cam_yaw = current_sm_command[4] * cam_yaw_scale

        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())
        
        # Update camera position if enabled.
        if MOVE_CAMERA:
            look_at = np.array(env.root_states[0, :3].cpu(), dtype=np.float64)
            camera_rot = (camera_rot + camera_rot_per_sec * env.dt + cam_yaw) % (2 * np.pi)
            camera_pitch_val = cam_pitch % (2 * np.pi)
            camera_relative_position = 1.2 * np.array([np.cos(camera_rot),
                                                       np.sin(camera_rot),
                                                       0.45 + np.sin(camera_pitch_val)])
            env.set_camera(look_at + camera_relative_position, look_at)

        # Record state log data.
        if i < stop_state_log:
            state_data = {
                'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                'dof_torque': env.torques[robot_index, joint_index].item(),
                'command_x': env.commands[robot_index, 0].item(),
                'command_y': env.commands[robot_index, 1].item(),
                'command_yaw': env.commands[robot_index, 2].item(),
                'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy().tolist()
            }
            logger.log_states(state_data)
        elif i == stop_state_log:
            logger.plot_states()

        if 0 < i < stop_rew_log:
            if infos.get("episode", None):
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()

        # --- Record frame data if recording is active ---
        if recording_active:
            try:
                frame_data = {
                    "frame_index": i,
                    "js_input": current_sm_command,
                    "obs": obs.detach().cpu().numpy().tolist(),
                    "action": actions.detach().cpu().numpy().tolist(),
                    "state_data": state_data  # Recorded state log data.
                }
                # Optionally record terrain data if available.
                if hasattr(env, "terrain") and hasattr(env, "surround_heights"):
                    heights = env.surround_heights()
                    frame_data["terrain"] = heights.cpu().numpy().tolist()

                # Retrieve HRI data from localhost:8080.
                try:
                    response = requests.get("http://localhost:8080/data", timeout=0.5)
                    if response.status_code == 200:
                        hri_data = response.json()
                    else:
                        hri_data = {"error": f"HTTP {response.status_code}"}
                except Exception as e:
                    hri_data = {"error": str(e)}
                frame_data["hri_data"] = hri_data

                episode_data.append(frame_data)
            except Exception as e:
                print("Error recording frame data:", e)

        # --- Check for success: if the robot is within 0.6m of the goal, save data and stop recording ---
        robot_pos = env.root_states[0, :2].cpu().numpy()
        goal_pos = env.goal[0, :2].cpu().numpy()
        if recording_active and np.linalg.norm(robot_pos - goal_pos) < 0.6:
            os.makedirs("./data/go2amp_op/", exist_ok=True)
            data_filename = os.path.join("./data/go2amp_op/", "{}.json".format(
                datetime.now().strftime("%Y%m%d_%H%M%S")))
            with open(data_filename, "w") as f:
                json.dump({"flag": "success", "data": episode_data}, f)
            print("Goal reached. Saved data (success):", data_filename)
            # Stop video recording.
            if screen_recorder_stop_event is not None:
                screen_recorder_stop_event.set()
            if screen_recorder_thread is not None:
                screen_recorder_thread.join()
                screen_recorder_thread = None
                screen_recorder_stop_event = None
            episode_data = []
            recording_active = False
            env.reset()

        # Update observations for the next iteration.
        obs = env.get_observations()

    # Ensure that video recording is stopped if still active after the loop ends.
    if recording_active and screen_recorder_stop_event is not None:
        screen_recorder_stop_event.set()
    if screen_recorder_thread is not None:
        screen_recorder_thread.join()

if __name__ == '__main__':
    # Set these flags as needed.
    RECORD_FRAMES = False
    RECORD_VID = False
    RECORD_DATA = True
    MOVE_CAMERA = True
    args = get_args()
    play(args)
    pygame.quit()
