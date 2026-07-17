import logging
import threading
import time
import yaml

import numpy as np
from booster_robotics_sdk_python import (
    B1LocoClient,
    B1LowCmdPublisher,
    B1LowStateSubscriber,
    ChannelFactory,
    LowCmd,
    LowState,
    RobotMode,
)

from utils.command import create_first_frame_rl_cmd, create_prepare_cmd
from utils.policy_walk_getup_k1 import Policy
from utils.remote_control_service import RemoteControlService
from utils.rotate import rotate_vector_inverse_rpy
from utils.timer import Timer, TimerConfig


class Controller:
    def __init__(self, cfg_file) -> None:
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        self.remote_control_service = RemoteControlService()
        self.policy = Policy(cfg=self.cfg)

        self._init_timer()
        self._init_low_state_values()
        self._init_communication()
        self.publish_runner = None
        self.running = True

    def _init_timer(self):
        self.timer = Timer(TimerConfig(time_step=self.cfg["common"]["dt"]))
        self.next_publish_time = self.timer.get_time()
        self.next_inference_time = self.timer.get_time()

    def _init_low_state_values(self):
        joint_cnt = self.cfg["common"]["joint_cnt"]
        self.base_rpy = np.zeros(3, dtype=np.float32)
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(joint_cnt, dtype=np.float32)
        self.dof_vel = np.zeros(joint_cnt, dtype=np.float32)
        self.dof_target = np.zeros(joint_cnt, dtype=np.float32)
        self.filtered_dof_target = np.zeros(joint_cnt, dtype=np.float32)
        self.dof_pos_latest = np.zeros(joint_cnt, dtype=np.float32)

    def _init_communication(self) -> None:
        self.low_cmd = LowCmd()
        self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)
        self.low_cmd_publisher = B1LowCmdPublisher()
        self.client = B1LocoClient()

        self.low_state_subscriber.InitChannel()
        self.low_cmd_publisher.InitChannel()
        self.client.Init()

    def _low_state_handler(self, low_state_msg: LowState):
        self.timer.tick_timer_if_sim()
        time_now = self.timer.get_time()
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            self.dof_pos_latest[i] = motor.q
        if time_now >= self.next_inference_time:
            self.base_rpy[:] = low_state_msg.imu_state.rpy
            self.projected_gravity[:] = rotate_vector_inverse_rpy(
                low_state_msg.imu_state.rpy[0],
                low_state_msg.imu_state.rpy[1],
                low_state_msg.imu_state.rpy[2],
                np.array([0.0, 0.0, -1.0]),
            )
            self.base_ang_vel[:] = low_state_msg.imu_state.gyro
            for i, motor in enumerate(low_state_msg.motor_state_serial):
                self.dof_pos[i] = motor.q
                self.dof_vel[i] = motor.dq

    def _send_cmd(self, cmd: LowCmd):
        self.low_cmd_publisher.Write(cmd)

    def cleanup(self) -> None:
        self.remote_control_service.close()
        if hasattr(self, "low_cmd_publisher"):
            self.low_cmd_publisher.CloseChannel()
        if hasattr(self, "low_state_subscriber"):
            self.low_state_subscriber.CloseChannel()
        if hasattr(self, "publish_runner") and self.publish_runner is not None:
            self.publish_runner.join(timeout=1.0)

    def start_custom_mode_conditionally(self):
        print(self.remote_control_service.get_custom_mode_operation_hint())
        while True:
            if self.remote_control_service.start_custom_mode():
                break
            time.sleep(0.1)
        create_prepare_cmd(self.low_cmd, self.cfg)
        for i in range(self.cfg["common"]["joint_cnt"]):
            self.dof_target[i] = self.low_cmd.motor_cmd[i].q
            self.filtered_dof_target[i] = self.low_cmd.motor_cmd[i].q
        self._send_cmd(self.low_cmd)
        self.client.ChangeMode(RobotMode.kCustom)

    def start_rl_gait_conditionally(self):
        print(self.remote_control_service.get_rl_gait_operation_hint())
        while True:
            if self.remote_control_service.start_rl_gait():
                break
            time.sleep(0.1)
        create_first_frame_rl_cmd(self.low_cmd, self.cfg)
        self._send_cmd(self.low_cmd)
        self.next_inference_time = self.timer.get_time()
        self.next_publish_time = self.timer.get_time()
        self.publish_runner = threading.Thread(target=self._publish_cmd)
        self.publish_runner.daemon = True
        self.publish_runner.start()
        print(self.remote_control_service.get_operation_hint())

    def run(self):
        time_now = self.timer.get_time()
        if time_now < self.next_inference_time:
            time.sleep(0.001)
            return
        self.next_inference_time += self.policy.get_policy_interval()
        self.dof_target[:] = self.policy.inference(
            time_now=time_now,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            base_ang_vel=self.base_ang_vel,
            projected_gravity=self.projected_gravity,
            base_rpy=self.base_rpy,
            vx=self.remote_control_service.get_vx_cmd(),
            vy=self.remote_control_service.get_vy_cmd(),
            vyaw=self.remote_control_service.get_vyaw_cmd(),
        )
        time.sleep(0.001)

    def _publish_cmd(self):
        publish_alpha = float(self.cfg.get("switching", {}).get("publish_filter_alpha", 0.2))
        while self.running:
            time_now = self.timer.get_time()
            if time_now < self.next_publish_time:
                time.sleep(0.001)
                continue
            self.next_publish_time += self.cfg["common"]["dt"]
            self.filtered_dof_target = self.filtered_dof_target * (1.0 - publish_alpha) + self.dof_target * publish_alpha
            for i in range(self.cfg["common"]["joint_cnt"]):
                self.low_cmd.motor_cmd[i].q = self.filtered_dof_target[i]
            self._send_cmd(self.low_cmd)
            time.sleep(0.001)

    def __enter__(self) -> "Controller":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()


if __name__ == "__main__":
    import argparse
    import os
    import signal
    import sys

    def signal_handler(sig, frame):
        print("\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Name of the configuration file.")
    parser.add_argument("--net", type=str, default="127.0.0.1", help="Network interface for SDK communication.")
    args = parser.parse_args()
    cfg_file = os.path.join("configs", args.config)

    print(f"Starting walk/getup custom controller, connecting to {args.net} ...")
    ChannelFactory.Instance().Init(0, args.net)

    with Controller(cfg_file) as controller:
        time.sleep(2)
        print("Initialization complete.")
        controller.start_custom_mode_conditionally()
        controller.start_rl_gait_conditionally()

        try:
            while controller.running:
                controller.run()
            controller.client.ChangeMode(RobotMode.kDamping)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Cleaning up...")
            controller.cleanup()
