"""
AgileX Piper robot arm controller.

Assumes the Piper SDK (piper_sdk) is installed.
Install: pip install piper-sdk
Docs:    https://github.com/agilex-robotics/piper_sdk

For MVP the drop zone is hardcoded in robot base frame.
Suction is controlled via the Piper's end-effector IO (gripper channel).
"""

import time
import numpy as np


# Hardcoded drop zone in robot base frame (metres). Tune to your setup.
DROP_ZONE_XYZ = np.array([0.35, -0.20, 0.15])

# How far above the target to hover before descending (metres)
APPROACH_CLEARANCE = 0.10

# Speed factor passed to Piper move commands (0-1)
MOVE_SPEED = 0.3


class RobotController:
    def __init__(self, can_interface: str = "can0"):
        """
        can_interface: CAN bus port the Piper is connected on (e.g. "can0").
        """
        from piper_sdk import C_PiperInterface
        self._arm = C_PiperInterface(can_interface)
        self._arm.ConnectPort()
        self._arm.EnableArm(7)      # enable all joints
        time.sleep(1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def home(self):
        """Move to a safe resting position above the tray."""
        self._arm.MotionCtrl_2(0x01, 0x00, int(MOVE_SPEED * 100))
        # Piper uses joint-angle home via built-in command
        self._arm.JointCtrl(0, 0, 0, 0, 0, 0)
        self._wait_for_motion()

    def pick_and_drop(self, robot_xyz: np.ndarray):
        """
        Full pick sequence:
          1. Move above target
          2. Descend
          3. Suction on
          4. Lift
          5. Move to drop zone
          6. Suction off
          7. Return home
        """
        x, y, z = robot_xyz.tolist()

        # 1. Approach (hover above target)
        self._move_cartesian(x, y, z + APPROACH_CLEARANCE)

        # 2. Descend to pick height
        self._move_cartesian(x, y, z)

        # 3. Activate suction
        self._set_suction(on=True)
        time.sleep(0.3)     # brief dwell to establish seal

        # 4. Lift back to approach height
        self._move_cartesian(x, y, z + APPROACH_CLEARANCE)

        # 5. Move to drop zone (approach height)
        dx, dy, dz = DROP_ZONE_XYZ.tolist()
        self._move_cartesian(dx, dy, dz + APPROACH_CLEARANCE)

        # 6. Release
        self._set_suction(on=False)
        time.sleep(0.2)

        # 7. Home
        self.home()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _move_cartesian(self, x: float, y: float, z: float):
        """Move end-effector to (x, y, z) in robot base frame. Blocking."""
        # Piper SDK expects mm and millidegrees; convert metres -> mm.
        # Orientation: keep wrist pointing straight down (Rx=0, Ry=0, Rz=0).
        self._arm.EndPoseCtrl(
            int(x * 1000), int(y * 1000), int(z * 1000),
            0, 0, 0,
            int(MOVE_SPEED * 100)
        )
        self._wait_for_motion()

    def _set_suction(self, on: bool):
        # Piper controls end-effector IO through GripperCtrl.
        # value=1000 = fully closed (suction on), value=0 = open (suction off).
        gripper_val = 1000 if on else 0
        self._arm.GripperCtrl(gripper_val, 1000, 0x01, 0)

    def _wait_for_motion(self, timeout: float = 10.0, poll: float = 0.05):
        """Block until arm reports motion complete or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self._arm.GetArmStatus()
            if status.arm_status.motion_status == 0:   # 0 = idle
                return
            time.sleep(poll)
        raise TimeoutError("Robot motion did not complete within timeout")

    def stop(self):
        self._arm.DisableArm(7)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()
