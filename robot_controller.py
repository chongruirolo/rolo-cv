"""
AgileX Piper robot arm controller.

Assumes the Piper SDK (piper_sdk) is installed.
Install: pip install piper-sdk
Docs:    https://github.com/agilex-robotics/piper_sdk

For MVP the drop zone is hardcoded in robot base frame.
The end-effector is a gripper; GripperCtrl value 0 = open, 1000 = closed.
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
    def __init__(self, can_interface: str = "can0", grip_z_offset_m: float = 0.02):
        """
        can_interface: CAN bus port the Piper is connected on (e.g. "can0").
        """
        self._grip_z_offset = grip_z_offset_m
        from piper_sdk import C_PiperInterface
        self._arm = C_PiperInterface(can_interface)
        self._arm.ConnectPort()
        self._arm.MasterSlaveConfig(0xFC, 0, 0, 0)
        time.sleep(0.1)
        self._arm.MotionCtrl_1(0x00, 0x00, 0x02)  # exit drag-teach mode
        time.sleep(0.1)
        self._arm.EnableArm(7)      # enable all joints
        time.sleep(1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def home(self):
        """Move to a safe resting position above the tray."""
        self._arm.MotionCtrl_2(0x01, 0x00, int(MOVE_SPEED * 100))
        self._arm.JointCtrl(0, 0, 0, 0, 0, 0)
        self._wait_for_motion()

    def pick_and_drop(self, robot_xyz: np.ndarray):
        """
        Full pick sequence:
          1. Open gripper
          2. Move above target
          3. Descend
          4. Close gripper
          5. Lift
          6. Move to drop zone
          7. Open gripper to release
          8. Return home
        """
        x, y, z = robot_xyz.tolist()

        # 1. Open gripper before approaching
        self._open_gripper()

        # 2. Approach (hover above target)
        self._move_cartesian(x, y, z + APPROACH_CLEARANCE)

        # 3. Descend to pick height — offset below top surface so gripper
        #    closes around the middle of the wing, not above it
        self._move_cartesian(x, y, z - self._grip_z_offset)

        # 4. Close gripper to grip the wing
        self._close_gripper()
        time.sleep(0.3)     # brief dwell to confirm grip

        # 5. Lift back to approach height
        self._move_cartesian(x, y, z + APPROACH_CLEARANCE)

        # 6. Move to drop zone (approach height)
        dx, dy, dz = DROP_ZONE_XYZ.tolist()
        self._move_cartesian(dx, dy, dz + APPROACH_CLEARANCE)

        # 7. Release
        self._open_gripper()
        time.sleep(0.2)

        # 8. Home
        self.home()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _move_cartesian(self, x: float, y: float, z: float):
        """Move end-effector to (x, y, z) in robot base frame. Blocking."""
        # Piper SDK expects mm; convert metres -> mm.
        # Orientation: keep wrist pointing straight down (Rx=0, Ry=0, Rz=0).
        self._arm.EndPoseCtrl(
            int(x * 1000), int(y * 1000), int(z * 1000),
            0, 0, 0,
            int(MOVE_SPEED * 100)
        )
        self._wait_for_motion()

    def _open_gripper(self):
        self._arm.GripperCtrl(0, 1000, 0x01, 0)

    def _close_gripper(self):
        self._arm.GripperCtrl(1000, 1000, 0x01, 0)

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
