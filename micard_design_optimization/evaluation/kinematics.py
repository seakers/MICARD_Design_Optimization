"""Forward kinematics + Jacobian for a serial chain of revolute joints.

The chain grows along local +X; each joint rotates about its orientation
axis (roll=X, pitch=Y, yaw=Z) [design.py]. This backs reachability and the
Yoshikawa manipulability index [MICARD 1.0].
"""
import numpy as np

from micard_design_optimization.utils.design import ORIENTATION_AXES


def _rotation(axis, theta):
    """Rodrigues rotation matrix for a unit axis and angle."""
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    C = 1 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
    ])


def joint_transforms(design, joint_angles):
    """Return the world-frame position + rotation after each joint.

    Each segment: rotate about the joint axis by its angle, then translate
    along local +X by the link length.
    """
    R = np.eye(3)
    p = np.zeros(3)
    axes_world = []      # joint axis in world frame (for the Jacobian)
    positions = [p.copy()]

    for seg, theta in zip(design.segments, joint_angles):
        axis_local = np.array(ORIENTATION_AXES[seg.orientation])
        axis_world = R @ axis_local
        axes_world.append(axis_world)
        R = R @ _rotation(axis_local, theta)
        p = p + R @ np.array([seg.link_length_m, 0.0, 0.0])
        positions.append(p.copy())

    return np.array(positions), axes_world


def forward_kinematics(design, joint_angles):
    """End-effector position (m)."""
    positions, _ = joint_transforms(design, joint_angles)
    return positions[-1]


def jacobian(design, joint_angles):
    """Geometric Jacobian (6 x n): linear + angular velocity contributions."""
    positions, axes_world = joint_transforms(design, joint_angles)
    p_end = positions[-1]
    n = len(design.segments)
    J = np.zeros((6, n))
    for i in range(n):
        axis = axes_world[i]
        # joint i acts at joint origin positions[i]
        J[:3, i] = np.cross(axis, p_end - positions[i])  # linear
        J[3:, i] = axis                                   # angular
    return J