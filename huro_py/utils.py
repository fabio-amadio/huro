import numpy as np

def quat_rotate_inverse(q, v):
    q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]
    q_conj = np.array([q_w, -q_x, -q_y, -q_z])
    t = 2.0 * np.cross(q_conj[1:], v)  # ✅ Correct: q_conj[1:] = [x, y, z]
    return v + q_conj[0] * t + np.cross(q_conj[1:], t)  # ✅ Complete formula