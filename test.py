import sys

import numpy as np
from scipy.spatial.transform import Rotation


def strip_fixes(s, prefix=None, suffix=None):
    if prefix and s.startswith(prefix):
        s = s[len(prefix) :]
    if suffix and s.endswith(suffix):
        s = s[: -len(suffix)]
    return s


def parse_inertia(s):
    vals = s.split(" ")
    vals = vals[1:]
    I = {}
    for i, v in enumerate(vals):
        vals[i] = strip_fixes(v, prefix="<", suffix="/>")
        kv = vals[i].split("=")
        I[kv[0]] = float(kv[1])

    return np.array([[I["ixx"], I["ixy"], I["ixz"]], [I["ixy"], I["iyy"], I["iyz"]], [I["ixz"], I["iyz"], I["izz"]]])


s = sys.argv[1]
I = parse_inertia(s)

# I = np.array([[0.00457, 1e-6, 2e-6], [1e-6, 0.004831, 4.48e-4], [2e-6, 4.48e-4, 0.001409]])

eigvals, eigvecs = np.linalg.eigh(I)
# eigvals → diaginertia (MuJoCo wants them in descending or any order matching eigvecs)

# eigvecs columns are the principal axes; convert rotation matrix to quaternion
R = eigvecs
r = Rotation.from_matrix(R)
quat_xyzw = r.as_quat()  # scipy: x, y, z, w
quat_wxyz = np.roll(quat_xyzw, 1)  #
print(quat_wxyz)
