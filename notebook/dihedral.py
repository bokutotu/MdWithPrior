import numpy as np


def cal_dihedral(v1, v2, v3, v4):
    ab = v1 - v2
    cb = v3 - v2
    db = v4 - v3
    u = pow(ab, cb)
    v = pow(db, cb)
    w = pow(u, v)
    angle = _angle(u, v)
    # Determine sign of angle
    try:
        if _angle(cb, w) > 0.001:
            angle = -angle
    except ZeroDivisionError:
        # dihedral=pi
        pass
    return angle


def pow(x, y):
    a, b, c = x
    d, e, f = y
    c1 = np.linalg.det(np.array(((b, c), (e, f))))
    c2 = -np.linalg.det(np.array(((a, c), (d, f))))
    c3 = np.linalg.det(np.array(((a, b), (d, e))))
    return np.array((c1, c2, c3))


def _angle(self, other):
    """Return angle between two vectors."""
    n1 = norm(self)
    n2 = norm(other)
    c = (self * other) / (n1 * n2)
    # Take care of roundoff errors
    # c = min(c, 1)
    # c = max(-1, c)
    return np.arccos(c)


def norm(self):
    """Return vector norm."""
    return np.sqrt(sum(self * self))
