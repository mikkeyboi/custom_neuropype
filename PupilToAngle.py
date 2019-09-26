import logging
import numpy as np
from neuropype.engine import *

logger = logging.getLogger(__name__)


class PupilToAngle(Node):
    # --- Input/output ports ---
    data = Port(None, Packet, "Data to process.", required=True,
                editable=False, mutating=True)

    # --- Properties ---
    use_3d_gaze = BoolPort(True, help="""Use 3D gaze data. Else use 2D norm position.""")

    @classmethod
    def description(cls):
        return Description(name='Pupil To Angle',
                           description="""Convert Pupil-Labs 3D gaze data to degrees of visual angle.
                           https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/eye_movement/utils.py#L74
                           """,
                           version='0.1',
                           license=Licenses.MIT)

    @data.setter
    def data(self, pkt):
        for n, chnk in enumerate_chunks(pkt, nonempty=True, only_signals=True, with_axes=(time,)):

            if self.use_3d_gaze:
                keep_chans = ['gaze_point_3d_' + _ for _ in ['x', 'y', 'z']]
                dat_3d = chnk.block[space[keep_chans], ...].data.astype(np.float32)
                # Sometimes it is possible that the predicted gaze is
                # behind the camera which is physically impossible.
                dat_3d[:, dat_3d[2, :] < 0] *= -1.0
            else:
                keep_chans = ['norm_pos_' + _ for _ in ['x', 'y']]
                dat_2d = chnk.block[space[keep_chans], ...].data.astype(np.float32)
                width, height = [1000, 1000]
                dat_2d[0] *= width
                dat_2d[1] = (1.0 - dat_2d[1]) * height
                dat_3d = unprojectPoints(dat_2d)

            # Convert x,y,z to degrees visual angle.
            r, theta, psi = cart_to_spherical(dat_3d)
            angles = [theta, psi]
            angles = np.rad2deg(angles)
            ang_axes = (SpaceAxis(names=['gaze_ang_deg_' + _ for _ in ['x', 'y']]),
                        deepcopy_most(chnk.block.axes[time]))
            ang_blk = Block(data=angles, axes=ang_axes)
            chnk.block = concat(space, chnk.block, ang_blk)

        self._data = pkt


def cart_to_spherical(xyz):
    # convert to spherical coordinates
    # source: http://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    r = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)
    theta = np.arccos(xyz[1] / r)  # for elevation angle defined from Z-axis down
    psi = np.arctan2(xyz[2], xyz[0])
    return r, theta, psi


def unprojectPoints(pts_2d, use_distortion=True, normalize=False):
    """
    Undistorts points according to the camera model.
    cv2.fisheye.undistortPoints does *NOT* perform the same unprojection step the original cv2.unprojectPoints does.
    Thus we implement this function ourselves.
    https://github.com/pupil-labs/pupil/blob/6bef222339317092ac8e8392187d8485f4290979/pupil_src/shared_modules/camera_models.py#L342-L392

    :param pts_2d:, shape: Nx2:
    :param use_distortion:
    :param normalize:
    :return: Array of unprojected 3d points, shape: Nx3
    """
    import cv2  # pip install opencv-python-headless
    # intrinsics taken from dummy world camera with 1280 x 720 resolution.
    intrinsics = {
        'K': np.array(
            [[1000.,    0.,  640.],
             [0.,    1000.,  360.],
             [0.,       0.,    1.]]),
        'D': np.array([[0., 0., 0., 0., 0.]])
    }

    pts_2d = np.array(pts_2d.T, dtype=np.float32)

    eps = np.finfo(np.float32).eps

    f = np.array((intrinsics['K'][0, 0], intrinsics['K'][1, 1])).reshape(1, 2)
    c = np.array((intrinsics['K'][0, 2], intrinsics['K'][1, 2])).reshape(1, 2)
    if use_distortion:
        k = intrinsics['D'].ravel().astype(np.float32)
    else:
        k = np.asarray(
            [1.0 / 3.0, 2.0 / 15.0, 17.0 / 315.0, 62.0 / 2835.0], dtype=np.float32
        )

    pi = pts_2d.astype(np.float32)
    pw = (pi - c) / f

    theta_d = np.linalg.norm(pw, ord=2, axis=1)
    theta = theta_d
    for j in range(10):
        theta2 = theta ** 2
        theta4 = theta2 ** 2
        theta6 = theta4 * theta2
        theta8 = theta6 * theta2
        theta = theta_d / (
            1 + k[0] * theta2 + k[1] * theta4 + k[2] * theta6 + k[3] * theta8
        )

    scale = np.tan(theta) / (theta_d + eps)

    pts_2d_undist = pw * scale.reshape(-1, 1)

    pts_3d = cv2.convertPointsToHomogeneous(pts_2d_undist)
    pts_3d.shape = -1, 3

    if normalize:
        pts_3d /= np.linalg.norm(pts_3d, axis=1)[:, np.newaxis]

    return pts_3d.T
