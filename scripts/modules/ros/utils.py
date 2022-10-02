import numpy as np
from detect.msg import Candidate, RotatedBoundingBox
from geometry_msgs.msg import Point, Quaternion
from image_geometry import PinholeCameraModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from std_msgs.msg import MultiArrayDimension
from tf.transformations import quaternion_from_matrix


class PointProjector:
    def __init__(self, cam_info):
        self.cam_info = cam_info

    def pixel_to_3d(self, u, v, depth, margin_mm=0) -> Point:
        """
        ピクセルをカメラ座標系へ３次元投影
        ---
        u,v: ピクセル位置
        depth: 深度画像
        margin_mm: 物体表面から中心までの距離[mm]
        """
        unit_v = self._get_direction(u, v)
        distance = depth[u, v] / 1000 + margin_mm  # mm to m
        object_point = Point(*(unit_v * distance))
        return object_point

    def _get_direction(self, u, v):
        """カメラ座標系原点から対象点までの方向ベクトルを算出"""
        cam_model = PinholeCameraModel()
        cam_model.fromCameraInfo(self.cam_info)
        vector = np.array(cam_model.projectPixelTo3dRay((u, v)))
        return vector


class PoseEstimator:
    def __init__(self):
        self.pca = PCA(n_components=3)
        self.ss = StandardScaler()

    def get_orientation(self, depth, mask) -> Quaternion:
        """マスクに重なったデプスからインスタンスの姿勢を算出"""
        # ここの値あってるか要検証...
        pts = [(x, y, depth[y, x]) for y, x in zip(*np.where(mask > 0))]
        self.pca.fit(self.ss.fit_transform(pts))
        n, t, b = self.pca.components_
        rmat_44 = np.eye(4)
        rmat_33 = np.dstack([n, t, b])[0]
        rmat_44[:3, :3] = rmat_33
        # 4x4回転行列しか受け入れない罠
        q = quaternion_from_matrix(rmat_44)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


# ref: https://qiita.com/kotarouetake/items/3c467e3c8aee0c51a50f
def numpy2multiarray(multiarray_type, np_array):
    """Convert numpy.ndarray to multiarray"""
    multiarray = multiarray_type()
    multiarray.layout.dim = [MultiArrayDimension(
        "dim%d" % i, np_array.shape[i], np_array.shape[i] * np_array.dtype.itemsize)
        for i in range(np_array.ndim)]
    multiarray.data = np_array.reshape(1, -1)[0].tolist()
    return multiarray


def multiarray2numpy(pytype, dtype, multiarray):
    """Convert multiarray to numpy.ndarray"""
    dims = [x.size for x in multiarray.layout.dim]
    res = np.array(multiarray.data, dtype=pytype).reshape(dims).astype(dtype)
    return res


def bboxmsg2list(msg: RotatedBoundingBox):
    return np.int0([msg.upper_left, msg.upper_right, msg.lower_right, msg.lower_left])


def candidatemsg2list(msg: Candidate):
    p1 = (msg.p1_u, msg.p1_v)
    p2 = (msg.p2_u, msg.p2_v)
    return (p1, p2)
