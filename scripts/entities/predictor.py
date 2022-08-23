from typing import Any, Tuple, TypedDict

import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes
from detectron2.structures import Instances as RawInstances
from detectron2.utils.visualizer import ColorMode, Visualizer
from torch import Tensor

from entities.image import BinaryMask


class Instances(RawInstances):
    """predict完了して各値がセットされた後のInstances(type指定用)"""

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        super().__init__(image_size, **kwargs)
        self.boxes: Boxes
        self.pred_masks: Tensor
        self.scores: Tensor
        self.pred_classes: Tensor

    # 戻り値の型を上書きするためにオーバーライド
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        super().to(*args, **kwargs)


class OutputsDictType(TypedDict):
    instances: Instances


class PredictResult:
    """detecton2のpredictの出力を扱いやすい形にパースする"""

    def __init__(self, outputs_dict: OutputsDictType, device: str = "cpu"):
        self._instances: Instances = outputs_dict['instances'].to(device)

        self.mask_array: np.ndarray = self._instances.pred_masks.numpy().astype("uint8")
        self.scores: np.ndarray = self._instances.scores.numpy()
        self.labels: np.ndarray = self._instances.pred_classes.numpy()

        self.num_instances: int = self.mask_array.shape[0]

        # rotated_bboxの形式は(center, weight, height, angle)の方がよい？
        # radiusも返すべき？
        # contourはどうやってｍｓｇに渡す？
        self.contours = []
        self.centers = []
        self.bboxes = []
        self.areas = []
        for each_mask_array in self.mask_array:
            each_mask = BinaryMask(each_mask_array)
            self.contours.append(each_mask.contour)
            self.centers.append(each_mask.get_center())
            self.bboxes.append(each_mask.get_rotated_bbox())
            self.areas.append(each_mask.get_area())

    def draw_instances(self, img, metadata={}, scale=0.5, instance_mode=ColorMode.IMAGE_BW):
        v = Visualizer(
            img,
            metadata=metadata,
            scale=scale,
            # remove the colors of unsegmented pixels.
            # This option is only available for segmentation models
            instance_mode=instance_mode
        )
        return v.draw_instance_predictions(self._instances).get_image()[:, :, ::-1]


# これは別の場所に置くべきでは
class Predictor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, img):
        outputs = self.predictor(img)
        parsed_outputs = PredictResult(outputs)
        return parsed_outputs
