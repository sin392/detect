import os
from typing import Tuple, Union

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from torch import Tensor


class BinaryMask:
    def __init__(self, mask: Union[np.ndarray, Tensor]):
        self.mask = np.asarray(mask, dtype=np.uint8)
        self.contour = None

    # private
    def __get_contour(self):
        contours, _ = cv2.findContours(
            self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            contour = max(contours, key=lambda x: cv2.contourArea(x))
        else:
            contour = contours[0]

        return contour

    def get_center(self):
        if not self.contour:
            self.contour = self.__get_contour()
        mu = cv2.moments(self.contour)
        center = np.array([int(mu["m10"]/mu["m00"]), int(mu["m01"]/mu["m00"])])
        return np.int0(center)

    def get_rotated_bbox(self):
        if not self.contour:
            self.contour = self.__get_contour()
        # (upper_left, upper_right, lower_right, lower_left)
        return np.int0(cv2.boxPoints(cv2.minAreaRect(self.contour)))


# このクラス必要か？
class OutputsDict(dict):
    def parse(self):
        instances = self['instances']
        mask_array = instances.pred_masks.to("cpu").numpy().astype("int8")
        num_instances = mask_array.shape[0]
        scores = instances.scores.to("cpu").numpy()
        labels = instances.pred_classes.to("cpu").numpy()
        areas = instances.pred_masks.area().to("cpu").numpy()
        # centers = instances.pred_boxes.get_centers().to("cpu").numpy()
        # bboxes = instances.pred_boxes.to("cpu").tensor.numpy()
        centers, bboxes = self.__get_centers_and_bboxes_from_mask(mask_array)

        return {"num_instances": num_instances, "mask_array": mask_array,
                "scores": scores, "labels": labels, "bboxes": bboxes,
                "centers": centers, "areas": areas}

    @staticmethod
    def __get_centers_and_bboxes_from_mask(mask_array: np.ndarray):
        centers = []
        rotated_bboxes = []
        for each_mask_array in mask_array:
            each_mask = BinaryMask(each_mask_array)
            centers.append(each_mask.get_center())
            rotated_bboxes.append(each_mask.get_rotated_bbox())
        return centers, rotated_bboxes


class Predictor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.predictor = DefaultPredictor(self.cfg)

    def predict(self, img):
        outputs = OutputsDict(self.predictor(img))
        return outputs


if __name__ == "__main__":
    from argparse import ArgumentParser
    from pathlib import Path
    from time import time

    parser = ArgumentParser()
    parser.add_argument("--weight", "-w")
    parser.add_argument("--device", "-d", default="cuda:0")
    parser.add_argument("--save", action='store_false')
    args = parser.parse_args()

    user_dir = os.path.expanduser("~")
    p = Path(f"{user_dir}/catkin_ws/src/detect")
    dataset_path = p.joinpath("resources/datasets")

    cfg = get_cfg()
    cfg.merge_from_file(p.joinpath("resources/configs/config.yaml").as_posix())
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = args.weight or p.joinpath(
        "outputs/2022_08_04_07_40/model_final.pth").as_posix()

    dataset_name = "cabbage_val"
    register_coco_instances(dataset_name,
                            {"thing_classes": ["cabbage"]},
                            dataset_path.joinpath("val/COCO_Cabbage_val.json"),
                            dataset_path.joinpath("val/images"))
    metadata = MetadataCatalog.get(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)

    print("device :", cfg.MODEL.DEVICE)
    print("weight :", cfg.MODEL.WEIGHTS)
    print("-" * 40)

    predictor = Predictor(cfg)

    save_dir = p.joinpath("outputs/images").as_posix()
    os.makedirs(save_dir, exist_ok=True)
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        start = time()
        outputs = predictor.predict(img)
        end = time()

        v = Visualizer(
            img[:, :, ::-1],
            metadata=metadata,
            scale=0.5,
            # remove the colors of unsegmented pixels.
            # This option is only available for segmentation models
            instance_mode=ColorMode.IMAGE_BW
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        fname = Path(d["file_name"]).stem + ".png"
        print(f"{end - start:.4f}", fname)

        if args.save:
            cv2.imwrite(
                f"{save_dir}/{fname}",
                out.get_image()[:, :, ::-1])
