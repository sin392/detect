import os

import cv2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

from scripts.entities.predictor import Predictor

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

        out = outputs.draw_instances(img[:, :, ::-1])
        fname = Path(d["file_name"]).stem + ".png"
        print(f"{end - start:.4f}", fname)

        if args.save:
            cv2.imwrite(
                f"{save_dir}/{fname}",
                out.get_image()[:, :, ::-1])
