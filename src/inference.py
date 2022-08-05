import cv2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


class OutputsDict(dict):
    def parse(self):
        instances = self['instances']
        mask_array = instances.pred_masks.to("cpu").numpy()
        num_instances = mask_array.shape[0]
        scores = instances.scores.to("cpu").numpy()
        labels = instances.pred_classes.to("cpu").numpy()
        bboxes = instances.pred_boxes.to("cpu").tensor.numpy()
        centers = instances.pred_boxes.get_centers().to("cpu").numpy()
        areas = instances.pred_boxes.area().to("cpu").numpy()

        return {"num_instances": num_instances, "mask_array": mask_array,
                "scores": scores, "labels": labels, "bboxes": bboxes,
                "centers": centers, "areas": areas}


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
    args = parser.parse_args()

    p = Path("./")
    dataset_path = p.joinpath("datasets")

    cfg = get_cfg()
    cfg.merge_from_file(str(p.joinpath("configs", "config.yaml")))
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHTS = args.weight or f"{cfg.OUTPUT_DIR}/2022_08_04_07_40/model_final.pth"

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
        # cv2.imshow("result", out.get_image()[:, :, ::-1])
        fname = Path(d["file_name"]).stem + ".png"
        save_path = str(p.joinpath("output", "images", fname))
        # print(outputs["instances"])
        print(f"{end - start:.4f}", save_path)
        # print(outputs['instances'].pred_boxes.get_centers())
        # print(outputs['instances'].pred_boxes.area())
        cv2.imwrite(
            save_path,
            out.get_image()[:, :, ::-1])
