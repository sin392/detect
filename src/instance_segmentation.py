#!/usr/bin/env python3
from os.path import expanduser
from pathlib import Path
from typing import Union

import rospy
from cv_bridge import CvBridge
from detect.msg import BoundingBox, Instance, InstancesStamped
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer
from geometry_msgs.msg import Point
from inference import Predictor
from sensor_msgs.msg import Image
from std_msgs.msg import Header


def callback(msg: Image, callback_args: Union[list, tuple]):
    instances_publisher: rospy.Publisher = callback_args[0]
    seg_publisher: rospy.Publisher = callback_args[1]
    bridge = CvBridge()

    try:
        rospy.loginfo(msg.header)
        img = bridge.imgmsg_to_cv2(msg)
        v = Visualizer(
            img[:, :, ::-1],
            metadata={},
            scale=0.5,
            # remove the colors of unsegmented pixels.
            # This option is only available for segmentation models
            instance_mode=ColorMode.IMAGE_BW
        )
        outputs = predictor.predict(img)
        parsed_outputs = outputs.parse()
        num_instances = parsed_outputs["num_instances"]

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        res_img = out.get_image()[:, :, ::-1]
        res_img_msg = bridge.cv2_to_imgmsg(res_img, "rgb8")

        # outputs info publish
        instances = []
        for i in range(num_instances):
            instances.append(
                Instance(
                    label=str(parsed_outputs["labels"][i]),
                    score=parsed_outputs["scores"][i],
                    bbox=BoundingBox(*parsed_outputs["bboxes"][i]),
                    center=Point(*parsed_outputs["centers"][i], None),
                    area=parsed_outputs["areas"][i],
                    mask=bridge.cv2_to_imgmsg(
                        parsed_outputs["mask_array"][i].astype("int8"))
                )
            )

        header = Header(stamp=rospy.get_rostime(),
                        frame_id=msg.header.frame_id)
        instances_publisher.publish(
            InstancesStamped(
                header=header,
                num_instances=num_instances,
                instances=instances,
            ))
        res_img_msg.header = header
        seg_publisher.publish(res_img_msg)

    except Exception as err:
        rospy.logerr(err)


if __name__ == "__main__":
    rospy.init_node("instance_segmentaion_node", log_level=rospy.INFO)

    user_dir = expanduser("~")
    p = Path(f"{user_dir}/catkin_ws/src/detect")

    config_path = rospy.get_param(
        "config", str(p.joinpath("resources/configs/config.yaml")))
    weight_path = rospy.get_param("weight", str(
        p.joinpath("outputs/2022_08_04_07_40/model_final.pth")))
    device = rospy.get_param("device", "cuda:0")
    image_topics = rospy.get_param(
        "image_topic", "/body_camera/color/image_raw")

    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.DEVICE = device
    predictor = Predictor(cfg)

    for image_topic in image_topics.split():
        instances_publisher = rospy.Publisher(
            image_topic + "/instances", InstancesStamped, queue_size=10)
        seg_publisher = rospy.Publisher(
            image_topic + "/seg", Image, queue_size=10)
        rospy.loginfo(f"sub: {image_topic}")
        rospy.Subscriber(image_topic, Image, callback=callback,
                         callback_args=(
                             instances_publisher,
                             seg_publisher
                         ))

    rospy.spin()
