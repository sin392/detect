from cv_bridge import CvBridge
from detect.msg import DetectedObject, DetectedObjectsStamped, InstancesStamped
from geometry_msgs.msg import PointStamped, PoseStamped
from rospy import Publisher, Time
from sensor_msgs.msg import Image
from std_msgs.msg import Header

bridge = CvBridge()


class InstancesPublisher(Publisher):
    def __init__(self, name, subscriber_listener=None, tcp_nodelay=False, latch=False, headers=None, queue_size=None):
        super().__init__(name, InstancesStamped, subscriber_listener,
                         tcp_nodelay, latch, headers, queue_size)

    def publish(self, num_instances, instances, frame_id: str, stamp: Time):
        msg = InstancesStamped(
            header=Header(frame_id=frame_id, stamp=stamp),
            num_instances=num_instances,
            instances=instances,
        )
        super().publish(msg)


class ImageMatPublisher(Publisher):
    global bridge

    def __init__(self, name, subscriber_listener=None, tcp_nodelay=False, latch=False, headers=None, queue_size=None):
        super().__init__(name, Image, subscriber_listener,
                         tcp_nodelay, latch, headers, queue_size)

    def publish(self, img_mat, frame_id: str, stamp: Time):
        msg = bridge.cv2_to_imgmsg(img_mat, "rgb8")
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp
        super().publish(msg)


class DetectedObjectsPublisher(Publisher):
    def __init__(self, name, subscriber_listener=None, tcp_nodelay=False, latch=False, headers=None, queue_size=None):
        super().__init__(name, DetectedObjectsStamped,
                         subscriber_listener, tcp_nodelay, latch, headers, queue_size)
        self.clear_stack()

    def push_item(self, p1: PointStamped, p2: PointStamped, center: PoseStamped, short_radius: float, long_radius: float):
        msg = DetectedObject(
            p1=p1,
            p2=p2,
            center=center,
            short_radius=short_radius,
            long_radius=long_radius
        )
        self.stack.append(msg)

    def publish_stack(self, frame_id: str, stamp: Time):
        msg = DetectedObjectsStamped(objects=self.stack)
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp
        super().publish(msg)
        self.clear_stack()

    def clear_stack(self):
        self.stack = []
