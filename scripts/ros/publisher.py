from cv_bridge import CvBridge
from detect.msg import InstancesStamped
from rospy import Publisher
from sensor_msgs.msg import Image
from std_msgs.msg import Header

bridge = CvBridge()


class InstancesPublisher(Publisher):
    def __init__(self, name, subscriber_listener=None, tcp_nodelay=False, latch=False, headers=None, queue_size=None):
        super().__init__(name, InstancesStamped, subscriber_listener,
                         tcp_nodelay, latch, headers, queue_size)

    def publish(self, num_instances, instances, header=Header()):
        msg = InstancesStamped(
            header=header,
            num_instances=num_instances,
            instances=instances,
        )
        super().publish(msg)


class ImageMatPublisher(Publisher):
    global bridge

    def __init__(self, name, subscriber_listener=None, tcp_nodelay=False, latch=False, headers=None, queue_size=None):
        super().__init__(name, Image, subscriber_listener,
                         tcp_nodelay, latch, headers, queue_size)

    def publish(self, img_mat, header=Header()):
        msg = bridge.cv2_to_imgmsg(img_mat, "rgb8")
        msg.header = header
        super().publish(msg)
