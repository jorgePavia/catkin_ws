import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class image_converter:
    def __init__(self):
        self.image_pub = rospy.Publisher("image_topic",Image,queue_size=1) 
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camara_pub_img",Image,self.callback) 
    
    def callback(self, data):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 100, 20], np.uint8)
        upper_red = np.array([8, 255, 255], np.uint8)
        
        mask_r = cv2.inRange(hsv, lower_red, upper_red) 
        res_r = cv2.bitwise_and(cv_image,cv_image, mask= mask_r)  

        cv2.imshow('Mask',res_r) 
        cv2.imshow('Original',cv_image) 
        cv2.waitKey(1) & 0xFF == ord('q')

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)

def main(args):
    #--- Create the object from the class we defined before
    ic = image_converter()
    
    #--- Initialize the ROS node
    rospy.init_node('teVeo_pub_img', anonymous=True) 
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        
    #--- In the end remember to close all cv windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
        main(sys.argv)