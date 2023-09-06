from asyncore import write
import imp
from multiprocessing.connection import wait
from rclpy.node import Node
from geometry_msgs.msg import Twist
import rclpy
from rclpy.qos import QoSProfile
import math
from localization import Localization
import numpy as np
import utm
import csv
import time

class Motion(Node):
	robot = None
	
	def __init__(self):
		super().__init__('motion')
		self._publisher = self.create_publisher(Twist, '/rosbot2/cmd_vel', QoSProfile(depth=10))
		self.robot=Localization()
	

		while not (self.robot.odom and self.robot.yawdeg and self.robot.gnss):			
			print("waiting")
			rclpy.spin_once(self.robot)
		

		self.robot.reset()
		rclpy.spin_once(self.robot)

		print("okker")
	
	def create_twist(self, lin_x=0.0, lin_y=0.0, lin_z=0.0, ang_x=0.0, ang_y=0.0, ang_z=0.0):
		twist = Twist()
		twist.linear.x = lin_x
		twist.linear.y = lin_y
		twist.linear.z = lin_z
		twist.angular.x = ang_x
		twist.angular.y = ang_y
		twist.angular.z = ang_z
		return twist
	
	def stop_twist(self):
		twist = self.create_twist()
		while True:
			self._publisher.publish(twist)
	
def main():
	rclpy.init(args=None)
	rosbot=Motion()
	rclpy.shutdown()
   
if __name__ == '__main__':
    main()