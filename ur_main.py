#!/usr/bin/env python
import rospy
from ur_class import UR_Ctrl
#urx_control.chdir("~/catkin_ws/src/gripper/scripts")
from talker import Gripper
#talker.chdir("~/catkin_ws/src/gripper/scripts")

def main():
	rospy.init_node('robot_main_node', anonymous=True)
	gripper = Gripper()
	ur_robot = UR_Ctrl(gripper)
	i=0
	j=0
	print("start")
	ur_robot.start_ctrl()
	flag=0

	while i<2 :
	 	while j<2 : 
	 		ur_robot.start()

	 		if flag == 0:
	 			ur_robot.sensorcheck()

	 		ur_robot.movetobottle()
	 		ur_robot.rotatewirst(90)
	 		ur_robot.pickupbottle()
	 		ur_robot.movetobox()
	 		if flag==0 :
	 			ur_robot.findBoxPosition()
	 			ur_robot.firstBoxLinePosition()
	 			flag=1
	 		else :
	 			ur_robot.secondBoxLinePosition()

	 		ur_robot.putdownbottle()
		
	 		j=j+1
	 	i=i+1
	 	j=0
	 	flag = 0
	 	ur_robot.moveconv()
	print('end')
	ur_robot.end_ctrl()
if __name__ == '__main__':
	main()