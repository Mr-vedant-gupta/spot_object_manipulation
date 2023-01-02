# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

import sys
import time
import cv2
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.api import geometry_pb2
from bosdyn.api import manipulation_api_pb2
from bosdyn.client import frame_helpers
from helpers.constrained_manipulation_helper import *
from helpers.vision_helpers import *
from helpers.time_consistency_helpers import *
from helpers.movement_helpers import *
from helpers.object_specific_helpers import *

from fetch_helper import *

VELOCITY = 0.5
FORCE_LIMIT = 50

class FetchModel:

    def __init__(self, robot, vision_model, robot_state_client, robot_command_client, manipulation_api_client):
        self.initialize_sdk()
        self.robot = robot
        self.vision_model = vision_model
        self.robot_state_client = robot_state_client
        self.robot_command_client = robot_command_client
        self.manipulation_api_client = manipulation_api_client

    def get_center_of_image(self, dogtoy, vision_tform_dogtoy):
        # Walk to the object.
        # NOTE: compare object we want to go to, vs object we see
        walk_rt_vision, heading_rt_vision = compute_stand_location_and_yaw(
            vision_tform_dogtoy, self.robot_state_client, distance_margin=1.5)

        move_cmd = RobotCommandBuilder.trajectory_command(
            goal_x=walk_rt_vision[0],
            goal_y=walk_rt_vision[1],
            goal_heading=heading_rt_vision,
            frame_name=frame_helpers.VISION_FRAME_NAME,
            params=get_walking_params(0.5, 0.5))
        end_time = 5.0
        cmd_id = self.robot_command_client.robot_command(command=move_cmd,
                                              end_time_secs=time.time() +
                                              end_time)

        # Wait until the robot reports that it is at the goal.
        block_for_trajectory_cmd(self.robot_command_client, cmd_id, timeout_sec=5, verbose=True)

        # The ML result is a bounding box.  Find the center.
        return self.vision_model.find_center_px(dogtoy.image_properties.coordinates)

    def run_fetch(self, label, seed_tform_obj):
        # This script assumes the robot is already standing via the tablet.  We'll take over from the
        # tablet.
        #self.lease_client.take()


        #with bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True):

        grasp_completed = False
        while not grasp_completed:

            # Capture an image and run ML on it.
            dogtoy, image, vision_tform_dogtoy, source = self.vision_model.get_object_and_image(label)

            #TODO: convert vision_tform_dogoty to global frame and compare it to seed_tform_obj
            if dogtoy is None:
                # Didn't find anything, keep searching.
                continue

            # Detected Object. Request pick up.

            # Stow the arm in case it is deployed
            stow_cmd = RobotCommandBuilder.arm_stow_command()
            self.robot_command_client.robot_command(stow_cmd)


            (center_px_x,
             center_px_y) = self.get_center_of_image(dogtoy, vision_tform_dogtoy)

            
            if label =="door_handle":
                print("DOOR HANDLE")
                execute_open_door(self.robot,self.vision_model.image_sources, source, center_px_x, center_px_y, image)
            if label =="handle":
                print("DRAWER")
                #execute_open_drawer(self.robot, source, center_px_x, center_px_y, image)
                # Request Pick Up on that pixel.
                pick_vec = geometry_pb2.Vec2(x=center_px_x, y=center_px_y)
                grasp = manipulation_api_pb2.PickObjectInImage(
                    pixel_xy=pick_vec,
                    transforms_snapshot_for_camera=image.shot.transforms_snapshot,
                    frame_name_image_sensor=image.shot.frame_name_image_sensor,
                    camera_model=image.source.pinhole)

                # We can specify where in the gripper we want to grasp. About halfway is generally good for
                # small objects like this. For a bigger object like a shoe, 0 is better (use the entire
                # gripper)
                grasp.grasp_params.grasp_palm_to_fingertip = 0.6

                # Tell the grasping system that we want a top-down grasp.

                # Add a constraint that requests that the x-axis of the gripper is pointing in the
                # negative-z direction in the vision frame.

                # The axis on the gripper is the x-axis.
                
                axis_on_gripper_ewrt_gripper, axis_to_align_with_ewrt_vision = grasp_directions(label)

                # The axis in the vision frame is the negative z-axis
                

                # Add the vector constraint to our proto.
                constraint = grasp.grasp_params.allowable_orientation.add()
                constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
                    axis_on_gripper_ewrt_gripper)
                constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
                    axis_to_align_with_ewrt_vision)

                # We'll take anything within about 15 degrees for top-down or horizontal grasps.
                constraint.vector_alignment_with_tolerance.threshold_radians = 0.25

                # Specify the frame we're using.
                grasp.grasp_params.grasp_params_frame_name = frame_helpers.VISION_FRAME_NAME

                # Build the proto
                grasp_request = manipulation_api_pb2.ManipulationApiRequest(
                    pick_object_in_image=grasp)

                # Send the request
                print('Sending grasp request...')
                cmd_response = self.manipulation_api_client.manipulation_api_command(
                    manipulation_api_request=grasp_request)

                # Wait for the grasp to finish
                grasp_done = False
                failed = False
                time_start = time.time()

                while not grasp_done:
                    feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                        manipulation_cmd_id=cmd_response.manipulation_cmd_id)

                    # Send a request for feedback
                    response = self.manipulation_api_client.manipulation_api_feedback_command(
                        manipulation_api_feedback_request=feedback_request)

                    current_state = response.current_state
                    current_time = time.time() - time_start
                    # if current_time > 20:
                    #     failed = False
                    #     break
                    print('Current state ({time:.1f} sec): {state}'.format(
                        time=current_time,
                        state=manipulation_api_pb2.ManipulationFeedbackState.Name(
                            current_state)),
                          end='                \r')
                    sys.stdout.flush()

                    failed_states = [manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
                                     manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION,
                                     manipulation_api_pb2.MANIP_STATE_GRASP_FAILED_TO_RAYCAST_INTO_MAP,
                                     manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_WAITING_DATA_AT_EDGE]

                    failed = current_state in failed_states
                    grasp_done = current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or failed

                    time.sleep(0.1)

                grasp_completed = not failed
                print(grasp_completed)
                if not grasp_completed:
                    print("Didnt find object")
                    continue

                # Move the arm to a carry position.
                print('Grasp finished, search for a person...')
            
                #carry_cmd = RobotCommandBuilder.arm_carry_command()
                command = construct_drawer_task(-VELOCITY, force_limit=FORCE_LIMIT)
                # command_client.robot_command(command)
                command.full_body_command.constrained_manipulation_request.end_time.CopyFrom(
                    self.robot.time_sync.robot_timestamp_from_local_secs(time.time() + 10))
                self.robot_command_client.robot_command_async(command)

                # Wait for the carry command to finish
                print("Finished action 1")
                time.sleep(2)

                command = construct_drawer_task(VELOCITY, force_limit=FORCE_LIMIT)
                # command_client.robot_command(command)
                command.full_body_command.constrained_manipulation_request.end_time.CopyFrom(
                    self.robot.time_sync.robot_timestamp_from_local_secs(time.time() + 10))
                self.robot_command_client.robot_command_async(command)

                print("Finished action 2")
                time.sleep(2)

def main(argv):

    cv2.namedWindow("FetchModel")
    cv2.waitKey(500)

    fetch = FetchModel()
    fetch.run_fetch(None,"drawer")

if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
