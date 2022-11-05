# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

import argparse
import sys
import time
import numpy as np
import cv2
import math
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, block_until_arm_arrives
from bosdyn.api import geometry_pb2
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.api import image_pb2
from bosdyn.client.network_compute_bridge_client import NetworkComputeBridgeClient
from bosdyn.api import network_compute_bridge_pb2
from google.protobuf import wrappers_pb2
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.api import manipulation_api_pb2
from bosdyn.api import basic_command_pb2
from bosdyn.client import frame_helpers
from bosdyn.client import math_helpers

from helpers.constrained_manipulation_helper import *
from helpers.vision_helpers import *
from helpers.time_consistency_helpers import *
from helpers.movement_helpers import *
from helpers.object_specific_helpers import *
from arm_door import *


import argparse
import math
import sys
import time

import cv2
import numpy as np

from bosdyn import geometry
from bosdyn.api import basic_command_pb2, geometry_pb2, manipulation_api_pb2
from bosdyn.api.manipulation_api_pb2 import (ManipulationApiFeedbackRequest, ManipulationApiRequest,
                                             WalkToObjectInImage)
from bosdyn.api.spot import door_pb2
from bosdyn.client import create_standard_sdk, frame_helpers
from bosdyn.client.door import DoorClient
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.util import add_base_arguments, authenticate, setup_logging

def main(argv):
    #create parser
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument(
        '-s',
        '--ml-service',
        help='Service name of external machine learning server.',
        required=True)
    parser.add_argument('-m',
                        '--model',
                        help='Model name running on the external server.',
                        required=True)
    parser.add_argument(
        '-l',
        '--label',
        help='name of label we want Spot to manipulate, should be one of handle, ...',
        required=True)
    parser.add_argument('-c',
                        '--confidence-dogtoy',
                        help='Minimum confidence to return an object for the dogoy (0.0 to 1.0)',
                        default=0.5,
                        type=float)
    options = parser.parse_args(argv)

    cv2.namedWindow(options.label)
    cv2.waitKey(500)

    sdk = bosdyn.client.create_standard_sdk('SpotFetchClient')
    sdk.register_service_client(NetworkComputeBridgeClient)
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)

    # Time sync is necessary so that time-based filter requests can be converted
    robot.time_sync.wait_for_sync()

    network_compute_client = robot.ensure_client(
        NetworkComputeBridgeClient.default_service_name)
    robot_state_client = robot.ensure_client(
        RobotStateClient.default_service_name)
    command_client = robot.ensure_client(
        RobotCommandClient.default_service_name)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    manipulation_api_client = robot.ensure_client(
        ManipulationApiClient.default_service_name)

    # This script assumes the robot is already standing via the tablet.  We'll take over from the
    # tablet.
    lease_client.take()
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Store the position of the hand at the last toy drop point.
         vision_tform_hand_at_drop = None

        # while True:
         grasp_completed = False
         while not grasp_completed:
            print(options.label)
            # Capture an image and run ML on it.
            dogtoy, image, vision_tform_dogtoy, source = get_obj_and_img(
                network_compute_client, options.ml_service, options.model,
                options.confidence_dogtoy, kImageSources, options.label)
            if dogtoy is None:
                # Didn't find anything, keep searching.
                continue

            # Detected Object. Request pick up.

            # Stow the arm in case it is deployed
            stow_cmd = RobotCommandBuilder.arm_stow_command()
            command_client.robot_command(stow_cmd)


            # Walk to the object.
            walk_rt_vision, heading_rt_vision = compute_stand_location_and_yaw(
                vision_tform_dogtoy, robot_state_client, distance_margin=1.0)

            move_cmd = RobotCommandBuilder.trajectory_command(
                goal_x=walk_rt_vision[0],
                goal_y=walk_rt_vision[1],
                goal_heading=heading_rt_vision,
                frame_name=frame_helpers.VISION_FRAME_NAME,
                params=get_walking_params(0.5, 0.5))
            end_time = 5.0
            cmd_id = command_client.robot_command(command=move_cmd,
                                                  end_time_secs=time.time() +
                                                  end_time)

            # Wait until the robot reports that it is at the goal.
            block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=5, verbose=True)

            # The ML result is a bounding box.  Find the center.
            (center_px_x,
             center_px_y) = find_center_px(dogtoy.image_properties.coordinates)
            execute_open_door(robot, source, center_px_x, center_px_y, image)










if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)


