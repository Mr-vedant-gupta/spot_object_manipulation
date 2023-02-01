# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Command line interface for graph nav with options to download/upload a map and to navigate a map. """

import math
import os
import sys
import copy
import time
import random
from bosdyn.client import math_helpers

import graph_nav_util
import pickle
from threading import Thread
import bosdyn.client.channel
import bosdyn.client.util
from vision_model import VisionModel
from fetch_model import FetchModel
from bosdyn.client.network_compute_bridge_client import NetworkComputeBridgeClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.api import robot_state_pb2
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.client.exceptions import ResponseError
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.power import PowerClient, power_on, safe_power_off
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand, block_until_arm_arrives
from bosdyn.geometry import EulerZXY
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.api import world_object_pb2
from bosdyn.client.world_object import WorldObjectClient
from navigation import estop_gui
from skills.spot_manipulation_skills import execute_trajectory_from_poses_oo, close_lid, look_around, execute_trajectory_from_poses
from collections import defaultdict

import numpy as np

# HOSTNAME = "138.16.161.22"         #02
HOSTNAME = "192.168.80.3"           #12
# UPLOAD_FILEPATH = "./navigation/maps/downloaded_graph"
#UPLOAD_FILEPATH = "./navigation/maps/cit121_02/downloaded_graph"
# UPLOAD_FILEPATH = "./navigation/maps/cit121_115_02/downloaded_graph"
#UPLOAD_FILEPATH = "./navigation/maps/cit121_115_12/downloaded_graph"
# UPLOAD_FILEPATH = "./navigation/maps/cit121_12/downloaded_graph"
UPLOAD_FILEPATH = "./navigation/maps/max12/downloaded_graph"

# UPLOAD_FILEPATH = "/home/vedantgupta/drawer/navigation/maps/downloaded_graph"
NAVIGATION_TO_OBJECT_ACCEPTABLE_DISTANCE = 3.0


class GraphNavInterface(object):
    """GraphNav service command line interface."""

    def __init__(self, robot, upload_path):
        self._robot = robot

        # Force trigger timesync.
        self._robot.time_sync.wait_for_sync()

        # Create robot state and command clients.
        self._robot_command_client = self._robot.ensure_client(
            RobotCommandClient.default_service_name)

        self._robot_state_client = self._robot.ensure_client(RobotStateClient.default_service_name)

        # Create the client for the Graph Nav main service.
        self._graph_nav_client = self._robot.ensure_client(GraphNavClient.default_service_name)

        # Create a power client for the robot.
        self._power_client = self._robot.ensure_client(PowerClient.default_service_name)

        self._network_compute_client = self._robot.ensure_client(NetworkComputeBridgeClient.default_service_name)

        self._manipulation_api_client = self._robot.ensure_client(ManipulationApiClient.default_service_name)

        self.vision_model = VisionModel(self._graph_nav_client, self._network_compute_client, self._robot)
        self.fetch_model = FetchModel(self._robot, self.vision_model, self._robot_state_client,
                                      self._robot_command_client, self._manipulation_api_client)

        # Boolean indicating the robot's power state.
        power_state = self._robot_state_client.get_robot_state().power_state
        self._started_powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        self._powered_on = self._started_powered_on

        # Number of attempts to wait before trying to re-power on.
        self._max_attempts_to_wait = 50

        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  # maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()
        self._object_fiducials_dict = {'coffee_pot': 'filtered_fiducial_523',
                                       'cupboard': 'filtered_fiducial_534',
                                       'table': 'filtered_fiducial_529',
                                       'drawers': 'filtered_fiducial_535'
                                       }
        self._skill_offset_dict = {'pour_grinds': {'coffee_pot': (0, 0, 0.8)},
                                   'push_button': {'coffee_pot': (0, 0, 0.8)},
                                   'pour_water': {'coffee_pot': (0, 0, 0.8)},
                                   'close_lid': {'coffee_pot': (0, -0.6, 0.1)},
                                   'go_to': {'coffee_pot': (0, 0, 0.7),
                                             'cupboard': (0, 0, .9),
                                             'table': (0, 0, 0.8),
                                             'drawers': (0, 0, 0.8),
                                             }
                                   }

        # Filepath for uploading a saved graph's and snapshots too.
        if upload_path[-1] == "/":
            self._upload_filepath = upload_path[:-1]
        else:
            self._upload_filepath = upload_path

        self._command_dictionary = {
            '1': self._get_localization_state,
            '2': self._set_initial_localization_fiducial,
            '3': self._set_initial_localization_waypoint,
            '4': self._list_graph_waypoint_and_edge_ids,
            '5': self._upload_graph_and_snapshots,
            '6': self._navigate_to,
            '7': self._navigate_route,
            '8': self._navigate_to_anchor,
            '9': self._clear_graph,
            '10': self._navigate_all,
            '11': self._list_objects,
            '12': self._navigate_to_object,
            '13': self._manipulate_object,
            '14': self._upload_clusters,
            '15': self._blocking_stand,
            '16': self._pour_grinds,
            '17': self._pour_water,
            '18': self._close_lid,
            '19': self._navigate_all_fiducials,
            '20': self._goto_locale,
            '21': self._pick_cup,
            '22': self._push_button,
            '23': self._open_gripper_and_stow,
            '24': self._place_cup_left,
            '25': self._place_cup_right,
            '26': self._execute_plan,
            '27': self._cam_search,
            '28': self._find_where_objects_are,
            '29': self._generate_aosm_and_plan,
            '30': self._open_noodle_door,
            '31': self._collect_aosm_data,
            '32': self._run_LOMDP

        }


    def _run_LOMDP(self, *args):
        sys.path.append('../Locality')
        from coffeeLomdp import runLOMDP_robot

        runLOMDP_robot(self)

    def _collect_aosm_data(self, *args):
        desription_list = [
            "robot_is_holding",
            "obj_left_of_coffeepot",
            "obj_right_of_coffeepot",
            "obj_on_drawers",
            "obj_on_table",
            "obj_on_cupboard",
            "coffee_in_cup",
            "water_in_cup",
            "coffee_in_pot",
            "water_in_pot",
            "lid_open",
            "button_pushed",
            "coffee_made",
            "robot_loc"
        ]
        obj_idxs = [None, "coffee_cup","water_cup"]
        location_idxs = ["robot_is_holding","obj_left_of_coffeepot","obj_right_of_coffeepot","obj_on_drawers","obj_on_table","obj_on_cupboard"]
        locales = list(self._object_fiducials_dict.keys())

        def generate_start_state():
            start_state = [0]*len(desription_list)

            #First make robot be at a location
            start_state[13] = random.choice(range(len(locales)))
            print(f"robot start loc is {locales[start_state[13]]}")

            #First fill in object locations
            #obj_locs[0] is coffee_cup location, obj_locs[1] is water_cup location
            obj_locs = random.sample(location_idxs[1:],2)
            print(f"Coffee cup and water cup are at: {obj_locs}")
            obj_loc_idx_coffee_cup = location_idxs.index(obj_locs[0])
            start_state[obj_loc_idx_coffee_cup] = 1
            obj_loc_idx_water_cup = location_idxs.index(obj_locs[1])
            start_state[obj_loc_idx_water_cup] = 2

            #Then fill in whether water and coffee are in pot
            start_state[6] = random.choice([0,1])
            start_state[7] = random.choice([0, 1])
            print(f"coffee in cup? {start_state[6]}")
            print(f"water in cup? {start_state[7]}")

            #Coffee pot status, opposite of cups by definition
            start_state[8] = int(not start_state[6])
            start_state[9] = int(not start_state[7])

            #lid always starts open, button always not pushed and coffee not made
            start_state[10] = 1
            start_state[11] = 0
            start_state[12] = 0

            print(f"start state: {start_state}")
            return(start_state)


        actions_to_try = [
            [self._goto_locale, ["cupboard"]],
            [self._goto_locale, ["drawers"]],
            [self._goto_locale, ["coffee_pot"]],
            [self._goto_locale, ["table"]],
            [self._pick_cup, ["water_cup"]],
            [self._pick_cup, ["coffee_cup"]],
            #[self._pour_grinds, ["coffee_pot"]],
            #[self._pour_water, ["coffee_pot"]],
            #[self._place_cup_left,["coffee_pot"]],
            #[self._place_cup_right,["coffee_pot"]],
            #[self._close_lid, ["coffee_pot"]],
            #[self._push_button, ["coffee_pot"]]
        ]

        self._upload_clusters()

        data_collection = [] #collection of data transitions that were feasible
        attempt_action = [] #list of states and whether an action was possible or not
        num_data = 100

        cur_state = generate_start_state()
        #move robot to start location
        self._goto_locale([locales[cur_state[13]]])

        for _ in range(num_data):
            try_action_idx = random.choice(range(len(actions_to_try)))
            try_action = actions_to_try[try_action_idx]
            input(f"Going to do action: {try_action}")
            start_state = copy.deepcopy(cur_state)
            print(f"start state {start_state}")
            if try_action[0] == self._goto_locale:
                #go to locale actions are always valid
                self._goto_locale([try_action[1][0]])
                #Robot now at new location
                cur_state[13] = locales.index(try_action[1][0])

                end_state = cur_state

                #check if end of episode (either for task success or just to reset)
                resp = input("Should take be reset?")
                if resp == "y":
                    done = 1
                    cur_state = generate_start_state()
                    # move robot to start location
                    self._goto_locale([locales[cur_state[13]]])
                else:
                    done = 0
                    cur_state = end_state

                print(f"end state {end_state}")


                attempt_action.append([start_state, 1])
                data_collection.append([start_state, try_action_idx, end_state])

            elif try_action[0] == self._pick_cup: #Trying pick cup action
                print("considering action pickcup")
                if start_state[0] == 0 and self._can_pick_cup(try_action[1]): #if robot not holding cup and cup is actually visible, great, try pick up
                    print("Executng pick up cup action")
                    self._pick_cup(try_action[1])
                    #object is no longer at place it was before
                    for idx, val in enumerate(cur_state[:7]):
                        if val == obj_idxs.index(try_action[1][0]): #this is where the item was previously, make it no longer there
                            cur_state[idx] = 0
                    cur_state[0] = obj_idxs.index(try_action[1][0]) #robot is now holding that cup
                    end_state = cur_state

                    #check if end of episode (either for task success or just to reset)
                    resp = input("Should take be reset?")
                    if resp == "y":
                        done = 1
                        cur_state = generate_start_state()
                        # move robot to start location
                        self._goto_locale([locales[cur_state[13]]])
                    else:
                        done = 0
                        cur_state = end_state

                    print(f"end state {end_state}")

                    attempt_action.append([start_state, 1])
                    data_collection.append([start_state, try_action_idx, end_state])
                else:
                    print("can not pick up cup :(")
                    attempt_action.append([start_state, 0])
            elif try_action[0] == self._pour_grinds: #Trying to pour grinds
                if start_state[13] == locales.index("coffee_pot") and start_state[0] == obj_idxs.index("coffee_cup") and start_state[6] == 1: #if robot is at coffeepot, holding coffee grinds, and coffee_cup has grinds
                    #TODO: execute pour coffee grinds skill for real

                    #After skill execution, coffee grounds is no longer true
                    cur_state[6] = 0 #no more grounds in coffee cup
                    cur_state[8] = 1 #coffee grinds in coffee machine!

                    end_state = cur_state

                    #check if end of episode (either for task success or just to reset)
                    resp = input("Should take be reset?")
                    if resp == "y":
                        done = 1
                        cur_state = generate_start_state()
                        # move robot to start location
                        self._goto_locale([locales[cur_state[13]]])
                    else:
                        done = 0
                        cur_state = end_state

                    print(f"end state {end_state}")

                    attempt_action.append([start_state, 1])
                    data_collection.append([start_state, try_action_idx, end_state])
                else:
                    print("can not pour coffee grinds :(")
                    attempt_action.append([start_state, 0])

            elif try_action[0] == self._pour_water: #Trying to pour water
                if start_state[13] == locales.index("coffee_pot") and start_state[0] == obj_idxs.index("water_cup") and start_state[7] == 1: #if robot is at coffeepot, holding water cup, and watercup has water
                    #TODO: execute water skill

                    #After skill execution, water cup is no longer true
                    cur_state[7] = 0 #no more water in cup
                    cur_state[9] = 1 #water in coffee machine!

                    end_state = cur_state

                    #check if end of episode (either for task success or just to reset)
                    resp = input("Should take be reset?")
                    if resp == "y":
                        done = 1
                        cur_state = generate_start_state()
                        # move robot to start location
                        self._goto_locale([locales[cur_state[13]]])
                    else:
                        done = 0
                        cur_state = end_state

                    print(f"end state {end_state}")

                    attempt_action.append([start_state, 1])
                    data_collection.append([start_state, try_action_idx, end_state])
                else:
                    print("can not pour water :(")
                    attempt_action.append([start_state, 0])
            elif try_action[0] == self._place_cup_left: #place cup
                if start_state[13] == locales.index("coffee_pot") and start_state[0] != 0 and start_state[1] == 0: #if robot is at coffeepot, holding something, and nothing at left spot
                    #TODO: place object

                    #After skill execution, cup is placed
                    cur_state[1] = cur_state[0] #whatever was in hand is now at left coffee pot location
                    cur_state[0] = 0 #not holding anything anymore

                    end_state = cur_state

                    #check if end of episode (either for task success or just to reset)
                    resp = input("Should take be reset?")
                    if resp == "y":
                        done = 1
                        cur_state = generate_start_state()
                        # move robot to start location
                        self._goto_locale([locales[cur_state[13]]])
                    else:
                        done = 0
                        cur_state = end_state

                    print(f"end state {end_state}")

                    attempt_action.append([start_state, 1])
                    data_collection.append([start_state, try_action_idx, end_state])
                else:
                    print("can not place cup on left side")
                    attempt_action.append([start_state, 0])

            elif try_action[0] == self._place_cup_right: #place cup
                if start_state[13] == locales.index("coffee_pot") and start_state[0] != 0 and start_state[2] == 0: #if robot is at coffeepot, holding something, and nothing at right spot
                    #TODO: place object

                    #After skill execution, cup is placed
                    cur_state[2] = cur_state[0] #whatever was in hand is now at left coffee pot location
                    cur_state[0] = 0 #not holding anything anymore

                    end_state = cur_state

                    #check if end of episode (either for task success or just to reset)
                    resp = input("Should take be reset?")
                    if resp == "y":
                        done = 1
                        cur_state = generate_start_state()
                        # move robot to start location
                        self._goto_locale([locales[cur_state[13]]])
                    else:
                        done = 0
                        cur_state = end_state

                    print(f"end state {end_state}")

                    attempt_action.append([start_state, 1])
                    data_collection.append([start_state, try_action_idx, end_state])
                else:
                    print("can not place cup on right side")
                    attempt_action.append([start_state, 0])

            elif try_action[0] == self._close_lid: #close lid
                if start_state[13] == locales.index("coffee_pot") and start_state[0] == 0 and start_state[10] == 1: #if robot is at coffeepot, holding nothing, and lid is open
                    #TODO: close lid

                    #After skill execution
                    cur_state[10] = 0 #lid closed now

                    end_state = cur_state

                    #check if end of episode (either for task success or just to reset)
                    resp = input("Should take be reset?")
                    if resp == "y":
                        done = 1
                        cur_state = generate_start_state()
                        # move robot to start location
                        self._goto_locale([locales[cur_state[13]]])
                    else:
                        done = 0
                        cur_state = end_state

                    print(f"end state {end_state}")

                    attempt_action.append([start_state, 1])
                    data_collection.append([start_state, try_action_idx, end_state])
                else:
                    print("can not close lid")
                    attempt_action.append([start_state, 0])

            elif try_action[0] == self._push_button: #push button
                if start_state[13] == locales.index("coffee_pot") and start_state[0] == 0 and start_state[8] == 1 and start_state[9] == 1 and start_state[10] == 0 and start_state[11] == 0 : #if robot is at coffeepot, holding nothing, lid is closed, water and grinds are in coffee pot, button is off
                    #TODO: push button

                    #After skill execution
                    cur_state[11] = 1 #button on
                    cur_state[12] = 1 #coffee made

                    end_state = cur_state


                    #check if end of episode (either for task success or just to reset)
                    resp = input("Should take be reset?")
                    if resp == "y":
                        done = 1
                        cur_state = generate_start_state()
                        # move robot to start location
                        self._goto_locale([locales[cur_state[13]]])
                    else:
                        done = 0
                        cur_state = end_state

                    print(f"end state {end_state}")

                    attempt_action.append([start_state, 1])
                    data_collection.append([start_state, try_action_idx, end_state])
                else:
                    print("can not push button")
                    attempt_action.append([start_state, 0])


            else:
                raise Exception("tried infeasible action, sadness")

        print(f"data collection {data_collection}")
        print(f"attempt action: {attempt_action}")





    def _open_noodle_door(self, *args):

        best_obj, best_obj_label, image_full, best_vision_tform_obj, seed_tform_obj, source = self.vision_model.detect_objects_hand(5,"noodle_handle")

        self.fetch_model.open_noodle_door(image_full, best_obj, best_vision_tform_obj)

    def _generate_aosm_and_plan(self, *args):
        #identify where the fiducials are
        #self._navigate_all_fiducials()
        self._upload_clusters()

        #Find where the objects are at
        object_locale_dict = self._find_where_objects_are()

        #generate plan based on where objects are at
        #First we get the water cup (water_cup), then we get coffee grinds cup
        plan = [
            [self._goto_locale, [object_locale_dict["water_cup"]]],
            [self._pick_cup, ["water_cup"]],
            [self._goto_locale, ["coffee_pot"]],
            [self._pour_water, ["coffee_pot"]],
            [self._place_cup_left,["coffee_pot"]],
            [self._goto_locale,[object_locale_dict["coffee_cup"]]],
            [self._pick_cup, ["coffee_cup"]],
            [self._goto_locale, ["coffee_pot"]],
            [self._pour_grinds, ["coffee_pot"]],
            [self._place_cup_right,["coffee_pot"]],
            [self._close_lid, ["coffee_pot"]],
            [self._push_button, ["coffee_pot"]]
        ]

        for step in plan:
            print(f"Executing step {step[0]} on {step[1]}")
            step[0](step[1])

    def _find_where_objects_are(self, *args):
        locales_to_vist = list(self._object_fiducials_dict.keys())
        object_locale_dict = {}
        for locale in locales_to_vist:
            self._goto_locale([locale])
            found_object = self._cam_search()
            if found_object != None:
                object_locale_dict[found_object] = locale
            self.stow_arm()

        print(object_locale_dict)
        return(object_locale_dict)

    def _cam_search(self, *args):
        ret_obj = None
        look_around(self._robot, self._robot_command_client, 0)
        best_obj, best_obj_label, image_full, best_vision_tform_obj, seed_tform_obj, source = self.vision_model.detect_objects_hand(
            5)
        print(f"I see {best_obj_label}")
        ret_obj = best_obj_label

        look_around(self._robot, self._robot_command_client, 1)
        best_obj, best_obj_label, image_full, best_vision_tform_obj, seed_tform_obj, source = self.vision_model.detect_objects_hand(
            5)
        print(f"I see {best_obj_label}")
        if best_obj_label != None:
            ret_obj = best_obj_label
        #self.stow_arm()
        return(best_obj_label)

    def _execute_plan(self, *args):
        plan = [
            [self._goto_locale, ["drawers"]],
            [self._pick_cup, []],
            [self._goto_locale, ["coffee_pot"]],
            [self._pour_grinds, ["coffee_pot"]],
            [self._place_cup_left,["coffee_pot"]],
            [self._goto_locale, ["table"]],
            [self._pick_cup, []],
            [self._goto_locale, ["coffee_pot"]],
            [self._pour_water, ["coffee_pot"]],
            [self._place_cup_right,["coffee_pot"]],
            [self._close_lid, ["coffee_pot"]],
            [self._push_button, ["coffee_pot"]]
        ]

        for step in plan:
            print(f"Executing step {step[0]} on {step[1]}")
            step[0](step[1])

    def carry_pose(self):
        # Stow the arm
        # Build the stow command using RobotCommandBuilder
        carry = RobotCommandBuilder.arm_carry_command()

        # Issue the command via the RobotCommandClient
        carry_command_id = self._robot_command_client.robot_command(carry)

        block_until_arm_arrives(self._robot_command_client, carry_command_id, 3.0)

    def stow_arm(self):
        # Stow the arm
        # Build the stow command using RobotCommandBuilder
        stow = RobotCommandBuilder.arm_stow_command()

        # Issue the command via the RobotCommandClient
        stow_command_id = self._robot_command_client.robot_command(stow)

        block_until_arm_arrives(self._robot_command_client, stow_command_id, 3.0)

    def _blocking_stand(self, *args):
        self.toggle_power(should_power_on=True)
        blocking_stand(self._robot_command_client, timeout_sec=10)

    def _goto_locale(self, *args):
        location = args[0][0]
        if location not in self._skill_offset_dict['go_to'].keys():
            print(f"{location} not valid")
            return
        print(f"going to {location}")
        # print(self._object_fiducials_dict[location])
        # print(self._skill_offset_dict['go_to'][location])
        # print("Going to object")
        self._go_to_fiducial_global(self._object_fiducials_dict[location], self._skill_offset_dict['go_to'][location])

    def _refine_locale(self, *args):
        location = args[0][0]
        if location not in self._skill_offset_dict['go_to'].keys():
            print(f"{location} not valid")
            return
        print(f"refining {location}")
        # print(self._object_fiducials_dict[location])
        # print(self._skill_offset_dict['go_to'][location])
        # print("Going to object")
        self._go_to_fiducial(self._object_fiducials_dict[location], self._skill_offset_dict['go_to'][location])


    def _push_button(self, *args):
        # location = args[0][0]
        location = 'coffee_pot'
        fid_number = self._object_fiducials_dict[location]

        # This realigns the robot based on local fiducial tracking
        # self._go_to_fiducial(self._object_fiducials_dict[location], self._skill_offset_dict['push_button'][location])

        get_body_tform_goal_fid = lambda position_rot_list: self.get_body_tform_goal(position_rot_list, fid_number)

        looking_negative_z = math_helpers.Quat.from_pitch(np.pi / 2)

        # In the fiducial frame
        cmd_poses_fiducial_frame = [[
            [0.25, -0.09, 0.1, looking_negative_z, 3.0],
            [0.25, -0.1, -0.05, looking_negative_z, 6.0]
        ],
            [
                [0.25, -0.1, -0.05, looking_negative_z, 1.0]
            ],
            [
                [0.25, -0.1, -0.05, looking_negative_z, 3.0],
                [0.25, -0.2, 0.07, looking_negative_z, 5.0]
            ]
        ]

        processed_cmd_poses_body_frame = []
        for l in cmd_poses_fiducial_frame:
            # List consiting of (SE(3),timing)
            cmd_poses_body_frame = [(get_body_tform_goal_fid(cmd_pose[:-1]), cmd_pose[-1]) for cmd_pose in l]
            processed_cmd_poses_body_frame.append(cmd_poses_body_frame)

        execute_trajectory_from_poses_oo(self._robot, self._robot_command_client, processed_cmd_poses_body_frame[0],
                                         1.0)
        execute_trajectory_from_poses_oo(self._robot, self._robot_command_client, processed_cmd_poses_body_frame[1],
                                         0.0)
        execute_trajectory_from_poses_oo(self._robot, self._robot_command_client, processed_cmd_poses_body_frame[2],
                                         1.0)

        self.stow_arm()

    def _open_gripper_and_stow(self, *args):
        gripper_open = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
        cmd_id = self._robot_command_client.robot_command(gripper_open)
        self.stow_arm()


    def _place_cup_left(self, *args):
        location = "coffee_pot"#args[0][0]
        fid_number = self._object_fiducials_dict[location]

        # This realigns the robot based on local fiducial tracking
        # self._go_to_fiducial(self._object_fiducials_dict[location], self._skill_offset_dict['push_button'][location])

        get_body_tform_goal_fid = lambda position_rot_list: self.get_body_tform_goal(position_rot_list, fid_number)

        looking_negative_z = math_helpers.Quat.from_pitch(np.pi / 2)
        holding_quat = math_helpers.Quat.from_roll(1.57)

        # In the fiducial frame
        cmd_poses_fiducial_frame = [[
            [0.5, 0.25, 0, looking_negative_z, 3.0],
            [0.5, 0.25, -0.15, looking_negative_z, 5.0],
            [0.34, 0.25, -0.15, looking_negative_z, 7.0]
        ],
        [
            [0.34, 0.3, -0.15, looking_negative_z, 2],
            [0.34, 0.33, .08, looking_negative_z, 3],
        ]
        ]

        processed_cmd_poses_body_frame = []

        for seq in cmd_poses_fiducial_frame:
            seq_1_poses_body_frame = []
            for pose in seq:  # for each pose in first move sequence
                cmd_pose_body_frame = (get_body_tform_goal_fid(pose[:-1]), pose[-1])

                # print(type(cmd_pose_body_frame[0].position))
                # print(cmd_pose_body_frame[0].position)
                new_pose_body_frame = [cmd_pose_body_frame[0].position.x, cmd_pose_body_frame[0].position.y,
                                       cmd_pose_body_frame[0].position.z, holding_quat, pose[-1]]
                # print(new_pose_body_frame)
                seq_1_poses_body_frame.append(new_pose_body_frame)
            processed_cmd_poses_body_frame.append(seq_1_poses_body_frame)


        # for seq in processed_cmd_poses_body_frame:
        execute_trajectory_from_poses(self._robot, self._robot_command_client, processed_cmd_poses_body_frame[0], 0)
        execute_trajectory_from_poses(self._robot, self._robot_command_client, processed_cmd_poses_body_frame[1], 1)

        self.stow_arm()

    def _place_cup_right(self, *args):
        location = "coffee_pot"#args[0][0]
        fid_number = self._object_fiducials_dict[location]

        # This realigns the robot based on local fiducial tracking
        # self._go_to_fiducial(self._object_fiducials_dict[location], self._skill_offset_dict['push_button'][location])

        get_body_tform_goal_fid = lambda position_rot_list: self.get_body_tform_goal(position_rot_list, fid_number)

        looking_negative_z = math_helpers.Quat.from_pitch(np.pi / 2)
        holding_quat = math_helpers.Quat.from_roll(1.57)

        # In the fiducial frame
        cmd_poses_fiducial_frame = [[
            [0.5, -0.23, 0, looking_negative_z, 3.0],
            [0.5, -0.23, -0.15, looking_negative_z, 5.0],
            [0.34, -0.23, -0.15, looking_negative_z, 7.0]
        ],
            [
                [0.34, -0.19, -0.15, looking_negative_z, 2],
                [0.34, -0.19, .08, looking_negative_z, 3],
            ]
        ]

        processed_cmd_poses_body_frame = []

        for seq in cmd_poses_fiducial_frame:
            seq_1_poses_body_frame = []
            for pose in seq:  # for each pose in first move sequence
                cmd_pose_body_frame = (get_body_tform_goal_fid(pose[:-1]), pose[-1])

                # print(type(cmd_pose_body_frame[0].position))
                # print(cmd_pose_body_frame[0].position)
                new_pose_body_frame = [cmd_pose_body_frame[0].position.x, cmd_pose_body_frame[0].position.y,
                                       cmd_pose_body_frame[0].position.z, holding_quat, pose[-1]]
                # print(new_pose_body_frame)
                seq_1_poses_body_frame.append(new_pose_body_frame)
            processed_cmd_poses_body_frame.append(seq_1_poses_body_frame)

        # for seq in processed_cmd_poses_body_frame:
        execute_trajectory_from_poses(self._robot, self._robot_command_client, processed_cmd_poses_body_frame[0], 0)
        execute_trajectory_from_poses(self._robot, self._robot_command_client, processed_cmd_poses_body_frame[1], 1)

        self.stow_arm()
    def _pour_water(self, *args):
        # location = args[0][0]
        location = "coffee_pot"
        fid_number = self._object_fiducials_dict[location]

        # This realigns the robot based on local fiducial tracking
        # self._go_to_fiducial(self._object_fiducials_dict[location], self._skill_offset_dict['pour_water'][location])

        get_body_tform_goal_fid = lambda position_rot_list: self.get_body_tform_goal(position_rot_list, fid_number)

        looking_negative_z = math_helpers.Quat.from_pitch(-np.pi / 2)

        cmd_poses_fiducial_frame = [[  # sequence 1, holding over machine
            #[0.5, 0, 0.1, looking_negative_z, 2.0],
            [0.7, -0.05, 0.1, looking_negative_z, 2.0],
            [0.7, -0.08, -0.24, looking_negative_z, 3.0],
        ],
            [  # sequence 2, shaking
                [0.7, -0.08, -0.24, looking_negative_z, 1.0],
                [0.7, -0.08, -0.24, looking_negative_z, 2.0]
            ],
            [  # sequence 3, retracting
                [0.7, -0.08, -0.24, looking_negative_z, 2.0],
                [0.7, -0.08, 0.1, looking_negative_z, 3.0],
                #[0.5, 0, 0.1, looking_negative_z, 6.0],
            ]
        ]

        processed_cmd_poses_body_frame = []

        holding_quat = math_helpers.Quat.from_roll(np.pi / 2)

        seq_1_poses_body_frame = []
        for pose in cmd_poses_fiducial_frame[0]:  # for each pose in first move sequence
            cmd_pose_body_frame = (get_body_tform_goal_fid(pose[:-1]), pose[-1])

            # print(type(cmd_pose_body_frame[0].position))
            # print(cmd_pose_body_frame[0].position)
            new_pose_body_frame = [cmd_pose_body_frame[0].position.x, cmd_pose_body_frame[0].position.y,
                                   cmd_pose_body_frame[0].position.z, holding_quat, pose[-1]]
            # print(new_pose_body_frame)
            seq_1_poses_body_frame.append(new_pose_body_frame)
        # print(seq_1_poses_body_frame)

        # make the shaking motion at the second pose
        # print("seq 2")
        pouring_quat_a = math_helpers.Quat.from_roll(-1.5)
        pouring_quat_b = math_helpers.Quat.from_roll(-1.57)
        shake_pose = cmd_poses_fiducial_frame[1][0]
        # print(f"shake pose fid frame{shake_pose}")
        shake_pose_body_frame = (get_body_tform_goal_fid(shake_pose[:-1]), shake_pose[-1])
        # print(f"shake pose body frame{shake_pose_body_frame}")
        seq_2_poses_body_frame = [
            [shake_pose_body_frame[0].position.x, shake_pose_body_frame[0].position.y, shake_pose_body_frame[0].position.z, pouring_quat_a, 3],
            [shake_pose_body_frame[0].position.x, shake_pose_body_frame[0].position.y, shake_pose_body_frame[0].position.z, pouring_quat_b, 3.4],
            [shake_pose_body_frame[0].position.x, shake_pose_body_frame[0].position.y, shake_pose_body_frame[0].position.z, pouring_quat_a, 3.6],
            [shake_pose_body_frame[0].position.x, shake_pose_body_frame[0].position.y, shake_pose_body_frame[0].position.z, pouring_quat_b, 3.8],
            [shake_pose_body_frame[0].position.x, shake_pose_body_frame[0].position.y, shake_pose_body_frame[0].position.z, pouring_quat_a, 5],
            [shake_pose_body_frame[0].position.x, shake_pose_body_frame[0].position.y, shake_pose_body_frame[0].position.z, holding_quat, 6],
        ]

        # print("seq 3")
        seq_3_poses_body_frame = []
        for pose in cmd_poses_fiducial_frame[2]:  # for each pose in first move sequence
            cmd_pose_body_frame = (get_body_tform_goal_fid(pose[:-1]), pose[-1])

            # print(type(cmd_pose_body_frame[0].position))
            # print(cmd_pose_body_frame[0].position)
            new_pose_body_frame = [cmd_pose_body_frame[0].position.x, cmd_pose_body_frame[0].position.y,
                                   cmd_pose_body_frame[0].position.z, holding_quat, pose[-1]]
            # print(new_pose_body_frame)
            seq_3_poses_body_frame.append(new_pose_body_frame)
        # print(seq_1_poses_body_frame)

        execute_trajectory_from_poses(self._robot, self._robot_command_client, seq_1_poses_body_frame, 0)
        execute_trajectory_from_poses(self._robot, self._robot_command_client, seq_2_poses_body_frame, 0)
        execute_trajectory_from_poses(self._robot, self._robot_command_client, seq_3_poses_body_frame, 0)

        self.stow_arm()
        pass

    def _pour_grinds(self, *args):
        # location = args[0][0]
        location = "coffee_pot"
        fid_number = self._object_fiducials_dict[location]

        # This realigns the robot based on local fiducial tracking
        # self._go_to_fiducial(self._object_fiducials_dict[location], self._skill_offset_dict['push_button'][location])

        get_body_tform_goal_fid = lambda position_rot_list: self.get_body_tform_goal(position_rot_list, fid_number)

        looking_negative_z = math_helpers.Quat.from_pitch(-np.pi / 2)

        cmd_poses_fiducial_frame = [[  # sequence 1, holding over machine
            #[0.5, 0, 0.1, looking_negative_z, 2.0],
            [0.7, -0.01, 0.1, looking_negative_z, 2.0],
            [0.7, -0.04, -0.11, looking_negative_z, 3.0],
        ],
            [  # sequence 2, shaking
                [0.7, -0.04, -0.11, looking_negative_z, 1.0],
                [0.7, -0.04, -0.11, looking_negative_z, 1.0]
            ],
            [  # sequence 3, retracting
                [0.7, -0.04, -0.11, looking_negative_z, 2.0],
                [0.7, -0.04, 0.1, looking_negative_z, 3.0],
                #[0.5, 0, 0.1, looking_negative_z, 6.0],
            ]

        ]

        processed_cmd_poses_body_frame = []

        holding_quat = math_helpers.Quat.from_roll(np.pi / 2)

        seq_1_poses_body_frame = []
        for pose in cmd_poses_fiducial_frame[0]:  # for each pose in first move sequence
            cmd_pose_body_frame = (get_body_tform_goal_fid(pose[:-1]), pose[-1])

            # print(type(cmd_pose_body_frame[0].position))
            # print(cmd_pose_body_frame[0].position)
            new_pose_body_frame = [cmd_pose_body_frame[0].position.x, cmd_pose_body_frame[0].position.y,
                                   cmd_pose_body_frame[0].position.z, holding_quat, pose[-1]]
            # print(new_pose_body_frame)
            seq_1_poses_body_frame.append(new_pose_body_frame)
        # print(seq_1_poses_body_frame)

        # make the shaking motion at the second pose
        # print("seq 2")
        pouring_quat_a = math_helpers.Quat.from_roll(-1.1)
        pouring_quat_b = math_helpers.Quat.from_roll(-1.2)
        shake_pose = cmd_poses_fiducial_frame[1][0]
        # print(f"shake pose fid frame{shake_pose}")
        shake_pose_body_frame = (get_body_tform_goal_fid(shake_pose[:-1]), shake_pose[-1])
        # print(f"shake pose body frame{shake_pose_body_frame}")
        seq_2_poses_body_frame = [
            [shake_pose_body_frame[0].position.x, shake_pose_body_frame[0].position.y, shake_pose_body_frame[0].position.z, pouring_quat_a, 3],
            [shake_pose_body_frame[0].position.x, shake_pose_body_frame[0].position.y, shake_pose_body_frame[0].position.z, pouring_quat_b, 3.4],
            [shake_pose_body_frame[0].position.x, shake_pose_body_frame[0].position.y, shake_pose_body_frame[0].position.z, pouring_quat_a, 3.6],
            [shake_pose_body_frame[0].position.x, shake_pose_body_frame[0].position.y, shake_pose_body_frame[0].position.z, pouring_quat_b, 3.8],
            [shake_pose_body_frame[0].position.x, shake_pose_body_frame[0].position.y, shake_pose_body_frame[0].position.z, pouring_quat_a, 5],
            [shake_pose_body_frame[0].position.x, shake_pose_body_frame[0].position.y, shake_pose_body_frame[0].position.z, holding_quat, 6],
        ]

        # print("seq 3")
        seq_3_poses_body_frame = []
        for pose in cmd_poses_fiducial_frame[2]:  # for each pose in first move sequence
            cmd_pose_body_frame = (get_body_tform_goal_fid(pose[:-1]), pose[-1])

            # print(type(cmd_pose_body_frame[0].position))
            # print(cmd_pose_body_frame[0].position)
            new_pose_body_frame = [cmd_pose_body_frame[0].position.x, cmd_pose_body_frame[0].position.y,
                                   cmd_pose_body_frame[0].position.z, holding_quat, pose[-1]]
            # print(new_pose_body_frame)
            seq_3_poses_body_frame.append(new_pose_body_frame)
        # print(seq_1_poses_body_frame)

        execute_trajectory_from_poses(self._robot, self._robot_command_client, seq_1_poses_body_frame, 0)
        execute_trajectory_from_poses(self._robot, self._robot_command_client, seq_2_poses_body_frame, 0)
        execute_trajectory_from_poses(self._robot, self._robot_command_client, seq_3_poses_body_frame, 0)

        self.stow_arm()
        pass


    def _close_lid(self, *args):
        # location = args[0][0]
        location = 'coffee_pot'
        print(f"location is {location}")
        # print(self._object_fiducials_dict[location])
        # print(self._skill_offset_dict['close_lid'][location])
        self._go_to_fiducial(self._object_fiducials_dict[location], self._skill_offset_dict['close_lid'][location])
        close_lid(self._robot, self._robot_command_client)
        self.stow_arm()
        self._go_to_fiducial(self._object_fiducials_dict[location], self._skill_offset_dict['push_button'][location])

    def average_3d_pose(self, l):
        av_x = 0
        av_y = 0
        av_z = 0

        quat_list = []

        for pose in l:
            av_x += pose.position.x
            av_y += pose.position.y
            av_z += pose.position.z

            quat = [pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w]
            quat_list.append(quat)

        av_x /= float(len(l))
        av_y /= float(len(l))
        av_z /= float(len(l))

        q = np.transpose(np.array(quat_list))
        # print(f"q shape is {q.shape}")
        q_qt = np.matmul(q, np.transpose(q))
        w, v = np.linalg.eig(q_qt)
        # print(f"eigen values: {w}")
        # print(f"eigen vectors: {v}")
        largest_eigen_value_idx = np.argmax(w)
        # print(f"largest eigen value idx: {largest_eigen_value_idx}")
        largest_eigen_vector = v[:, largest_eigen_value_idx]
        # print(f"largest eigen vector: {largest_eigen_vector}")

        av_quat = Quat(w=largest_eigen_vector[3], x=largest_eigen_vector[0], y=largest_eigen_vector[1],
                       z=largest_eigen_vector[2])

        return (SE3Pose(av_x, av_y, av_z, av_quat))  # l[0].rotation))

    def _go_to_fiducial_global(self, fid_number, offsets):
        x, y, z = offsets
        print('go_to_fid_global:')
        print(fid_number, offsets)
        localization_state = self._graph_nav_client.get_localization_state()
        seed_tform_body = SE3Pose.from_obj(localization_state.localization.seed_tform_body)

        vision_tform_body = bosdyn.client.frame_helpers.get_vision_tform_body(self._robot.get_frame_tree_snapshot())
        body_tform_vision = vision_tform_body.inverse()

        seed_tform_vision = seed_tform_body * body_tform_vision

        # print("Seed_tform_body:", seed_tform_body)
        world_object_client = self._robot.ensure_client(WorldObjectClient.default_service_name)
        # Get all fiducial objects (an object of a specific type).
        # request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]

        seed_tform_fiducial = self.average_3d_pose(
            self.all_fiducial_to_pose_dict[fid_number])  # self.all_fiducial_to_pose_dict[fid_number][0]

        fiducial_tform_goto = SE3Pose(x, y, z, Quat())
        seed_tform_goto = seed_tform_fiducial * fiducial_tform_goto

        # print("Seed tform fiducial", seed_tform_fiducial)
        # print("Seed tform goto:", seed_tform_goto)

        self._navigate_to_anchor(
            [seed_tform_goto.position.x, seed_tform_goto.position.y, seed_tform_goto.rotation.to_yaw()])

    def _go_to_fiducial(self, fid_number, offsets):
        x, y, z = offsets
        localization_state = self._graph_nav_client.get_localization_state()
        seed_tform_body = SE3Pose.from_obj(localization_state.localization.seed_tform_body)
        # print(f"seed_tform_body{seed_tform_body}")

        vision_tform_body = bosdyn.client.frame_helpers.get_vision_tform_body(self._robot.get_frame_tree_snapshot())
        body_tform_vision = vision_tform_body.inverse()

        seed_tform_vision = seed_tform_body * body_tform_vision

        # print("Seed_tform_body:", seed_tform_body)
        world_object_client = self._robot.ensure_client(WorldObjectClient.default_service_name)
        # Get all fiducial objects (an object of a specific type).
        request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]
        fiducial_objects = world_object_client.list_world_objects(
            object_type=request_fiducials).world_objects


        for fiducial_object in fiducial_objects:
            if fid_number in list(fiducial_object.transforms_snapshot.child_to_parent_edge_map.keys()):
                vision_tform_fiducial = fiducial_object.transforms_snapshot.child_to_parent_edge_map[
                    fid_number].parent_tform_child
                break
        # # TODO: We need to loop through fiducial_objects and look for a valid fid_number instance
        # vision_tform_fiducial = fiducial_objects[0].transforms_snapshot.child_to_parent_edge_map[
        #     fid_number].parent_tform_child
        vision_tform_fiducial = SE3Pose(vision_tform_fiducial.position.x, vision_tform_fiducial.position.y,
                                        vision_tform_fiducial.position.z, vision_tform_fiducial.rotation)

        seed_tform_fiducial = seed_tform_vision * vision_tform_fiducial

        fiducial_tform_goto = SE3Pose(x, y, z, Quat())
        seed_tform_goto = seed_tform_fiducial * fiducial_tform_goto

        # print("Seed tform fiducial", seed_tform_fiducial)
        # print("Seed tform goto:", seed_tform_goto)

        self._navigate_to_anchor(
            [seed_tform_goto.position.x, seed_tform_goto.position.y, seed_tform_goto.rotation.to_yaw()])

    def _list_objects(self, *args):
        if self.vision_model.clusters is None:
            print("There are no objects. Please upload clusters first.")
        for label in self.vision_model.clusters.keys():
            print(label)

    def _upload_clusters(self, *args):

        if not os.path.isfile("fiducial_info.pkl"):
            print("fiducial_info.pkl does not exist")
            return

        with open('fiducial_info.pkl', 'rb') as handle:
            self.all_fiducial_to_pose_dict = pickle.load(handle)


    def _navigate_to_object(self, *args):

        """Navigate to a specific waypoint."""
        print("args: ", args)
        # Take the first argument as the destination object.

        if len(args) < 1:
            # If no object name is given as input, then return without requesting navigation.
            print("No object provided as a destination for navigate to.")
            return

        if not args[0][0] in self.vision_model.clusters.keys():
            print(args[0][0] + " not in clusters.")
            return

        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        thetas = [0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330]

        seed_T_goal = self.vision_model.clusters[args[0][0]][0]  # the list will only have one element
        # seed_T_goal = SE3Pose(seed_T_goal[0],seed_T_goal[1],0,Quat())
        # print(seed_T_goal)

        nav_to_cmd_id = None

        found_object = False

        # Navigate to the destination.
        for theta in thetas:
            # print("Angle: " + str(theta))

            x = (1.0 * math.cos(math.radians(theta))) - (0 * math.sin(math.radians(theta)))
            y = (1.0 * math.sin(math.radians(theta))) + (0 * math.cos(math.radians(theta)))
            goal_tform_goto = SE3Pose(x, y, 0, Quat())
            # print(goal_tform_goto)
            seed_tform_goto = seed_T_goal * goal_tform_goto
            # print(seed_tform_goto)

            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                if not self._navigate_to_anchor(
                        [seed_tform_goto.position.x, seed_tform_goto.position.y, seed_tform_goto.rotation.to_yaw()]):
                    print("Could not successfuly navigate to object location.")
                    continue
                # nav_to_cmd_id = self._graph_nav_client.navigate_to_anchor(seed_tform_goto.to_proto(), 5.0, command_id=nav_to_cmd_id)

            except ResponseError as e:
                print("Error while navigating {}".format(e))
                continue
                # return False

            # time.sleep(5.5)  # Sleep for second and a half to allow for command execution.

            # the specified location is not attainable
            # if not self._check_success(nav_to_cmd_id):
            #     print("Could not successfuly navigate to object location.")
            #     continue

            best_obj, best_obj_label, image, vision_tform_obj, seed_tform_obj, source = self.vision_model.get_object_and_image()

            if best_obj is None:
                print("No object found.")
                # Didn't find anything, keep searching.
                continue

            prediction = \
            self.vision_model.kmeans_model.predict([[seed_tform_obj.x, seed_tform_obj.y, seed_tform_obj.z]])[0]

            if str(prediction) not in args[0][0]:
                found_object = True
                print("The object found does not match the cluster we specified")
                continue
            else:
                print("Correct object found!.")
                break

        # This needs to only work for objects that need a fiducial (something like the coffee pot)
        # if "coffee_pot" in args[0][0] and found_object:
        #     self.__orient_about_fiducial('filtered_fiducial_527')

        return found_object

    def __orient_about_fiducial(self, fiducial_name):
        # Navigate to the destination.

        # TODO: Add code that makes spot rotate around the goal point to find the fiducial

        localization_state = self._graph_nav_client.get_localization_state()
        seed_tform_body = SE3Pose.from_obj(localization_state.localization.seed_tform_body)

        # orient robot based on fiducial marking
        vision_tform_body = bosdyn.client.frame_helpers.get_vision_tform_body(self._robot.get_frame_tree_snapshot())
        body_tform_vision = vision_tform_body.inverse()

        seed_tform_vision = seed_tform_body * body_tform_vision

        world_object_client = self._robot.ensure_client(WorldObjectClient.default_service_name)
        # Get all fiducial objects (an object of a specific type).
        request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]
        fiducial_objects = world_object_client.list_world_objects(
            object_type=request_fiducials).world_objects

        vision_tform_fiducial = fiducial_objects[0].transforms_snapshot.child_to_parent_edge_map[
            fiducial_name].parent_tform_child
        vision_tform_fiducial = SE3Pose(vision_tform_fiducial.position.x, vision_tform_fiducial.position.y,
                                        vision_tform_fiducial.position.z, vision_tform_fiducial.rotation)

        seed_tform_fiducial = seed_tform_vision * vision_tform_fiducial

        fiducial_tform_goto = SE3Pose(0, 0, 1, Quat())
        seed_tform_goto = seed_tform_fiducial * fiducial_tform_goto

        self._navigate_to_anchor(
            [seed_tform_goto.position.x, seed_tform_goto.position.y, seed_tform_goto.rotation.to_yaw()])

    def _manipulate_object(self, *args):
        # self._navigate_to_object(args)
        label = None
        print("going to manipualte an object!")

        if "door_handle" in args[0][0]:
            label = "door_handle"

        elif "drawer" in args[0][0]:
            label = "drawer"

        elif "coffee_cup" in args[0][0]:
            label = "coffee_cup"
        else:
            label = "coffee_pot"
        print("Object to manipulate:", label)
        self.fetch_model.run_fetch(label, args[0][0])

        # """Navigate to a specific waypoint."""
        # print("args: ", args)
        # # Take the first argument as the destination object.
        #
        # if len(args) < 1:
        #     # If no object name is given as input, then return without requesting navigation.
        #     print("No object provided as a destination for navigate to.")
        #     return
        #
        # if not args[0][0] in self.vision_model.clusters.keys():
        #     print(args[0][0] + " not in clusters.")
        #     return
        #
        # seed_T_goal = self.vision_model.clusters[args[0][0]][0]
        #
        # if not self.toggle_power(should_power_on=True):
        #     print("Failed to power on the robot, and cannot complete navigate to request.")
        #     return
        #
        # nav_to_cmd_id = None
        # # Navigate to the destination.
        # is_finished = False
        # while not is_finished:
        #     """Get the current localization and state of the robot."""
        #     state = self._graph_nav_client.get_localization_state()
        #     seed_tform_body = SE3Pose.from_obj(state.localization.seed_tform_body)
        #
        #     distance = math.dist([seed_tform_body.x, seed_tform_body.y], [seed_T_goal.x, seed_T_goal.y])
        #
        #     # Need and acceptable distance away from object we want to manipulate
        #     if distance <= NAVIGATION_TO_OBJECT_ACCEPTABLE_DISTANCE:
        #         is_finished = True
        #         continue
        #
        #     # Issue the navigation command about twice a second such that it is easy to terminate the
        #     # navigation command (with estop or killing the program).
        #     try:
        #         nav_to_cmd_id = self._graph_nav_client.navigate_to_anchor(
        #             seed_T_goal.to_proto(), 1.0, command_id=nav_to_cmd_id)
        #     except ResponseError as e:
        #         print("Error while navigating {}".format(e))
        #         break
        #     time.sleep(.1)  # Sleep for tenth of a second to allow for command execution.
        #     # Poll the robot for feedback to determine if the navigation command is complete. Then sit
        #     # the robot down once it is finished.
        #     is_finished = self._check_success(nav_to_cmd_id)

        # # Power off the robot if appropriate.
        # if self._powered_on and not self._started_powered_on:
        #     # Sit the robot down + power off after the navigation command is complete.
        #     self.toggle_power(should_power_on=False)

    # def _test(self, *args):
    #     self.fetch_model.run_fetch("door_handle", None)

    def _get_localization_state(self, *args):
        """Get the current localization and state of the robot."""
        state = self._graph_nav_client.get_localization_state()
        print('Got localization: \n%s' % str(state.localization))
        odom_tform_body = get_odom_tform_body(state.robot_kinematics.transforms_snapshot)
        print('Got robot state in kinematic odometry frame: \n%s' % str(odom_tform_body))

    def _set_initial_localization_fiducial(self, *args):
        """Trigger localization when near a fiducial."""
        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an empty instance for initial localization since we are asking it to localize
        # based on the nearest fiducial.
        localization = nav_pb2.Localization()
        self._graph_nav_client.set_localization(initial_guess_localization=localization,
                                                ko_tform_body=current_odom_tform_body)

    def _set_initial_localization_waypoint(self, *args):
        """Trigger localization to a waypoint."""
        # Take the first argument as the localization waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without initializing.
            print("No waypoint specified to initialize to.")
            return
        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the unique waypoint id.
            return

        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an initial localization to the specified waypoint as the identity.
        localization = nav_pb2.Localization()
        localization.waypoint_id = destination_waypoint
        localization.waypoint_tform_body.rotation.w = 1.0
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization,
            # It's hard to get the pose perfect, search +/-20 deg and +/-20cm (0.2m).
            max_distance=0.2,
            max_yaw=20.0 * math.pi / 180.0,
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
            ko_tform_body=current_odom_tform_body)

    def _list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""

        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print("Empty graph.")
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = graph_nav_util.update_waypoints_and_edges(
            graph, localization_id)
        # print("waypoints:", self._current_annotation_name_to_wp_id)

    def _upload_graph_and_snapshots(self, *args):
        """Upload the graph and snapshots to the robot."""
        print("Loading the graph from disk into local storage...")
        with open(self._upload_filepath + "/graph", "rb") as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
            print("Loaded graph has {} waypoints and {} edges".format(
                len(self._current_graph.waypoints), len(self._current_graph.edges)))
        for waypoint in self._current_graph.waypoints:
            # Load the waypoint snapshots from disk.
            with open(self._upload_filepath + "/waypoint_snapshots/{}".format(waypoint.snapshot_id),
                      "rb") as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        for edge in self._current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            # Load the edge snapshots from disk.
            with open(self._upload_filepath + "/edge_snapshots/{}".format(edge.snapshot_id),
                      "rb") as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        print("Uploading the graph and snapshots to the robot...")
        true_if_empty = not len(self._current_graph.anchoring.anchors)
        response = self._graph_nav_client.upload_graph(graph=self._current_graph,
                                                       generate_new_anchoring=true_if_empty)
        # Upload the snapshots to the robot.
        for snapshot_id in response.unknown_waypoint_snapshot_ids:
            waypoint_snapshot = self._current_waypoint_snapshots[snapshot_id]
            self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
            print("Uploaded {}".format(waypoint_snapshot.id))
        for snapshot_id in response.unknown_edge_snapshot_ids:
            edge_snapshot = self._current_edge_snapshots[snapshot_id]
            self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
            print("Uploaded {}".format(edge_snapshot.id))

        # The upload is complete! Check that the robot is localized to the graph,
        # and if it is not, prompt the user to localize the robot before attempting
        # any navigation commands.
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            print("\n")
            print("Upload complete! The robot is currently not localized to the map; please localize", \
                  "the robot using commands (2) or (3) before attempting a navigation command.")

    def _navigate_to_anchor(self, *args):
        """Navigate to a pose in seed frame, using anchors."""
        # The following options are accepted for arguments: [x, y], [x, y, yaw], [x, y, z, yaw],
        # [x, y, z, qw, qx, qy, qz].
        # When a value for z is not specified, we use the current z height.
        # When only yaw is specified, the quaternion is constructed from the yaw.
        # When yaw is not specified, an identity quaternion is used.

        if len(args) < 1 or len(args[0]) not in [2, 3, 4, 7]:
            print("Invalid arguments supplied.")
            return False

        seed_T_goal = SE3Pose(float(args[0][0]), float(args[0][1]), 0.0, Quat())

        if len(args[0]) in [4, 7]:
            seed_T_goal.z = float(args[0][2])
        else:
            localization_state = self._graph_nav_client.get_localization_state()
            if not localization_state.localization.waypoint_id:
                print("Robot not localized")
                return False
            seed_T_goal.z = localization_state.localization.seed_tform_body.position.z

        if len(args[0]) == 3:
            seed_T_goal.rot = Quat.from_yaw(float(args[0][2]))
        elif len(args[0]) == 4:
            seed_T_goal.rot = Quat.from_yaw(float(args[0][3]))
        elif len(args[0]) == 7:
            seed_T_goal.rot = Quat(w=float(args[0][3]), x=float(args[0][4]), y=float(args[0][5]),
                                   z=float(args[0][6]))

        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return False

        nav_to_cmd_id = None
        # Navigate to the destination.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to_anchor(
                    seed_T_goal.to_proto(), 1.0, command_id=nav_to_cmd_id)
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished, status = self._check_success(nav_to_cmd_id)

        # Power off the robot if appropriate.
        # if self._powered_on and not self._started_powered_on:
        #     # Sit the robot down + power off after the navigation command is complete.
        #     self.toggle_power(should_power_on=False)

        return is_finished and status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL

    def _navigate_to(self, *args):
        """Navigate to a specific waypoint."""
        print("args: ", args)
        # Take the first argument as the destination waypoint.

        if len(args) < 1:
            # If no waypoint id is given as input, then return without requesting navigation.
            print("No waypoint provided as a destination for navigate to.")
            return

        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                                   command_id=nav_to_cmd_id)
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished, _ = self._check_success(nav_to_cmd_id)

        # Power off the robot if appropriate.
        # if self._powered_on and not self._started_powered_on:
        #     # Sit the robot down + power off after the navigation command is complete.
        #     self.toggle_power(should_power_on=False)

    def _navigate_route(self, *args):
        print("args: ", args)
        """Navigate through a specific route of waypoints."""
        if len(args) < 1 or len(args[0]) < 1:
            # If no waypoint ids are given as input, then return without requesting navigation.
            print("No waypoints provided for navigate route.")
            return
        waypoint_ids = args[0]
        for i in range(len(waypoint_ids)):
            waypoint_ids[i] = graph_nav_util.find_unique_waypoint_id(
                waypoint_ids[i], self._current_graph, self._current_annotation_name_to_wp_id)
            if not waypoint_ids[i]:
                # Failed to find the unique waypoint id.
                return

        edge_ids_list = []
        all_edges_found = True
        # Attempt to find edges in the current graph that match the ordered waypoint pairs.
        # These are necessary to create a valid route.
        for i in range(len(waypoint_ids) - 1):
            start_wp = waypoint_ids[i]
            end_wp = waypoint_ids[i + 1]
            edge_id = self._match_edge(self._current_edges, start_wp, end_wp)
            if edge_id is not None:
                edge_ids_list.append(edge_id)
            else:
                all_edges_found = False
                print("Failed to find an edge between waypoints: ", start_wp, " and ", end_wp)
                print(
                    "List the graph's waypoints and edges to ensure pairs of waypoints has an edge."
                )
                break

        if all_edges_found:
            if not self.toggle_power(should_power_on=True):
                print("Failed to power on the robot, and cannot complete navigate route request.")
                return

            # Navigate a specific route.
            route = self._graph_nav_client.build_route(waypoint_ids, edge_ids_list)
            is_finished = False
            while not is_finished:
                # Issue the route command about twice a second such that it is easy to terminate the
                # navigation command (with estop or killing the program).
                nav_route_command_id = self._graph_nav_client.navigate_route(
                    route, cmd_duration=1.0)
                time.sleep(.5)  # Sleep for half a second to allow for command execution.
                # Poll the robot for feedback to determine if the route is complete. Then sit
                # the robot down once it is finished.
                is_finished, _ = self._check_success(nav_route_command_id)

            # Power off the robot if appropriate.
            if self._powered_on and not self._started_powered_on:
                # Sit the robot down + power off after the navigation command is complete.
                self.toggle_power(should_power_on=False)

    def _clear_graph(self, *args):
        """Clear the state of the map on the robot, removing all waypoints and edges."""
        return self._graph_nav_client.clear_graph()

    def get_body_tform_goal(self, fiducial_tform_goal_list, fid_number):
        '''
        Given an [x,y,z,w,x,y,z] in fiducial frame, convert it into body frame through the seed frame
        '''

        '''
        Turn fiducial_tform_goal into an SE(3) Pose
        '''
        x, y, z, rot = fiducial_tform_goal_list
        fiducial_tform_goal = SE3Pose(x, y, z, rot)
        # print(f"fiducial_tform_goal: {fiducial_tform_goal}")
        '''
        First get seed_tform_fiducial based on local fiducial viewing
        '''
        localization_state = self._graph_nav_client.get_localization_state()
        seed_tform_body = SE3Pose.from_obj(localization_state.localization.seed_tform_body)

        # print(f"seed_tform_body {seed_tform_body}")

        vision_tform_body = bosdyn.client.frame_helpers.get_vision_tform_body(self._robot.get_frame_tree_snapshot())
        # print(f"Vision_tform_body {vision_tform_body}")
        body_tform_vision = vision_tform_body.inverse()

        world_object_client = self._robot.ensure_client(WorldObjectClient.default_service_name)
        # Get all fiducial objects (an object of a specific type).
        request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]
        fiducial_objects = world_object_client.list_world_objects(
            object_type=request_fiducials).world_objects

        for fiducial_object in fiducial_objects:
            if fid_number in list(fiducial_object.transforms_snapshot.child_to_parent_edge_map.keys()):
                vision_tform_fiducial = fiducial_object.transforms_snapshot.child_to_parent_edge_map[
                    fid_number].parent_tform_child
                break
        vision_tform_fiducial = SE3Pose(vision_tform_fiducial.position.x, vision_tform_fiducial.position.y,
                                        vision_tform_fiducial.position.z, vision_tform_fiducial.rotation)

        # print(f"vision_tform_fiducial {vision_tform_fiducial}")

        body_tform_fiducial = body_tform_vision * vision_tform_fiducial
        # print(f"body_tform_fiducial {body_tform_fiducial}")
        body_tform_goal = body_tform_fiducial * fiducial_tform_goal
        # print(f"body_tform_goal {body_tform_goal}")

        fbody_tform_body = bosdyn.client.frame_helpers.get_a_tform_b(self._robot.get_frame_tree_snapshot(), "flat_body",
                                                                     "body")
        fbody_tform_goal = fbody_tform_body * body_tform_goal
        return (fbody_tform_goal)

    def toggle_power(self, should_power_on):
        """Power the robot on/off dependent on the current power state."""
        is_powered_on = self.check_is_powered_on()
        if not is_powered_on and should_power_on:
            # Power on the robot up before navigating when it is in a powered-off state.
            power_on(self._power_client)
            motors_on = False
            while not motors_on:
                future = self._robot_state_client.get_robot_state_async()
                state_response = future.result(
                    timeout=10)  # 10 second timeout for waiting for the state response.
                if state_response.power_state.motor_power_state == robot_state_pb2.PowerState.STATE_ON:
                    motors_on = True
                else:
                    # Motors are not yet fully powered on.
                    time.sleep(.25)
        elif is_powered_on and not should_power_on:
            # Safe power off (robot will sit then power down) when it is in a
            # powered-on state.
            safe_power_off(self._robot_command_client, self._robot_state_client)
        else:
            # Return the current power state without change.
            return is_powered_on
        # Update the locally stored power state.
        self.check_is_powered_on()
        return self._powered_on

    def check_is_powered_on(self):
        """Determine if the robot is powered on or off."""
        power_state = self._robot_state_client.get_robot_state().power_state
        self._powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        return self._powered_on

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot and sit when command succeeds."""
        if command_id == -1:
            # No command, so we have no status to check.
            return False, None
        status = self._graph_nav_client.navigation_feedback(command_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True, status
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            print("Robot got lost when navigating the route, the robot will now sit down.")
            return True, status
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            print("Robot got stuck when navigating the route, the robot will now sit down.")
            return True, status
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            print("Robot is impaired.")
            return True, status
        else:
            # Navigation command is not complete yet.
            return False, None

    def _match_edge(self, current_edges, waypoint1, waypoint2):
        """Find an edge in the graph that is between two waypoint ids."""
        # Return the correct edge id as soon as it's found.
        for edge_to_id in current_edges:
            for edge_from_id in current_edges[edge_to_id]:
                if (waypoint1 == edge_to_id) and (waypoint2 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint2, to_waypoint=waypoint1)
                elif (waypoint2 == edge_to_id) and (waypoint1 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint1, to_waypoint=waypoint2)
        return None

    def _look_for_obj(self):
        self.obj_found = False
        self.thread_stopped = False
        THRESHOLD = 4
        while self.thread_running:
            print("in while loop, thread running is: ", self.thread_running)
            dogtoy, image, vision_tform_dogtoy, seed_tform_obj, source = self.vision_model.get_object_and_image(
                self.label)
            if dogtoy is not None:
                # check if distance is within threshold
                print("a")
                distance = (seed_tform_obj.position.x - self.loc.position.x) ** 2 + \
                           (seed_tform_obj.position.y - self.loc.position.y) ** 2 + \
                           (seed_tform_obj.position.z - self.loc.position.z) ** 2
                print("distance: ", distance)
                if distance < THRESHOLD:
                    self.obj_found = True
                    self.thread_stopped = True
                    print("going to break loop")
                    break
        print("out of loop")
        self.thread_stopped = True

    def get_seed_tform_fiducial(self):
        """
        returns the poses (in the seed frame) of each fiducial seen by the robot in this pose
        """
        # Get pose of robot
        localization_state = self._graph_nav_client.get_localization_state()
        # Get body in seed frame
        seed_tform_body = SE3Pose.from_obj(localization_state.localization.seed_tform_body)

        # get body in vision frame
        vision_tform_body = bosdyn.client.frame_helpers.get_vision_tform_body(self._robot.get_frame_tree_snapshot())
        # calculate body from vision
        body_tform_vision = vision_tform_body.inverse()

        # calculate seed to vision frame
        seed_tform_vision = seed_tform_body * body_tform_vision

        # get world client to get fiducials
        world_object_client = self._robot.ensure_client(WorldObjectClient.default_service_name)
        # Get all fiducial objects (an object of a specific type).
        request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]
        fiducial_objects = world_object_client.list_world_objects(
            object_type=request_fiducials).world_objects

        fiducial_to_pose = {}

        for fiducial_object in fiducial_objects:  # iterates through fiducials
            for fiducial_name in fiducial_object.transforms_snapshot.child_to_parent_edge_map.keys():
                if "filtered_fiducial" in fiducial_name:  # pulls the correct transform
                    print(f"I saw fiducial {fiducial_name}")
                    vision_tform_fiducial = fiducial_object.transforms_snapshot.child_to_parent_edge_map[
                        fiducial_name].parent_tform_child
                    vision_tform_fiducial = SE3Pose(vision_tform_fiducial.position.x, vision_tform_fiducial.position.y,
                                                    vision_tform_fiducial.position.z, vision_tform_fiducial.rotation)

                    seed_tform_fiducial = seed_tform_vision * vision_tform_fiducial

                    # Check if fiducial is close enough to body to count as detected
                    body_tform_fiducial = body_tform_vision * vision_tform_fiducial
                    dist_vect = np.array(
                        [body_tform_vision.position.x, body_tform_fiducial.position.y, body_tform_vision.position.z])
                    dist = np.linalg.norm(dist_vect)
                    if dist > 4:
                        print(f"{fiducial_name} is {dist} away which is too far to count")
                    else:
                        fiducial_to_pose[fiducial_name] = seed_tform_fiducial

        return (fiducial_to_pose)


    def _navigate_all_fiducials(self, *args):

        self._list_graph_waypoint_and_edge_ids([])

        waypoints = list(self._current_annotation_name_to_wp_id.values())

        if waypoints == []:
            print(
                "waypoints list is empty! Upload the graph  and initialise the robot to fiducial before calling this function")
            return

        # print("The waypoint list:", waypoints)

        all_fiducial_to_pose_dict = defaultdict(list)

        # TODO: increase this (reduced for testing purposes)
        for i in range(1):

            print("NOT VISITING ALL WAYPOINTS OFR TESTING PURPOSES")

            for waypoint in waypoints[:15]:
                if waypoint != None:
                    self._navigate_to([waypoint])

                    print("Looking for fiducials")
                    try:
                        fiducial_to_pose_dict = self.get_seed_tform_fiducial()

                        for fiducial in fiducial_to_pose_dict.keys():
                            all_fiducial_to_pose_dict[fiducial].append(fiducial_to_pose_dict[fiducial])
                    except:
                        continue

        print("storing objects.")
        self.all_fiducial_to_pose_dict = all_fiducial_to_pose_dict
        print("Dumping into pkl")
        pickle.dump(all_fiducial_to_pose_dict, open("fiducial_info.pkl","wb"))
        print("pickle dumped")


    def _can_pick_cup(self, *args):

        if "coffee_cup" in args[0][0]:
            label = "coffee_cup"
        if "water_cup" in args[0][0]:
            label = "water_cup"

        best_obj, best_obj_label, image_full, best_vision_tform_obj, seed_tform_obj, source = self.vision_model.detect_objects_hand(5,label)
        return best_obj_label != None

    def _pick_cup(self, *args):

        if "coffee_cup" in args[0][0]:
            label = "coffee_cup"
        if "water_cup" in args[0][0]:
            label = "water_cup"

        best_obj, best_obj_label, image_full, best_vision_tform_obj, seed_tform_obj, source = self.vision_model.detect_objects_hand(5,label)
        self.fetch_model.test_cup_pick(image_full, best_obj, best_vision_tform_obj)
        self.carry_pose()

    def _navigate_all(self, *args):

        self._list_graph_waypoint_and_edge_ids([])

        waypoints = list(self._current_annotation_name_to_wp_id.values())

        if waypoints == []:
            print(
                "waypoints list is empty! Upload the graph  and initialise the robot to fiducial before calling this function")
            return

        print("The waypoint list:", waypoints)

        # TODO: increase this (reduced for testing purposes)
        for i in range(1):

            print("NOT VISITING ALL WAYPOINTS OFR TESTING PURPOSES")

            for waypoint in waypoints[:10]:
                try:
                    print("Waypoint:", waypoint)
                    self._navigate_to([waypoint])
                    print("Running vision model while at waypoint")
                    self.vision_model.detect_objects(5)
                    print("Done searching, stopping vision model")
                except:
                    continue

        print("storing objects.")
        self.vision_model.save_objects_detected()

        # check if each cluster is valid
        # print("validating each cluster")
        # clusters = self.vision_model.clusters
        # new_clusters = {}
        # for cluster in clusters:
        #     self.label = cluster[cluster.find("__") + 2:]
        #     print(1)
        #     self._navigate_to_object([cluster])
        #
        #     self.obj_found = False
        #     self.thread_running = True
        #     self.loc = clusters[cluster]
        #     print(2)
        #     Thread(target = self._look_for_obj).start()
        #     command_client = self._robot.ensure_client(RobotCommandClient.default_service_name)
        #     print(3)
        #     footprint_R_body = EulerZXY(yaw=100, roll=0, pitch=0)
        #     cmd = RobotCommandBuilder.synchro_stand_command(footprint_R_body=footprint_R_body, body_height = 0.0)
        #     command_client.robot_command(cmd)
        #     self.thread_running = False
        #     print("before while")
        #     #commented out for testin purposes
        #     # while not self.thread_stopped:
        #     time.sleep(0.5)
        #     print("after while")
        #     if self.obj_found:
        #         new_clusters[cluster] = clusters[cluster]
        #     else:
        #         print("No match found for ", cluster, ". This cluster will be discarded.")
        # self.vision_model.clusters = new_clusters
        # print("the new clusters are: ", new_clusters)
        # clusters_f = open("clusters.pkl","wb")
        # pickle.dump(new_clusters, clusters_f)
        # clusters_f.close()

        # temporary changes - remove!
        # waypoints_1 = [waypoints[0]]
        # for wp in waypoints[1:]:
        #     if self._match_edge(self._current_edges, waypoints_1[-1], wp):
        #         waypoints_1.append(wp)
        # #--------------------------
        # self._navigate_route(waypoints_1)

    def _on_quit(self):
        """Cleanup on quit from the command line interface."""
        # Sit the robot down + power off after the navigation command is complete.
        if self._powered_on and not self._started_powered_on:
            self._robot_command_client.robot_command(RobotCommandBuilder.safe_power_off_command(),
                                                     end_time_secs=time.time())

    def run(self):
        """Main loop for the command line interface."""
        while True:
            print("""
            Options:
            (1) Get localization state.
            (2) Initialize localization to the nearest fiducial (must be in sight of a fiducial).
            (3) Initialize localization to a specific waypoint (must be exactly at the waypoint)."""

                  """
            (4) List the waypoint ids and edge ids of the map on the robot.
            (5) Upload the graph and its snapshots.
            (6) Navigate to. The destination waypoint id is the second argument.
            (7) Navigate route. The (in-order) waypoint ids of the route are the arguments.
            (8) Navigate to in seed frame. The following options are accepted for arguments: [x, y],
                [x, y, yaw], [x, y, z, yaw], [x, y, z, qw, qx, qy, qz]. (Don't type the braces).
                When a value for z is not specified, we use the current z height.
                When only yaw is specified, the quaternion is constructed from the yaw.
                When yaw is not specified, an identity quaternion is used.
            (9) Clear the current graph.
            (10) Visit All Waypoints. Detect objects.
            (11) List Objects 
            (12) Move To Object.
            (13) Manipulate Object.
            (14) Load saved fiducial information
            (15) Stand up
            (16) Pour grinds
            (17) Pour water
            (18) Close lid
            (19) Navigate all fiducials
            (20) Go to locale
            (21) pick coffee cup
            (22) Push button
            (23) Open gripper and stow
            (24) Place cup on left side
            (25) Place cup on right side
            (26) Execute a plan
            (27) Search for object with handcam locally
            (28) Find where objects are via vision
            (29) Generate an AOSM and execute a plan
            (30) Open Noodle door
            (31) Collect data for AOSM experiments
            (32) run LOMDP
            (q) Exit.
            """)

            try:
                inputs = input('>')
            except NameError:
                pass
            req_type = str.split(inputs)[0]

            if req_type == 'q':
                self._on_quit()
                break

            if req_type not in self._command_dictionary:
                print("Request not in the known command dictionary.")
                continue
            try:
                cmd_func = self._command_dictionary[req_type]
                cmd_func(str.split(inputs)[1:])
            except Exception as e:
                print(e)


def main(argv):
    """Run the command-line interface."""
    # os.system('python navigation/estop_gui.py')
    # Thread(target=estop_gui.main).start()
    # Setup and authenticate the robot.
    sdk = bosdyn.client.create_standard_sdk('GraphNavClient')
    sdk.register_service_client(NetworkComputeBridgeClient)
    robot = sdk.create_robot(HOSTNAME)
    bosdyn.client.util.authenticate(robot)

    graph_nav_command_line = GraphNavInterface(robot, UPLOAD_FILEPATH)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    try:
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                graph_nav_command_line.run()
                return True
            except Exception as exc:  # pylint: disable=broad-except
                print(exc)
                print("Graph nav command line client threw an error.")
                return False
    except ResourceAlreadyClaimedError:
        print(
            "The robot's lease is currently in use. Check for a tablet connection or try again in a few seconds."
        )
        return False



if __name__ == '__main__':
    exit_code = 0
    if not main(sys.argv[1:]):
        exit_code = 1
    os._exit(exit_code)  # Exit hard, no cleanup that could block.
