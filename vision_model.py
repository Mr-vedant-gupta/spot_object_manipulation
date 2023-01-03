from threading import Thread
import pickle
import numpy as np
import cv2
import math
import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.api import network_compute_bridge_pb2
from google.protobuf import wrappers_pb2
from bosdyn.client import frame_helpers
from bosdyn.client import math_helpers

MODEL_NAME = "handle-model"
SERVER_NAME = "fetch-server"
CONFIDENCE_THRESHOLD = 0.5

class VisionModel:
    image_sources = [
            'frontleft_fisheye_image', 'frontright_fisheye_image',
            'left_fisheye_image', 'right_fisheye_image', 'back_fisheye_image'
        ]
    def __init__(self, graph_nav_client, network_compute_client, robot):
        self.graph_nav_client = graph_nav_client
        self.network_compute_client = network_compute_client
        self.robot = robot

        self.thread = Thread(target = self.__thread_start_object_detection)
        self.kill_thread = False

        #TODO: dont use magic strings
        self.labels = ["door_handle","handle"]

        self.load_model(None)
    def load_model(self,path):
        # Via network compute server
        pass
    def __thread_start_object_detection(self):

        objects = {}
        index = 0
        # create a binary pickle file 
        f = open("object_dictionary.pkl","wb")
        while True:
            if self.kill_thread:
                # write the python object (dict) to pickle file
                pickle.dump(objects, f)
                f.close()
                break
            for l in self.labels:
                best_obj, image_full, best_vision_tform_obj, seed_tform_obj, source = self.get_object_and_image(l)

                if seed_tform_obj is not None:
                    name = l + "_" + str(index)
                    objects[name] = seed_tform_obj
                    index += 1

    def start_object_detection(self):
        self.kill_thread = False
        self.thread.start()

    def stop_object_detection(self):
        self.kill_thread = True

    def get_object_and_image(self, label):
        for source in self.image_sources:
            # Build a network compute request for this image source.
            image_source_and_service = network_compute_bridge_pb2.ImageSourceAndService(
                image_source=source)

            # Input data:
            #   model name
            #   minimum confidence (between 0 and 1)
            #   if we should automatically rotate the image
            input_data = network_compute_bridge_pb2.NetworkComputeInputData(
                image_source_and_service=image_source_and_service,
                model_name=MODEL_NAME,
                min_confidence=CONFIDENCE_THRESHOLD,
                rotate_image=network_compute_bridge_pb2.NetworkComputeInputData.ROTATE_IMAGE_ALIGN_HORIZONTAL)

            # Server data: the service name
            server_data = network_compute_bridge_pb2.NetworkComputeServerConfiguration(
                service_name=SERVER_NAME)

            # Pack and send the request.
            process_img_req = network_compute_bridge_pb2.NetworkComputeRequest(
                input_data=input_data, server_config=server_data)

            resp = self.network_compute_client.network_compute_bridge_command(
                process_img_req)

            best_obj = None
            highest_conf = 0.0
            best_vision_tform_obj = None

            img = self.get_bounding_box_image(resp)
            image_full = resp.image_response

            # Show the image
            cv2.imshow("Object", img)
            cv2.waitKey(15)

            if len(resp.object_in_image) > 0:
                for obj in resp.object_in_image:
                    # Get the label
                    obj_label = obj.name.split('_label_')[-1]
                    if obj_label != label:
                        continue
                    conf_msg = wrappers_pb2.FloatValue()
                    obj.additional_properties.Unpack(conf_msg)
                    conf = conf_msg.value

                    try:
                        vision_tform_obj = frame_helpers.get_a_tform_b(
                            obj.transforms_snapshot,
                            frame_helpers.VISION_FRAME_NAME,
                            obj.image_properties.frame_name_image_coordinates)

                        #graph = graph_nav_client.download_graph()

                        #print("GRAPH")
                        #print(graph)
                        vision_tform_body = bosdyn.client.frame_helpers.get_vision_tform_body(self.robot.get_frame_tree_snapshot())
                        body_tform_vision = vision_tform_body.inverse()
                        #print("body vision: ", body_tform_vision)
                        localization_state = self.graph_nav_client.get_localization_state()
                        seed_tform_body = localization_state.localization.seed_tform_body
                        # need to convert from geometry_pb2.SE3Pose to math_helpers.SE3Pose
                        seed_tform_body =  math_helpers.SE3Pose(seed_tform_body.position.x,seed_tform_body.position.y,seed_tform_body.position.z, seed_tform_body.rotation)
                        if seed_tform_body == None:
                            print("Forgot to upload map")
                        elif vision_tform_obj is not None:
                            print("seed_tform_body")
                            print(type(seed_tform_body))
                            print("body_tform_vision")
                            print(type(body_tform_vision))
                            print("vision_tform_obj")
                            print(type(vision_tform_obj))

                            seed_tform_obj = seed_tform_body * body_tform_vision * vision_tform_obj
                            print("seed tfrom obj: ", seed_tform_obj)

                    except bosdyn.client.frame_helpers.ValidateFrameTreeError:
                        # No depth data available.
                        vision_tform_obj = None

                    if conf > highest_conf and vision_tform_obj is not None:
                        highest_conf = conf
                        best_obj = obj
                        best_vision_tform_obj = vision_tform_obj

            if best_obj is not None:
                return best_obj, image_full, best_vision_tform_obj, seed_tform_obj, source

        return None, None, None, None, None

    def get_bounding_box_image(self, response):
        dtype = np.uint8
        img = np.fromstring(response.image_response.shot.image.data, dtype=dtype)
        if response.image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(response.image_response.shot.image.rows,
                              response.image_response.shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)

        # Convert to BGR so we can draw colors
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Draw bounding boxes in the image for all the detections.
        for obj in response.object_in_image:
            conf_msg = wrappers_pb2.FloatValue()
            obj.additional_properties.Unpack(conf_msg)
            confidence = conf_msg.value

            polygon = []
            min_x = float('inf')
            min_y = float('inf')
            for v in obj.image_properties.coordinates.vertexes:
                polygon.append([v.x, v.y])
                min_x = min(min_x, v.x)
                min_y = min(min_y, v.y)

            polygon = np.array(polygon, np.int32)
            polygon = polygon.reshape((-1, 1, 2))
            cv2.polylines(img, [polygon], True, (0, 255, 0), 2)

            caption = "{} {:.3f}".format(obj.name, confidence)
            cv2.putText(img, caption, (int(min_x), int(min_y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img

    def find_center_px(self, polygon):
        min_x = math.inf
        min_y = math.inf
        max_x = -math.inf
        max_y = -math.inf
        for vert in polygon.vertexes:
            if vert.x < min_x:
                min_x = vert.x
            if vert.y < min_y:
                min_y = vert.y
            if vert.x > max_x:
                max_x = vert.x
            if vert.y > max_y:
                max_y = vert.y
        x = math.fabs(max_x - min_x) / 2.0 + min_x
        y = math.fabs(max_y - min_y) / 2.0 + min_y
        return (x, y)

