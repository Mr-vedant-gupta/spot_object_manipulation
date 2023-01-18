from threading import Thread
import pickle
import numpy as np
import cv2
import math
import time
import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.api import network_compute_bridge_pb2
from google.protobuf import wrappers_pb2
from bosdyn.client import frame_helpers
from bosdyn.client import math_helpers
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

MODEL_NAME = "handle-model"
SERVER_NAME = "fetch-server"
CONFIDENCE_THRESHOLD = 0.5

class VisionModel:
    image_sources = [
            'frontleft_fisheye_image', 'frontright_fisheye_image',
            'left_fisheye_image', 'right_fisheye_image', 'back_fisheye_image'
        ]
    def __init__(self, graph_nav_client, network_compute_client, robot):
        self.clusters = None
        self.kmeans_model = None 
        self.graph_nav_client = graph_nav_client
        self.network_compute_client = network_compute_client
        self.robot = robot
        self.objects = []

        #TODO: dont use magic strings
        self.labels = ["door_handle","drawer","coffee_pot","coffee_cup"]

        self.load_model(None)
    def load_model(self,path):
        # Via network compute server
        pass

    def __kmeans_cluster(self, objects):
        clusters = {}

        # objects is in form [(label,SE3Pose),(label,SE3Pose)...]
        X = np.array([[o[1].x, o[1].y, o[1].z] for o in objects])
        y = [o[0] for o in objects] 

        min_score = math.inf

        best_kmeans = None
        #print("objects array: ", objects)

        best_kmeans = KMeans(n_clusters=5, n_init="auto").fit(X)

        # # determine best k value
        # for i in range(2, len(objects)):
        #     #print(i, X)
        #     kmeans = KMeans(n_clusters=i, n_init="auto").fit(X)
        #     score = silhouette_score(X, kmeans.labels_)
        #     #print("score: ", score)
        #
        #     if score < min_score:
        #         best_kmeans = kmeans
        #         min_score = score

        # add objects to their proper cluster dictionary key name
        #print("min score, best_km: ", min_score, best_kmeans)

        for i,label in enumerate(best_kmeans.labels_):
            cluster_name = "object_" + str(label) + "__" + y[i] #adding two dashes before label name so that it can be extracted easily

            # Add cluster name to dictionary keys
            if not cluster_name in clusters:
                clusters[cluster_name] = []

            # add object (SE3Pose) to its correct cluster
            clusters[cluster_name].append(objects[i][1])

        return clusters, best_kmeans
    
    def _find_cluster_averages(self, clusters):
        averaged_cluster = {}
        for cluster_name in clusters:
            n = len(clusters[cluster_name])
            x_values = [i.position.x for i in clusters[cluster_name]]
            x = sum(x_values)/n
            y_values = [i.position.y for i in clusters[cluster_name]]
            y = sum(y_values)/n
            z_values = [i.position.z for i in clusters[cluster_name]]
            z = sum(z_values)/n

            #pick rotation value randomly from one of the SE3Poses
            rotation = clusters[cluster_name][0].rotation
            average_pose = math_helpers.SE3Pose(x, y, z, rotation)
            averaged_cluster[cluster_name] = [average_pose]
        return averaged_cluster

    def save_objects_detected(self):
        raw_data = [(obj[0],[obj[1].position.x, obj[1].position.y, obj[1].position.z, obj[1].rotation.w, obj[1].rotation.x, obj[1].rotation.y, obj[1].rotation.z]) for obj in self.objects]

        pickle.dump(raw_data, open("raw_data.pkl","wb"))


        # create a binary pickle file 
        clusters_f = open("clusters.pkl","wb")
        kmeans_f = open("kmeans_model.pkl","wb")

        # write the python object (dict) to pickle file
        self.clusters, self.kmeans_model = self.__kmeans_cluster(self.objects)
        #self.clusters = self._find_cluster_averages(self.clusters)

        pickle.dump(self.clusters, clusters_f)
        pickle.dump(self.kmeans_model, kmeans_f)

        clusters_f.close()
        kmeans_f.close()
        self.objects = []


    def detect_objects(self, n_seconds):
        index = 0
        
        t_end = time.time() + n_seconds

        while time.time() < t_end:
            # for l in self.labels:
            best_obj,best_obj_label, image_full, best_vision_tform_obj, seed_tform_obj, source = self.get_object_and_image()

            if seed_tform_obj is not None:

                print("Found " + best_obj_label +" while searching")

                self.objects.append((best_obj_label,seed_tform_obj))
                index += 1

    def get_object_and_image(self):
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
            try:
                resp = self.network_compute_client.network_compute_bridge_command(
                    process_img_req)
            except Exception as e:
                #print(e)
                #print("Moving on!")
                continue


            best_obj = None
            highest_conf = 0.0
            best_vision_tform_obj = None
            best_obj_label = None

            img = self.get_bounding_box_image(resp)
            image_full = resp.image_response

            if len(resp.object_in_image) > 0:
                # Show the image
                cv2.imshow("Object", img)
                cv2.waitKey(15)
                for obj in resp.object_in_image:
                    # Get the label
                    # obj_label = obj.name.split('_label_')[-1]
                    # if obj_label != label:
                    #     continue
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
                        best_obj_label = best_obj.name.split('_label_')[-1]

            if best_obj is not None:
                return best_obj, best_obj_label, image_full, best_vision_tform_obj, seed_tform_obj, source

        return None, None, None, None, None, None

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

