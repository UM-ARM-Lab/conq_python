# !/usr/bin/env python

from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonCore import Callback
from vtkmodules.vtkFiltersHybrid import vtkPolyDataSilhouette
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTerrain, vtkInteractorStyleTrackballCamera
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkPropPicker,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkRenderer
)
import argparse
import os
import numpy as np
import vtk

from bosdyn.api.graph_nav import map_pb2
from bosdyn.client.frame_helpers import *
from bosdyn.client.math_helpers import *
from bosdyn_vtk_utils import api_to_vtk_se3_pose, numpy_to_poly_data, vtk_to_mat, mat_to_vtk


def get_program_parameters():
    description = 'Load a map generated by spot and click on waypoints to navigate.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('path', type=str, help='Full filepath to folder with graph and snapshots.', 
                        nargs='?', default='/home/arthurlovekin/spot/maps/collabspace1/') #TODO: Remove once development is over
    parser.add_argument('-a', '--anchoring', action='store_true',
                        help='Draw the map according to the anchoring (in seed frame).')
    options = parser.parse_args()
    print(f"using path: {options.path}")
    return options.path, options.anchoring

# TODO: Create waypoint and edge objects that simultaneously store the data from the graph file,
# and are also renderable by VTK. The process should essentially be to read in the data and as you are doing so 
# create the VTK objects which are immediately rendered and stored in a graph object (as opposed to the current map, 
# which is a set of dictionaries)
# TODO: Use the SpotMap across both the armlab_graph_nav_command_line and the clickmap_nav (make a map class used by the GraphNavInterface base class)

class SpotMap():
    """ 
    Object that loads a map previously generated by spot and stores:
        self.graph - the graph protobuf
        self.waypoints - dict between waypoint ID and waypoint
        self.waypoint_snapshots - dict between waypoint ID and waypoint snapshot
        self.edge_snapshots - dict between edge ID and edge snapshot
        self.anchors - dict between anchor ID and anchor
        self.anchored_world_objects - dict between anchored world object ID and anchored world object
    """
    def __init__(self, path):
        # Set up maps from waypoint ID to waypoints, edges, snapshots, etc.
        self.graph = map_pb2.Graph()
        self.waypoints = {}
        self.waypoint_snapshots = {}
        self.edge_snapshots = {}
        self.anchors = {}
        self.anchored_world_objects = {}
        
        with open(os.path.join(path, 'graph'), 'rb') as graph_file:
            # Load the graph file and deserialize it. The graph file is a protobuf containing only the waypoints and the
            # edges between them.
            data = graph_file.read()
            self.graph.ParseFromString(data)

            # Load the anchored world objects first so we can look in each waypoint snapshot as we load it.
            for anchored_world_object in self.graph.anchoring.objects:
                self.anchored_world_objects[anchored_world_object.id] = (anchored_world_object,)
            # For each waypoint, load any snapshot associated with it.
            for waypoint in self.graph.waypoints:
                self.waypoints[waypoint.id] = waypoint

                if len(waypoint.snapshot_id) == 0:
                    continue
                # Load the snapshot. Note that snapshots contain all of the raw data in a waypoint and may be large.
                file_name = os.path.join(path, 'waypoint_snapshots', waypoint.snapshot_id)
                if not os.path.exists(file_name):
                    continue
                with open(file_name, 'rb') as snapshot_file:
                    waypoint_snapshot = map_pb2.WaypointSnapshot()
                    waypoint_snapshot.ParseFromString(snapshot_file.read())
                    self.waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot

                    for fiducial in waypoint_snapshot.objects:
                        if not fiducial.HasField('apriltag_properties'):
                            continue

                        str_id = str(fiducial.apriltag_properties.tag_id)
                        if (str_id in self.anchored_world_objects and
                                len(self.anchored_world_objects[str_id]) == 1):

                            # Replace the placeholder tuple with a tuple of (wo, waypoint, fiducial).
                            anchored_wo = self.anchored_world_objects[str_id][0]
                            self.anchored_world_objects[str_id] = (anchored_wo, waypoint, fiducial)

            # Similarly, edges have snapshot data.
            for edge in self.graph.edges:
                if len(edge.snapshot_id) == 0:
                    continue
                file_name = os.path.join(path, 'edge_snapshots', edge.snapshot_id)
                if not os.path.exists(file_name):
                    continue
                with open(file_name, 'rb') as snapshot_file:
                    edge_snapshot = map_pb2.EdgeSnapshot()
                    edge_snapshot.ParseFromString(snapshot_file.read())
                    self.edge_snapshots[edge_snapshot.id] = edge_snapshot
            for anchor in self.graph.anchoring.anchors:
                self.anchors[anchor.id] = anchor
            print(
                f'Loaded graph with {len(self.graph.waypoints)} waypoints, {len(self.graph.edges)} edges, '
                f'{len(self.graph.anchoring.anchors)} anchors, and {len(self.graph.anchoring.objects)} anchored world objects'
            )

class bosdynWaypointActor(vtk.vtkActor):
    def __init__(self, waypoint_id):
        super().__init__()
        self.waypoint_id = waypoint_id

class VTKEngine():
    def __init__(self, spot_map):

        colors = vtkNamedColors()
        # A renderer and render window
        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.05,0.1,0.15) #colors.GetColor3d('SteelBlue'))

        self.renderWindow = vtkRenderWindow()
        self.renderWindow.SetSize(640, 480)
        self.renderWindow.AddRenderer(self.renderer)

        # An interactor
        self.interactor = vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.renderWindow)

        # Create an interface to create actors from the map datastructure
        bosdyn_vtk_interface = BosdynVTKInterface(spot_map, self.renderer)

        # # Display map objects extracted from file
        # if anchoring:
        #     if len(map.graph.anchoring.anchors) == 0:
        #         print('No anchors to draw.')
        #         sys.exit(-1)
        #     avg_pos = bosdyn_vtk_interface.create_anchored_graph_objects()
        # else:
        #     avg_pos = bosdyn_vtk_interface.create_graph_objects()
        avg_pos = bosdyn_vtk_interface.create_graph_objects()
        camera_pos = avg_pos + np.array([-1.0, 0.0, 5.0])
        camera = self.renderer.GetActiveCamera()
        camera.SetViewUp(0.0, 0.0, 1.0)
        camera.SetPosition(camera_pos[0], camera_pos[1], camera_pos[2])

        # TODO: Move the silhouetteActor generation somewhere else
        # Create the silhouette pipeline, the input data will be set in the
        # interactor
        silhouette = vtkPolyDataSilhouette()
        silhouette.SetCamera(self.renderer.GetActiveCamera())

        # Create mapper and actor for silhouette
        silhouetteMapper = vtkPolyDataMapper()
        silhouetteMapper.SetInputConnection(silhouette.GetOutputPort())

        silhouetteActor = vtkActor()
        silhouetteActor.SetMapper(silhouetteMapper)
        silhouetteActor.GetProperty().SetColor(colors.GetColor3d("Tomato"))
        silhouetteActor.GetProperty().SetLineWidth(5)

        # Set the custom type to use for interaction.
        style = SpotCommandInteractorStyle(silhouette, silhouetteActor)
        style.SetDefaultRenderer(self.renderer)

        # Start
        self.interactor.Initialize()
        self.interactor.SetInteractorStyle(style)
        self.renderWindow.SetWindowName('HighlightWithSilhouette')
        # self.renderWindow.Render()

        # self.interactor.Start()

    def start(self):
        self.renderWindow.Render()
        self.interactor.Start()


    def render(self):
        self.renderWindow.Render()



class BosdynVTKInterface():
    """
    Converts the boston dynamics map data structure into actors that are renderable by VTK
    """
    def __init__(self, map, vtk_renderer):
        self.map = map
        self.renderer = vtk_renderer


    def create_fiducial_object(self,world_object, waypoint, renderer):
        """
        Creates a VTK object representing a fiducial.
        :param world_object: A WorldObject representing a fiducial.
        :param waypoint: The waypoint the AprilTag is associated with.
        :param renderer: The VTK renderer
        :return: a tuple of (vtkActor, 4x4 homogenous transform) representing the vtk actor for the fiducial, and its
        transform w.r.t the waypoint.
        """
        fiducial_object = world_object.apriltag_properties
        odom_tform_fiducial_filtered = get_a_tform_b(
            world_object.transforms_snapshot, ODOM_FRAME_NAME,
            world_object.apriltag_properties.frame_name_fiducial_filtered)
        waypoint_tform_odom = SE3Pose.from_proto(waypoint.waypoint_tform_ko)
        waypoint_tform_fiducial_filtered = api_to_vtk_se3_pose(
            waypoint_tform_odom * odom_tform_fiducial_filtered)
        plane_source = vtk.vtkPlaneSource()
        plane_source.SetCenter(0.0, 0.0, 0.0)
        plane_source.SetNormal(0.0, 0.0, 1.0)
        plane_source.Update()
        plane = plane_source.GetOutput()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(plane)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.5, 0.7, 0.9)
        actor.SetScale(fiducial_object.dimensions.x, fiducial_object.dimensions.y, 1.0)
        renderer.AddActor(actor)
        return actor, waypoint_tform_fiducial_filtered

    def make_point_cloud_actor(self, point_cloud_data, waypoint_id):
        """
        Create a VTK actor representing the point cloud in a snapshot. 
        :param point_cloud_data: (3xN) point cloud data (np.array)
        :param waypoint_id: the waypoint ID of the waypoint whose point cloud we want to render.
        """
        poly_data = numpy_to_poly_data(point_cloud_data) 
        arr = vtk.vtkFloatArray()
        for i in range(poly_data.GetNumberOfVerts()): 
            arr.InsertNextValue(point_cloud_data[i, 2])
        arr.SetName('z_coord')
        poly_data.GetPointData().AddArray(arr)
        poly_data.GetPointData().SetActiveScalars('z_coord')
        actor = bosdynWaypointActor(waypoint_id)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        mapper.ScalarVisibilityOn()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(2)
        actor.PickableOff()

        self.renderer.AddActor(actor)
        return actor



    def create_point_cloud_object(self, homogeneous_tf, waypoint_id):
        """
        Create a VTK object representing the point cloud in a snapshot. Note that in graph_nav, "point cloud" refers to the
        feature cloud of a waypoint -- that is, a collection of visual features observed by all five cameras at a particular
        point in time. The visual features are associated with points that are rigidly attached to a waypoint.
        :param waypoints: dict of waypoint ID to waypoint.
        :param snapshots: dict of waypoint snapshot ID to waypoint snapshot.
        :param waypoint_id: the waypoint ID of the waypoint whose point cloud we want to render.
        :return: a vtkActor containing the point cloud data.
        """
        wp = self.map.waypoints[waypoint_id]
        snapshot = self.map.waypoint_snapshots[wp.snapshot_id]
        cloud = snapshot.point_cloud

        point_cloud_data = np.frombuffer(cloud.data, dtype=np.float32).reshape(int(cloud.num_points), 3)
        point_cloud_data = SE3Pose.transform_cloud_from_matrix(homogeneous_tf,point_cloud_data)
        actor = self.make_point_cloud_actor(point_cloud_data, waypoint_id)

        return actor

    def make_sphere_actor(self, xyz_location, waypoint_id):
        """
        Create a VTK object representing the center of a waypoint as a sphere
        :param xyz_location: the 4x4 homogenous transform of the waypoint (np.array)
        :param waypoint_id: the waypoint id of the waypoint object we wish to create. (long string)
        :return: a vtkActor containing the center of the waypoint as a sphere
        """
        sphere = vtk.vtkSphereSource()
        
        sphere.SetCenter(xyz_location[0], xyz_location[1], xyz_location[2])
        sphere.SetRadius(0.3)
        sphere.Update()

        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())
        sphere_actor = bosdynWaypointActor(waypoint_id) #vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(1.0, 1.0, 1.0)

        self.renderer.AddActor(sphere_actor)
        return sphere_actor


    def make_line_actor(self, pt_A, pt_B, renderer):
        """
        Creates a VTK object which is a white line between two points.
        :param pt_A: starting point of the line.
        :param pt_B: ending point of the line.
        :param renderer: the VTK renderer.
        :return: A VTK object that is a while line between pt_A and pt_B.
        """
        line_source = vtk.vtkLineSource()
        line_source.SetPoint1(pt_A[0], pt_A[1], pt_A[2])
        line_source.SetPoint2(pt_B[0], pt_B[1], pt_B[2])
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(line_source.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetLineWidth(2)
        actor.GetProperty().SetColor(0.7, 0.7, 0.7)
        actor.PickableOff()
        renderer.AddActor(actor)
        return actor

    def make_text_actor(self, display_text, xyz_location):
        actor = vtk.vtkTextActor()
        actor.SetInput(display_text)
        prop = actor.GetTextProperty()
        prop.SetBackgroundColor(0.0, 0.0, 0.0)
        prop.SetBackgroundOpacity(0.5)
        prop.SetFontSize(16)
        coord = actor.GetPositionCoordinate()
        coord.SetCoordinateSystemToWorld()
        coord.SetValue((xyz_location[0], xyz_location[1], xyz_location[2]))
        actor.PickableOff()
        self.renderer.AddActor(actor)
        return actor


    def create_edge_object(self, curr_wp_tform_to_wp, world_tform_curr_wp, renderer):
        # Concatenate the edge transform.
        world_tform_to_wp = np.dot(world_tform_curr_wp, curr_wp_tform_to_wp)
        # Make a line between the current waypoint and the neighbor.
        self.make_line_actor(world_tform_curr_wp[:3, 3], world_tform_to_wp[:3, 3], renderer)
        return world_tform_to_wp

    # TODO: Simplify the data reading process so that this function is either unnecessary or merged with the create_graph_objects function
    def create_anchored_graph_objects(self):
        pass
    #     """
    #     Creates all the VTK objects associated with the graph, in seed frame, if they are anchored.
    #     :param current_graph: the graph to use.
    #     :param current_waypoint_snapshots: dict from snapshot id to snapshot.
    #     :param current_waypoints: dict from waypoint id to waypoint.
    #     :param renderer: The VTK renderer
    #     :return: the average position in world space of all the waypoints.
    #     """
    #     current_graph = self.map.graph
    #     current_anchors = self.map.anchors
    #     current_anchored_world_objects = self.map.anchored_world_objects
    #     renderer = self.renderer

    #     avg_pos = np.array([0.0, 0.0, 0.0])
    #     waypoints_in_anchoring = 0
    #     # Create VTK objects associated with each waypoint.
    #     for waypoint in current_graph.waypoints:
    #         if waypoint.id in current_anchors:
    #             seed_tform_waypoint = SE3Pose.from_proto(
    #                 current_anchors[waypoint.id].seed_tform_waypoint).to_matrix()
    #             waypoint_object = self.create_waypoint_actors(mat_to_vtk(seed_tform_waypoint), waypoint.id)                
    #             # print(f"seed_tform_waypoint id: {waypoint.id}: {seed_tform_waypoint}")

    #             # waypoint_object.SetUserTransform(mat_to_vtk(seed_tform_waypoint))
    #             self.make_text_actor(waypoint.annotations.name, seed_tform_waypoint[:3, 3])
    #             avg_pos += seed_tform_waypoint[:3, 3]
    #             waypoints_in_anchoring += 1

    #     avg_pos /= waypoints_in_anchoring

    #     # Create VTK objects associated with each edge.
    #     for edge in current_graph.edges:
    #         if edge.id.from_waypoint in current_anchors and edge.id.to_waypoint in current_anchors:
    #             seed_tform_from = SE3Pose.from_proto(
    #                 current_anchors[edge.id.from_waypoint].seed_tform_waypoint).to_matrix()
    #             from_tform_to = SE3Pose.from_proto(edge.from_tform_to).to_matrix()
    #             self.create_edge_object(from_tform_to, seed_tform_from, renderer)

    #     # Create VTK objects associated with each anchored world object.
    #     for anchored_wo in current_anchored_world_objects.values():
    #         # anchored_wo is a tuple of (anchored_world_object, waypoint, fiducial).
    #         (fiducial_object, _) = self.create_fiducial_object(anchored_wo[2], anchored_wo[1], renderer)
    #         seed_tform_fiducial = SE3Pose.from_proto(anchored_wo[0].seed_tform_object).to_matrix()
    #         fiducial_object.SetUserTransform(mat_to_vtk(seed_tform_fiducial))
    #         self.make_text_actor(anchored_wo[0].id, seed_tform_fiducial[:3, 3])

    #     return avg_pos


    def create_graph_objects(self):
        """
        Creates all the VTK objects associated with the graph.
        :paraints: dict from waypoint id to waypoint.
        :param renderer: The VTK renderer
        :return: the average position in world space of all the waypoints.
    m current_graph: the graph to use.
        :param current_waypoint_snapshots: dict from snapshot id to snapshot.
        :param current_waypo    """
        current_graph = self.map.graph
        current_waypoint_snapshots = self.map.waypoint_snapshots
        current_waypoints = self.map.waypoints
        renderer = self.renderer

        # Now, perform a breadth first search of the graph starting from an arbitrary waypoint. Graph nav graphs
        # have no global reference frame. The only thing we can say about waypoints is that they have relative
        # transformations to their neighbors via edges. So the goal is to get the whole graph into a global reference
        # frame centered on some waypoint as the origin.
        queue = [] # queue of (waypoint, homogeneous_tf)
        queue.append((current_graph.waypoints[0], np.eye(4)))
        visited_ids = set()
        # Get the camera in the ballpark of the right position by centering it on the average position of a waypoint.
        avg_pos = np.array([0.0, 0.0, 0.0])

        # Breadth first search.
        while len(queue) > 0:
            # Visit a waypoint.
            curr_element = queue.pop(0)
            curr_waypoint = curr_element[0]
            if curr_waypoint.id in visited_ids:
                continue
            visited_ids.add(curr_waypoint.id)

            # Create VTK actors to represent the waypoint at the waypoint's global pose
            world_tform_current_waypoint = curr_element[1]
            self.create_point_cloud_object(world_tform_current_waypoint, curr_waypoint.id)
            self.make_sphere_actor(world_tform_current_waypoint[:3,3], curr_waypoint.id)
            self.make_text_actor(curr_waypoint.annotations.name, world_tform_current_waypoint[:3,3])

            # For each fiducial in the waypoint's snapshot, add an object at the world pose of that fiducial.
            if (curr_waypoint.snapshot_id in current_waypoint_snapshots):
                snapshot = current_waypoint_snapshots[curr_waypoint.snapshot_id]
                for fiducial in snapshot.objects:
                    if fiducial.HasField('apriltag_properties'):
                        (fiducial_object, curr_wp_tform_fiducial) = self.create_fiducial_object(
                            fiducial, curr_waypoint, renderer)
                        world_tform_fiducial = np.dot(world_tform_current_waypoint,
                                                    vtk_to_mat(curr_wp_tform_fiducial))
                        fiducial_object.SetUserTransform(mat_to_vtk(world_tform_fiducial))
                        self.make_text_actor(str(fiducial.apriltag_properties.tag_id), world_tform_fiducial[:3, 3])

            # Now, for each edge, walk along the edge and concatenate the transform to the neighbor.
            for edge in current_graph.edges:
                # If the edge is directed away from us...
                if edge.id.from_waypoint == curr_waypoint.id and edge.id.to_waypoint not in visited_ids:
                    current_waypoint_tform_to_waypoint = SE3Pose.from_proto(
                        edge.from_tform_to).to_matrix()
                    world_tform_to_wp = self.create_edge_object(current_waypoint_tform_to_waypoint,
                                                        world_tform_current_waypoint, renderer)
                    # Add the neighbor to the queue.
                    queue.append((current_waypoints[edge.id.to_waypoint], world_tform_to_wp))
                    avg_pos += world_tform_to_wp[:3, 3]
                # If the edge is directed toward us...
                elif edge.id.to_waypoint == curr_waypoint.id and edge.id.from_waypoint not in visited_ids:
                    current_waypoint_tform_from_waypoint = (SE3Pose.from_proto(
                        edge.from_tform_to).inverse()).to_matrix()
                    world_tform_from_wp = self.create_edge_object(current_waypoint_tform_from_waypoint,
                                                            world_tform_current_waypoint, renderer)
                    # Add the neighbor to the queue.
                    queue.append((current_waypoints[edge.id.from_waypoint], world_tform_from_wp))
                    avg_pos += world_tform_from_wp[:3, 3]

        # Compute the average waypoint position to place the camera appropriately.
        avg_pos /= len(current_waypoints)
        return avg_pos



# vtkInteractorStyle is a base class that provides event-driven interface 
# to the rendering window (defines trackball mode), which supports vtkRenderWindowInteractor. 
# vtkRenderWindowInteractor implements platform dependent key/mouse routing and timer control, 
# which forwards events in a neutral form to vtkInteractorStyle.
class SpotCommandInteractorStyle(vtkInteractorStyleTerrain):
    """ 
    Custom Interactor that allows the user to click on an actor and highlight it with a silhouette.
    """
    def __init__(self, silhouette=None, silhouetteActor=None):
        super().__init__()
        self.AddObserver("KeyPressEvent", self.onKeyPressEvent)
        self.LastPickedActor = None
        self.Silhouette = silhouette
        self.SilhouetteActor = silhouetteActor
        self.observers = {'space': []}

    def add_external_observer(self, event, callback):
        if event in self.observers:
            self.observers[event].append(callback)

    def notify_external_observers(self, event, data):
        for callback in self.observers.get(event, []):
            callback(data)

    
    def onKeyPressEvent(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == 'space':
            click_x, click_y = self.GetInteractor().GetEventPosition()

            #  Pick actor at this location.
            picker = vtkPropPicker()
            picker.Pick(click_x, click_y, 0, self.GetDefaultRenderer())
            actor = picker.GetActor()

            if actor:
                print(f"Actor selected: {actor.waypoint_id}")   
                self.notify_external_observers(key, actor.waypoint_id)         

            self.LastPickedActor = actor

            # If we picked something before, remove the silhouette actor and
            # generate a new one.
            if self.LastPickedActor:
                self.GetDefaultRenderer().RemoveActor(self.SilhouetteActor)

                # Highlight the picked actor by generating a silhouette
                self.Silhouette.SetInputData(self.LastPickedActor.GetMapper().GetInput())
                self.GetDefaultRenderer().AddActor(self.SilhouetteActor)

            # render the image
            self.GetDefaultRenderer().GetRenderWindow().Render()
        #  Forward events
        self.OnKeyPress()
        return

   
def main():

    path, anchoring = get_program_parameters()
    spot_map = SpotMap(path)
    vtk_engine = VTKEngine(spot_map)
    vtk_engine.start()


if __name__ == "__main__":
    main()