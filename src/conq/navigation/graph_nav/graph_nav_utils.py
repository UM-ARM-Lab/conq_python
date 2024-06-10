from bosdyn.api import robot_state_pb2
from bosdyn.api.graph_nav import map_pb2, graph_nav_pb2, nav_pb2
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.power import power_on_motors, safe_power_off_motors, PowerClient
from bosdyn.client.exceptions import ResponseError
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
import bosdyn.client.util
import bosdyn.client.lease
from bosdyn.client.lease import LeaseClient
from bosdyn.client.frame_helpers import get_odom_tform_body

# misc
import os
import time
import math
from dotenv import load_dotenv

class GraphNav:
    # Constructor that will initialize everything that is needed for navigating to waypoints
    def __init__(self, robot, is_debug=False):
        # Load the graph_nav graph file from the environment variable GRAPH_NAV_GRAPH_FILEPATH
        load_dotenv('.env.local')
        self._upload_filepath = os.getenv('GRAPH_NAV_GRAPH_FILEPATH')
        self.location_cache = os.getenv('GRAPH_NAV_LOCATION_CACHE') 

        # Save the robot as a private member variable and sync with the robot
        self._robot = robot

        self._is_debug = is_debug
        
        self._robot.time_sync.wait_for_sync()

        # Member variables

        # Clients
        self._robot_command_client = self._robot.ensure_client(
            RobotCommandClient.default_service_name)
        self._robot_state_client = self._robot.ensure_client(RobotStateClient.default_service_name)

        # Create the client for the Graph Nav main service.
        self._graph_nav_client = self._robot.ensure_client(GraphNavClient.default_service_name)

        # Create the power client to ensure that all motors are functioning accordingly
        self._power_client = self._robot.ensure_client(PowerClient.default_service_name)

        # Update the robot's power state
        power_state = self._robot_state_client.get_robot_state().power_state
        self._started_powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        self._powered_on = self._started_powered_on

        # Graphs
        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  #maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()

        # Init the graph for spot to use
        self._init_graph()        

    #### PRIVATE UTILITY FUNCTIONS

    # This function uploads the graph file from your computer and loads the waypoint names
    def _init_graph(self):
        # Upload the graph from your local machine
        self._upload_graph_and_snapshots()
        # Load the name tables
        self._list_graph_waypoint_and_edge_ids()

        #self.localize()

    # This function uploads the graph file from your computer to spot
    def _upload_graph_and_snapshots(self):
        if(self._is_debug):
            print('Loading the graph from disk into local storage...')
        with open(self._upload_filepath + '/graph', 'rb') as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
            if(self._is_debug):
                print(
                    f'Loaded graph has {len(self._current_graph.waypoints)} waypoints and {self._current_graph.edges} edges'
                )
        for waypoint in self._current_graph.waypoints:
            # Load the waypoint snapshots from disk.
            with open(f'{self._upload_filepath}/waypoint_snapshots/{waypoint.snapshot_id}',
                      'rb') as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        for edge in self._current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            # Load the edge snapshots from disk.
            with open(f'{self._upload_filepath}/edge_snapshots/{edge.snapshot_id}',
                      'rb') as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        if(self._is_debug):
            print('Uploading the graph and snapshots to the robot...')
        true_if_empty = not len(self._current_graph.anchoring.anchors)
        response = self._graph_nav_client.upload_graph(graph=self._current_graph,
                                                       generate_new_anchoring=true_if_empty)
        # Upload the snapshots to the robot.
        for snapshot_id in response.unknown_waypoint_snapshot_ids:
            waypoint_snapshot = self._current_waypoint_snapshots[snapshot_id]
            self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
            if(self._is_debug):
                print(f'Uploaded {waypoint_snapshot.id}')
        for snapshot_id in response.unknown_edge_snapshot_ids:
            edge_snapshot = self._current_edge_snapshots[snapshot_id]
            self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
            if(self._is_debug):
                print(f'Uploaded {edge_snapshot.id}')

        # The upload is complete! Check that the robot is localized to the graph,
        # and if it is not, prompt the user to localize the robot before attempting
        # any navigation commands.
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            if(self._is_debug):
                print('\n')
                print(
                    'Graph map upload is complete! However the robot is not localized in the map')
    
    # Convert the given waypoint name into 
    def _find_unique_waypoint_id(self, short_code, graph, name_to_id):
        if graph is None: 
            if(self._is_debug):
                print(
                    'Waypoints have not been loaded properly'
                )
            return

        if len(short_code) != 2:
            # Not a short code, check if it is an annotation name (instead of the waypoint id).
            if short_code in name_to_id:
                # Short code is a waypoint's annotation name. Check if it is paired with a unique waypoint id.
                if name_to_id[short_code] is not None:
                    # Has an associated waypoint id!
                    return name_to_id[short_code]
                else:
                    if(self._is_debug):
                        print(
                            f'The waypoint name {short_code} is used for multiple different unique waypoints. Please use '
                            f'the waypoint id.')
                    return None
            # Also not a waypoint annotation name, so we will operate under the assumption that it is a
            # unique waypoint id.
            return short_code

        ret = short_code
        for waypoint in graph.waypoints:
            if short_code == self._id_to_short_code(waypoint.id):
                if ret != short_code:
                    return short_code  # Multiple waypoints with same short code.
                ret = waypoint.id
        return ret

    # This function will toggle the motor power on the robot to ensure that the robot will be able to move when trying to navigate
    def toggle_power(self, should_power_on):
        is_powered_on = self._check_is_powered_on()
        if not is_powered_on and should_power_on:
            # Power on the robot up before navigating when it is in a powered-off state.
            power_on_motors(self._power_client)
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
            safe_power_off_motors(self._robot_command_client, self._robot_state_client)
        else:
            # Return the current power state without change.
            return is_powered_on
        # Update the locally stored power state.
        self._check_is_powered_on()
        return self._powered_on
    
    # This function converts a given long string to a shortened string
    def _id_to_short_code(self, id):
        tokens = id.split('-')
        if len(tokens) > 2:
            return f'{tokens[0][0]}{tokens[1][0]}'
        return None
    
    # This function ensures the robot is powered on
    def _check_is_powered_on(self):
        power_state = self._robot_state_client.get_robot_state().power_state
        self._powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        return self._powered_on
    
    # This function checks to see if the robot successfully navigated to its desired waypoint
    def _check_success(self, command_id=-1):
        if command_id == -1:
            # No command, so we have no status to check.
            return False
        status = self._graph_nav_client.navigation_feedback(command_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            if(self._is_debug):
                print('Robot got lost when navigating the route, the robot will now sit down.')
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            if(self._is_debug):
                print('Robot got stuck when navigating the route, the robot will now sit down.')
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            if(self._is_debug):
                print('Robot is impaired.')
            return True
        else:
            # Navigation command is not complete yet.
            return False
        
    # This function connects the list of waypoint "names" to their unique IDs
    def _list_graph_waypoint_and_edge_ids(self):
        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print('Empty graph.')
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = self._update_waypoints_and_edges(
            graph, localization_id)
    
    # This function updates the waypoints and their respective connections
    def _update_waypoints_and_edges(self, graph, localization_id, do_print=True):
        name_to_id = dict()
        edges = dict()

        short_code_to_count = {}
        waypoint_to_timestamp = []
        for waypoint in graph.waypoints:
            # Determine the timestamp that this waypoint was created at.
            timestamp = -1.0
            try:
                timestamp = waypoint.annotations.creation_time.seconds + waypoint.annotations.creation_time.nanos / 1e9
            except:
                # Must be operating on an older graph nav map, since the creation_time is not
                # available within the waypoint annotations message.
                pass
            waypoint_to_timestamp.append((waypoint.id, timestamp, waypoint.annotations.name))

            # Determine how many waypoints have the same short code.
            short_code = self._id_to_short_code(waypoint.id)
            if short_code not in short_code_to_count:
                short_code_to_count[short_code] = 0
            short_code_to_count[short_code] += 1

            # Add the annotation name/id into the current dictionary.
            waypoint_name = waypoint.annotations.name
            if waypoint_name:
                if waypoint_name in name_to_id:
                    # Waypoint name is used for multiple different waypoints, so set the waypoint id
                    # in this dictionary to None to avoid confusion between two different waypoints.
                    name_to_id[waypoint_name] = None
                else:
                    # First time we have seen this waypoint annotation name. Add it into the dictionary
                    # with the respective waypoint unique id.
                    name_to_id[waypoint_name] = waypoint.id

        # Sort the set of waypoints by their creation timestamp. If the creation timestamp is unavailable,
        # fallback to sorting by annotation name.
        waypoint_to_timestamp = sorted(waypoint_to_timestamp, key=lambda x: (x[1], x[2]))

        # Print out the waypoints name, id, and short code in an ordered sorted by the timestamp from
        # when the waypoint was created.
        if do_print:
            if(self._is_debug):
                print(f'{len(graph.waypoints):d} waypoints:')
            for waypoint in waypoint_to_timestamp:
                self._pretty_print_waypoints(waypoint[0], waypoint[2], short_code_to_count, localization_id)

        for edge in graph.edges:
            if edge.id.to_waypoint in edges:
                if edge.id.from_waypoint not in edges[edge.id.to_waypoint]:
                    edges[edge.id.to_waypoint].append(edge.id.from_waypoint)
            else:
                edges[edge.id.to_waypoint] = [edge.id.from_waypoint]
            if do_print:
                if(self._is_debug):
                    print(f'(Edge) from waypoint {edge.id.from_waypoint} to waypoint {edge.id.to_waypoint} '
                    f'(cost {edge.annotations.cost.value})')

        return name_to_id, edges

    # This function prints a waypoint in a readable way
    def _pretty_print_waypoints(self, waypoint_id, waypoint_name, short_code_to_count, localization_id):
        short_code = self._id_to_short_code(waypoint_id)
        if short_code is None or short_code_to_count[short_code] != 1:
            short_code = '  '  # If the short code is not valid/unique, don't show it.

        waypoint_symbol = '->' if localization_id == waypoint_id else '  '
        if(self._is_debug):
            print(
                f'{waypoint_symbol} Waypoint name: {waypoint_name} id: {waypoint_id} short code: {short_code}'
            )

    # This function "should" update the robots pose while having turned the robot on and off
    def _get_localization_state(self):
        """Get the current localization and state of the robot."""
        state = self._graph_nav_client.get_localization_state(request_gps_state=self.use_gps)
        print(f'Got localization: \n{state.localization}')
        odom_tform_body = get_odom_tform_body(state.robot_kinematics.transforms_snapshot)
        print(f'Got robot state in kinematic odometry frame: \n{odom_tform_body}')
        if self.use_gps:
            print(f'GPS info:\n{state.gps}')

    # This is a higher level function for localizing in an entire graph
    def localize(self, waypoint_num):
        # Read the cached value for an estimate as to where we started in the map
        self._set_initial_localization_waypoint(waypoint_num)
                
                



    # This uses the Scan Match algorithm that the spok sdk has for using slam to find an estimated localization
    # This happens because in the set_localization function the argument FIDUCIAL_INIT_NO_FIDUCIAL is passed which signals that spot should use slam to help localize itself
    def _set_initial_localization_waypoint(self, waypoint_id):
        print("Localizing to waypoint: " + str(waypoint_id))
        name = "waypoint_" + str(waypoint_id)
        # Take the first argument as the localization waypoint.
        destination_waypoint = self._find_unique_waypoint_id(
            name, self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the unique waypoint id.
            return False

        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an initial localization to the specified waypoint as the identity.
        localization = nav_pb2.Localization()
        localization.waypoint_id = destination_waypoint
        localization.waypoint_tform_body.rotation.w = 1.0
        try:    
            self._graph_nav_client.set_localization(
                initial_guess_localization=localization,
                # It's hard to get the pose perfect, search +/-20 deg and +/-20cm (0.2m).
                max_distance=4.0,
                max_yaw=180.0 * math.pi / 180.0,
                fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
                ko_tform_body=current_odom_tform_body)
            return True
        except bosdyn.client.exceptions.TimedOutError:
            print("Localizing to waypoint: " + str(waypoint_id) + " Failed")
            return False

    #### PUBLIC MEMBER FUNCTIONS ####

    # This function will tell spot to go to a specific waypoint name, waypoint names are of the format waypoint_{id_number}
    def navigate_to(self, waypoint_number, sit_down_after_reached=True):
        print(f'Navigating to {waypoint_number}...')
        destination_waypoint = self._find_unique_waypoint_id(
            waypoint_number, self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        if not self.toggle_power(should_power_on=True):
            if(self._is_debug):
                print('Failed to power on the robot, and cannot complete navigate to request.')
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
                print(f'Error while navigating {e}')
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)

        # Power off the robot if appropriate.
        if self._powered_on and not self._started_powered_on and sit_down_after_reached:
            # Sit the robot down + power off after the navigation command is complete.
            self.toggle_power(should_power_on=False)

    # This function saves the current location so that spot can localize again on startup
    def save_current_location(self):
        print(self.location_cache)
        with open(self.location_cache, 'w') as cache:
            print("instide")
            state = self._graph_nav_client.get_localization_state()
            name = list(self._current_annotation_name_to_wp_id.keys())[list(self._current_annotation_name_to_wp_id.values()).index(state.localization.waypoint_id)]
            print(f'Got waypoint: \n{name}')
            cache.write(name)

    # Get the size of the current graph
    def get_graph_size(self):
        return len(self._current_annotation_name_to_wp_id)
        

# #Setup and authenticate the robot.
# sdk = bosdyn.client.create_standard_sdk('GraphNavClient')
# robot = sdk.create_robot('192.168.80.3')
# bosdyn.client.util.authenticate(robot) 

# lease_client = robot.ensure_client(LeaseClient.default_service_name)

# lease_client.take()

# with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
#     gn = GraphNav(robot)
#     gn.navigate_to('waypoint_0')
#     gn.save_current_location()

    
