import logging

import threading

import time

from dotenv import load_dotenv 

# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""WASD driving of robot."""
import curses
import io
import logging
import math
import os
import signal
import sys
import threading
import time
from collections import OrderedDict

from PIL import Image, ImageEnhance

import bosdyn.api.basic_command_pb2 as basic_command_pb2
import bosdyn.api.power_pb2 as PowerServiceProto
# import bosdyn.api.robot_command_pb2 as robot_command_pb2
import bosdyn.api.robot_state_pb2 as robot_state_proto
import bosdyn.api.spot.robot_command_pb2 as spot_command_pb2
import bosdyn.client.util
from bosdyn.api import geometry_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.async_tasks import AsyncGRPCTask, AsyncPeriodicQuery, AsyncTasks
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.time_sync import TimeSyncError
from bosdyn.util import duration_str, format_metric, secs_to_hms

LOGGER = logging.getLogger()

VELOCITY_BASE_SPEED = 0.5  # m/s
VELOCITY_BASE_ANGULAR = 0.8  # rad/sec
VELOCITY_CMD_DURATION = 0.6  # seconds
COMMAND_INPUT_RATE = 0.1



def _grpc_or_log(desc, thunk):
    try:
        return thunk()
    except (ResponseError, RpcError) as err:
        LOGGER.error('Failed %s: %s', desc, err)


def _image_to_ascii(image, new_width):
    """Convert an rgb image to an ASCII 'image' that can be displayed in a terminal."""

    ASCII_CHARS = '@#S%?*+;:,.'

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(0.8)

    # Scaling image before rotation by 90 deg.
    scaled_rot_height = new_width
    original_rot_width, original_rot_height = image.size
    scaled_rot_width = (original_rot_width * scaled_rot_height) // original_rot_height
    # Scaling rotated width (height, after rotation) by half because ASCII chars
    #  in terminal seem about 2x as tall as wide.
    image = image.resize((scaled_rot_width // 2, scaled_rot_height))

    # Rotate image 90 degrees, then convert to grayscale.
    image = image.transpose(Image.ROTATE_270)
    image = image.convert('L')

    def _pixel_char(pixel_val):
        return ASCII_CHARS[pixel_val * len(ASCII_CHARS) // 256]

    img = []
    row = [' '] * new_width
    last_col = new_width - 1
    for idx, pixel_char in enumerate(_pixel_char(val) for val in image.getdata()):
        idx_row = idx % new_width
        row[idx_row] = pixel_char
        if idx_row == last_col:
            img.append(''.join(row))
    return img


class ExitCheck(object):
    """A class to help exiting a loop, also capturing SIGTERM to exit the loop."""

    def __init__(self):
        self._kill_now = False
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        signal.signal(signal.SIGINT, self._sigterm_handler)

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        return False

    def _sigterm_handler(self, _signum, _frame):
        self._kill_now = True

    def request_exit(self):
        """Manually trigger an exit (rather than sigterm/sigint)."""
        self._kill_now = True

    @property
    def kill_now(self):
        """Return the status of the exit checker indicating if it should exit."""
        return self._kill_now


class CursesHandler(logging.Handler):
    """logging handler which puts messages into the curses interface"""

    def __init__(self, wasd_interface):
        super(CursesHandler, self).__init__()
        self._wasd_interface = wasd_interface

    def emit(self, record):
        msg = record.getMessage()
        msg = msg.replace('\n', ' ').replace('\r', '')
        self._wasd_interface.add_message(f'{record.levelname:s} {msg:s}')


class AsyncRobotState(AsyncPeriodicQuery):
    """Grab robot state."""

    def __init__(self, robot_state_client):
        super(AsyncRobotState, self).__init__('robot_state', robot_state_client, LOGGER,
                                              period_sec=0.2)

    def _start_query(self):
        return self._client.get_robot_state_async()


class AsyncImageCapture(AsyncGRPCTask):
    """Grab camera images from the robot."""

    def __init__(self, robot):
        super(AsyncImageCapture, self).__init__()
        self._image_client = robot.ensure_client(ImageClient.default_service_name)
        self._ascii_image = None
        self._video_mode = False
        self._should_take_image = False

    @property
    def ascii_image(self):
        """Return the latest captured image as ascii."""
        return self._ascii_image

    def toggle_video_mode(self):
        """Toggle whether doing continuous image capture."""
        self._video_mode = not self._video_mode

    def take_image(self):
        """Request a one-shot image."""
        self._should_take_image = True

    def _start_query(self):
        self._should_take_image = False
        source_name = 'frontright_fisheye_image'
        return self._image_client.get_image_from_sources_async([source_name])

    def _should_query(self, now_sec):  # pylint: disable=unused-argument
        return self._video_mode or self._should_take_image

    def _handle_result(self, result):
        import io
        image = Image.open(io.BytesIO(result[0].shot.image.data))
        self._ascii_image = _image_to_ascii(image, new_width=70)

    def _handle_error(self, exception):
        LOGGER.exception('Failure getting image: %s', exception)


class WasdInterface(object):
    """A curses interface for driving the robot."""

    def __init__(self, robot):
        self._robot = robot
        # Create clients -- do not use the for communication yet.
        self._lease_client = robot.ensure_client(LeaseClient.default_service_name)
        try:
            self._estop_client = self._robot.ensure_client(EstopClient.default_service_name)
            self._estop_endpoint = EstopEndpoint(self._estop_client, 'GNClient', 9.0)
        except:
            # Not the estop.
            self._estop_client = None
            self._estop_endpoint = None
        self._power_client = robot.ensure_client(PowerClient.default_service_name)
        self._robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self._robot_state_task = AsyncRobotState(self._robot_state_client)
        self._image_task = AsyncImageCapture(robot)
        self._async_tasks = AsyncTasks([self._robot_state_task, self._image_task])
        self._lock = threading.Lock()
        self._command_dictionary = {
            27: self._stop,  # ESC key
            ord('\t'): self._quit_program,
            ord('T'): self._toggle_time_sync,
            ord(' '): self._toggle_estop,
            ord('r'): self._self_right,
            ord('P'): self._toggle_power,
            ord('p'): self._toggle_power,
            ord('v'): self._sit,
            ord('b'): self._battery_change_pose,
            ord('f'): self._stand,
            ord('w'): self._move_forward,
            ord('s'): self._move_backward,
            ord('a'): self._strafe_left,
            ord('d'): self._strafe_right,
            ord('q'): self._turn_left,
            ord('e'): self._turn_right,
            ord('I'): self._image_task.take_image,
            ord('O'): self._image_task.toggle_video_mode,
            ord('u'): self._unstow,
            ord('j'): self._stow,
            ord('l'): self._toggle_lease
        }
        self._locked_messages = ['', '', '']  # string: displayed message for user
        self._estop_keepalive = None
        self._exit_check = None

        # Stuff that is set in start()
        self._robot_id = None
        self._lease_keepalive = None

    def start(self):
        """Begin communication with the robot."""
        # Construct our lease keep-alive object, which begins RetainLease calls in a thread.
        self._lease_keepalive = LeaseKeepAlive(self._lease_client, must_acquire=True,
                                               return_at_exit=True)

        self._robot_id = self._robot.get_id()

    def shutdown(self):
        """Release control of robot as gracefully as possible."""
        LOGGER.info('Shutting down WasdInterface.')
        if self._estop_keepalive:
            # This stops the check-in thread but does not stop the robot.
            self._estop_keepalive.shutdown()
        if self._lease_keepalive:
            self._lease_keepalive.shutdown()

    def flush_and_estop_buffer(self, stdscr):
        """Manually flush the curses input buffer but trigger any estop requests (space)"""
        key = ''
        while key != -1:
            key = stdscr.getch()
            if key == ord(' '):
                self._toggle_estop()

    def add_message(self, msg_text):
        """Display the given message string to the user in the curses interface."""
        with self._lock:
            self._locked_messages = [msg_text] + self._locked_messages[:-1]

    def message(self, idx):
        """Grab one of the 3 last messages added."""
        with self._lock:
            return self._locked_messages[idx]

    @property
    def robot_state(self):
        """Get latest robot state proto."""
        return self._robot_state_task.proto

    def drive(self, stdscr):
        """User interface to control the robot via the passed-in curses screen interface object."""
        with ExitCheck() as self._exit_check:
            curses_handler = CursesHandler(self)
            curses_handler.setLevel(logging.INFO)
            LOGGER.addHandler(curses_handler)

            stdscr.nodelay(True)  # Don't block for user input.
            stdscr.resize(26, 140)
            stdscr.refresh()

            # for debug
            curses.echo()

            try:
                while not self._exit_check.kill_now:
                    self._async_tasks.update()
                    self._drive_draw(stdscr, self._lease_keepalive)

                    try:
                        cmd = stdscr.getch()
                        # Do not queue up commands on client
                        self.flush_and_estop_buffer(stdscr)
                        self._drive_cmd(cmd)
                        time.sleep(COMMAND_INPUT_RATE)
                    except Exception:
                        # On robot command fault, sit down safely before killing the program.
                        self._safe_power_off()
                        time.sleep(2.0)
                        raise

            finally:
                LOGGER.removeHandler(curses_handler)

    def _drive_draw(self, stdscr, lease_keep_alive):
        """Draw the interface screen at each update."""
        stdscr.clear()  # clear screen
        stdscr.resize(26, 140)
        stdscr.addstr(0, 0, f'{self._robot_id.nickname:20s} {self._robot_id.serial_number}')
        stdscr.addstr(1, 0, self._lease_str(lease_keep_alive))
        stdscr.addstr(2, 0, self._battery_str())
        stdscr.addstr(3, 0, self._estop_str())
        stdscr.addstr(4, 0, self._power_state_str())
        stdscr.addstr(5, 0, self._time_sync_str())
        for i in range(3):
            stdscr.addstr(7 + i, 2, self.message(i))
        stdscr.addstr(10, 0, 'Commands: [TAB]: quit                               ')
        stdscr.addstr(11, 0, '          [T]: Time-sync, [SPACE]: Estop, [P]: Power')
        stdscr.addstr(12, 0, '          [I]: Take image, [O]: Video mode          ')
        stdscr.addstr(13, 0, '          [f]: Stand, [r]: Self-right               ')
        stdscr.addstr(14, 0, '          [v]: Sit, [b]: Battery-change             ')
        stdscr.addstr(15, 0, '          [wasd]: Directional strafing              ')
        stdscr.addstr(16, 0, '          [qe]: Turning, [ESC]: Stop                ')
        stdscr.addstr(17, 0, '          [l]: Return/Acquire lease                 ')
        stdscr.addstr(18, 0, '')

        # print as many lines of the image as will fit on the curses screen
        if self._image_task.ascii_image is not None:
            max_y, _max_x = stdscr.getmaxyx()
            for y_i, img_line in enumerate(self._image_task.ascii_image):
                if y_i + 17 >= max_y:
                    break

                stdscr.addstr(y_i + 17, 0, img_line)

        stdscr.refresh()

    def _drive_cmd(self, key):
        """Run user commands at each update."""
        try:
            cmd_function = self._command_dictionary[key]
            cmd_function()

        except KeyError:
            if key and key != -1 and key < 256:
                self.add_message(f'Unrecognized keyboard command: \'{chr(key)}\'')

    def _try_grpc(self, desc, thunk):
        try:
            return thunk()
        except (ResponseError, RpcError, LeaseBaseError) as err:
            self.add_message(f'Failed {desc}: {err}')
            return None

    def _try_grpc_async(self, desc, thunk):

        def on_future_done(fut):
            try:
                fut.result()
            except (ResponseError, RpcError, LeaseBaseError) as err:
                self.add_message(f'Failed {desc}: {err}')
                return None

        future = thunk()
        future.add_done_callback(on_future_done)

    def _quit_program(self):
        self._sit()
        if self._exit_check is not None:
            self._exit_check.request_exit()

    def _toggle_time_sync(self):
        if self._robot.time_sync.stopped:
            self._robot.start_time_sync()
        else:
            self._robot.time_sync.stop()

    def _toggle_estop(self):
        """toggle estop on/off. Initial state is ON"""
        if self._estop_client is not None and self._estop_endpoint is not None:
            if not self._estop_keepalive:
                self._estop_keepalive = EstopKeepAlive(self._estop_endpoint)
            else:
                self._try_grpc('stopping estop', self._estop_keepalive.stop)
                self._estop_keepalive.shutdown()
                self._estop_keepalive = None

    def _toggle_lease(self):
        """toggle lease acquisition. Initial state is acquired"""
        if self._lease_client is not None:
            if self._lease_keepalive is None:
                self._lease_keepalive = LeaseKeepAlive(self._lease_client, must_acquire=True,
                                                       return_at_exit=True)
            else:
                self._lease_keepalive.shutdown()
                self._lease_keepalive = None

    def _start_robot_command(self, desc, command_proto, end_time_secs=None):

        def _start_command():
            self._robot_command_client.robot_command(command=command_proto,
                                                     end_time_secs=end_time_secs)

        self._try_grpc(desc, _start_command)

    def _self_right(self):
        self._start_robot_command('self_right', RobotCommandBuilder.selfright_command())

    def _battery_change_pose(self):
        # Default HINT_RIGHT, maybe add option to choose direction?
        self._start_robot_command(
            'battery_change_pose',
            RobotCommandBuilder.battery_change_pose_command(
                dir_hint=basic_command_pb2.BatteryChangePoseCommand.Request.HINT_RIGHT))

    def _sit(self):
        self._start_robot_command('sit', RobotCommandBuilder.synchro_sit_command())

    def _stand(self):
        self._start_robot_command('stand', RobotCommandBuilder.synchro_stand_command())

    def _move_forward(self):
        self._velocity_cmd_helper('move_forward', v_x=VELOCITY_BASE_SPEED)

    def _move_backward(self):
        self._velocity_cmd_helper('move_backward', v_x=-VELOCITY_BASE_SPEED)

    def _strafe_left(self):
        self._velocity_cmd_helper('strafe_left', v_y=VELOCITY_BASE_SPEED)

    def _strafe_right(self):
        self._velocity_cmd_helper('strafe_right', v_y=-VELOCITY_BASE_SPEED)

    def _turn_left(self):
        self._velocity_cmd_helper('turn_left', v_rot=VELOCITY_BASE_ANGULAR)

    def _turn_right(self):
        self._velocity_cmd_helper('turn_right', v_rot=-VELOCITY_BASE_ANGULAR)

    def _stop(self):
        self._start_robot_command('stop', RobotCommandBuilder.stop_command())

    def _velocity_cmd_helper(self, desc='', v_x=0.0, v_y=0.0, v_rot=0.0):
        self._start_robot_command(
            desc, RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot),
            end_time_secs=time.time() + VELOCITY_CMD_DURATION)

    def _stow(self):
        self._start_robot_command('stow', RobotCommandBuilder.arm_stow_command())

    def _unstow(self):
        self._start_robot_command('stow', RobotCommandBuilder.arm_ready_command())

    def _return_to_origin(self):
        self._start_robot_command(
            'fwd_and_rotate',
            RobotCommandBuilder.synchro_se2_trajectory_point_command(
                goal_x=0.0, goal_y=0.0, goal_heading=0.0, frame_name=ODOM_FRAME_NAME, params=None,
                body_height=0.0, locomotion_hint=spot_command_pb2.HINT_SPEED_SELECT_TROT),
            end_time_secs=time.time() + 20)

    def _take_ascii_image(self):
        source_name = 'frontright_fisheye_image'
        image_response = self._image_client.get_image_from_sources([source_name])
        image = Image.open(io.BytesIO(image_response[0].shot.image.data))
        ascii_image = self._ascii_converter.convert_to_ascii(image, new_width=70)
        self._last_image_ascii = ascii_image

    def _toggle_ascii_video(self):
        if self._video_mode:
            self._video_mode = False
        else:
            self._video_mode = True


    def _toggle_power(self):
        power_state = self._power_state()
        if power_state is None:
            self.add_message('Could not toggle power because power state is unknown')
            return

        if power_state == robot_state_proto.PowerState.STATE_OFF:
            self._try_grpc_async('powering-on', self._request_power_on)
        else:
            self._try_grpc('powering-off', self._safe_power_off)

    def _request_power_on(self):
        request = PowerServiceProto.PowerCommandRequest.REQUEST_ON
        return self._power_client.power_command_async(request)

    def _safe_power_off(self):
        self._start_robot_command('safe_power_off', RobotCommandBuilder.safe_power_off_command())

    def _power_state(self):
        state = self.robot_state
        if not state:
            return None
        return state.power_state.motor_power_state

    def _lease_str(self, lease_keep_alive):
        if lease_keep_alive is None:
            alive = 'STOPPED'
            lease = 'RETURNED'
        else:
            try:
                _lease = lease_keep_alive.lease_wallet.get_lease()
                lease = f'{_lease.lease_proto.resource}:{_lease.lease_proto.sequence}'
            except bosdyn.client.lease.Error:
                lease = '...'
            if lease_keep_alive.is_alive():
                alive = 'RUNNING'
            else:
                alive = 'STOPPED'
        return f'Lease {lease} THREAD:{alive}'

    def _power_state_str(self):
        power_state = self._power_state()
        if power_state is None:
            return ''
        state_str = robot_state_proto.PowerState.MotorPowerState.Name(power_state)
        return f'Power: {state_str[6:]}'  # get rid of STATE_ prefix

    def _estop_str(self):
        if not self._estop_client:
            thread_status = 'NOT ESTOP'
        else:
            thread_status = 'RUNNING' if self._estop_keepalive else 'STOPPED'
        estop_status = '??'
        state = self.robot_state
        if state:
            for estop_state in state.estop_states:
                if estop_state.type == estop_state.TYPE_SOFTWARE:
                    estop_status = estop_state.State.Name(estop_state.state)[6:]  # s/STATE_//
                    break
        return f'Estop {estop_status} (thread: {thread_status})'

    def _time_sync_str(self):
        if not self._robot.time_sync:
            return 'Time sync: (none)'
        if self._robot.time_sync.stopped:
            status = 'STOPPED'
            exception = self._robot.time_sync.thread_exception
            if exception:
                status = f'{status} Exception: {exception}'
        else:
            status = 'RUNNING'
        try:
            skew = self._robot.time_sync.get_robot_clock_skew()
            if skew:
                skew_str = f'offset={duration_str(skew)}'
            else:
                skew_str = '(Skew undetermined)'
        except (TimeSyncError, RpcError) as err:
            skew_str = f'({err})'
        return f'Time sync: {status} {skew_str}'

    def _battery_str(self):
        if not self.robot_state:
            return ''
        battery_state = self.robot_state.battery_states[0]
        status = battery_state.Status.Name(battery_state.status)
        status = status[7:]  # get rid of STATUS_ prefix
        if battery_state.charge_percentage.value:
            bar_len = int(battery_state.charge_percentage.value) // 10
            bat_bar = f'|{"=" * bar_len}{" " * (10 - bar_len)}|'
        else:
            bat_bar = ''
        time_left = ''
        if battery_state.estimated_runtime:
            time_left = f'({secs_to_hms(battery_state.estimated_runtime.seconds)})'
        return f'Battery: {status}{bat_bar} {time_left}'


def _setup_logging(verbose):
    """Log to file at debug level, and log to console at INFO or DEBUG (if verbose).

    Returns the stream/console logger so that it can be removed when in curses mode.
    """
    LOGGER.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Save log messages to file wasd.log for later debugging.
    file_handler = logging.FileHandler('wasd.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    LOGGER.addHandler(file_handler)

    # The stream handler is useful before and after the application is in curses-mode.
    if verbose:
        stream_level = logging.DEBUG
    else:
        stream_level = logging.INFO

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(stream_level)
    stream_handler.setFormatter(log_formatter)
    LOGGER.addHandler(stream_handler)
    return stream_handler


def mainWASD():
    """Command-line interface."""
    # Create robot object.
    sdk = create_standard_sdk('WASDClient')
    robot = sdk.create_robot('192.168.80.3')
    try:
        bosdyn.client.util.authenticate(robot)
    except RpcError as err:
        LOGGER.error('Failed to communicate with robot: %s', err)
        return False
    print("starting wasd interface")
    wasd_interface = WasdInterface(robot)
    try:
        wasd_interface.start()
        print("did start")
    except (ResponseError, RpcError) as err:
        LOGGER.error('Failed to initialize robot communication: %s', err)
        return False
    try:
        try:
            # Prevent curses from introducing a 1-second delay for ESC key
            os.environ.setdefault('ESCDELAY', '0')
            # Run wasd interface in curses mode, then restore terminal config.
            curses.wrapper(wasd_interface.drive)
        finally:
            # Restore stream handler to show any exceptions or final messages.
            print("Something severly messed up")
    except Exception as e:  
        LOGGER.error('WASD has thrown an error: [%r] %s', e, e)
    finally:
        # Do any final cleanup steps.
        wasd_interface.shutdown()

    return True


def wasdThread():
    if not mainWASD():
        sys.exit(1)

# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Graph nav utility functions"""


def id_to_short_code(id):
    """Convert a unique id to a 2 letter short code."""
    tokens = id.split('-')
    if len(tokens) > 2:
        return f'{tokens[0][0]}{tokens[1][0]}'
    return None


def pretty_print_waypoints(waypoint_id, waypoint_name, short_code_to_count, localization_id):
    short_code = id_to_short_code(waypoint_id)
    if short_code is None or short_code_to_count[short_code] != 1:
        short_code = '  '  # If the short code is not valid/unique, don't show it.

    waypoint_symbol = '->' if localization_id == waypoint_id else '  '
    print(
        f'{waypoint_symbol} Waypoint name: {waypoint_name} id: {waypoint_id} short code: {short_code}'
    )


def find_unique_waypoint_id(short_code, graph, name_to_id):
    """Convert either a 2 letter short code or an annotation name into the associated unique id."""
    if graph is None:
        print(
            'Please list the waypoints in the map before trying to navigate to a specific one (Option #4).'
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
                print(
                    f'The waypoint name {short_code} is used for multiple different unique waypoints. Please use '
                    f'the waypoint id.')
                return None
        # Also not a waypoint annotation name, so we will operate under the assumption that it is a
        # unique waypoint id.
        return short_code

    ret = short_code
    for waypoint in graph.waypoints:
        if short_code == id_to_short_code(waypoint.id):
            if ret != short_code:
                return short_code  # Multiple waypoints with same short code.
            ret = waypoint.id
    return ret


def update_waypoints_and_edges(graph, localization_id, do_print=True):
    """Update and print waypoint ids and edge ids."""
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
        short_code = id_to_short_code(waypoint.id)
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
        print(f'{len(graph.waypoints):d} waypoints:')
        for waypoint in waypoint_to_timestamp:
            pretty_print_waypoints(waypoint[0], waypoint[2], short_code_to_count, localization_id)

    for edge in graph.edges:
        if edge.id.to_waypoint in edges:
            if edge.id.from_waypoint not in edges[edge.id.to_waypoint]:
                edges[edge.id.to_waypoint].append(edge.id.from_waypoint)
        else:
            edges[edge.id.to_waypoint] = [edge.id.from_waypoint]
        if do_print:
            print(f'(Edge) from waypoint {edge.id.from_waypoint} to waypoint {edge.id.to_waypoint} '
                  f'(cost {edge.annotations.cost.value})')

    return name_to_id, edges


def sort_waypoints_chrono(graph):
    """Sort waypoints by time created."""
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

    # Sort the set of waypoints by their creation timestamp. If the creation timestamp is unavailable,
    # fallback to sorting by annotation name.
    waypoint_to_timestamp = sorted(waypoint_to_timestamp, key=lambda x: (x[1], x[2]))

    return waypoint_to_timestamp


"""Command line interface integrating options to record maps with WASD controls. """
import argparse
import logging
import os
import sys
import time

import google.protobuf.timestamp_pb2
import grpc
from google.protobuf import wrappers_pb2 as wrappers

import bosdyn.client.channel
import bosdyn.client.util
from bosdyn.api.graph_nav import map_pb2, map_processing_pb2, recording_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.map_processing import MapProcessingServiceClient
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.recording import GraphNavRecordingServiceClient


class RecordingInterface(object):
    """Recording service command line interface."""

    def __init__(self, robot, download_filepath, client_metadata, use_gps=False):
        # Keep the robot instance and it's ID.
        self._robot = robot

        self.use_gps = use_gps

        # Force trigger timesync.
        self._robot.time_sync.wait_for_sync()

        # Filepath for the location to put the downloaded graph and snapshots.
        self._download_filepath = os.path.join(download_filepath, 'downloaded_graph')

        # Set up the recording service client.
        self._recording_client = self._robot.ensure_client(
            GraphNavRecordingServiceClient.default_service_name)

        # Create the recording environment.
        self._recording_environment = GraphNavRecordingServiceClient.make_recording_environment(
            waypoint_env=GraphNavRecordingServiceClient.make_waypoint_environment(
                client_metadata=client_metadata))

        # Set up the graph nav service client.
        self._graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)

        self._map_processing_client = robot.ensure_client(
            MapProcessingServiceClient.default_service_name)

        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  #maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()

        # Add recording service properties to the command line dictionary.
        self._command_dictionary = {
            '0': self._clear_map,
            '1': self.start_recording,
            '2': self.stop_recording,
            '3': self._get_recording_status,
            '4': self._create_default_waypoint,
            '5': self.download_full_graph,
            '6': self._list_graph_waypoint_and_edge_ids,
            '7': self._create_new_edge,
            '8': self._create_loop,
            '9': self._auto_close_loops_prompt,
            'a': self._optimize_anchoring
        }

    def should_we_start_recording(self):
        # Before starting to record, check the state of the GraphNav system.
        graph = self._graph_nav_client.download_graph()
        if graph is not None:
            # Check that the graph has waypoints. If it does, then we need to be localized to the graph
            # before starting to record
            if len(graph.waypoints) > 0:
                localization_state = self._graph_nav_client.get_localization_state()
                if not localization_state.localization.waypoint_id:
                    # Not localized to anything in the map. The best option is to clear the graph or
                    # attempt to localize to the current map.
                    # Returning false since the GraphNav system is not in the state it should be to
                    # begin recording.
                    return False
        # If there is no graph or there exists a graph that we are localized to, then it is fine to
        # start recording, so we return True.
        return True

    def _clear_map(self, *args):
        """Clear the state of the map on the robot, removing all waypoints and edges."""
        return self._graph_nav_client.clear_graph()

    def start_recording(self, *args):
        """Start recording a map."""
        should_start_recording = self.should_we_start_recording()
        if not should_start_recording:
            print('The system is not in the proper state to start recording.'
                  'Try using the graph_nav_command_line to either clear the map or'
                  'attempt to localize to the map.')
            return
        try:
            status = self._recording_client.start_recording(
                recording_environment=self._recording_environment)
            print('Successfully started recording a map.')
        except Exception as err:
            print(f'Start recording failed: {err}')

    def stop_recording(self, *args):
        """Stop or pause recording a map."""
        first_iter = True
        while True:
            try:
                status = self._recording_client.stop_recording()
                print('Successfully stopped recording a map.')
                break
            except bosdyn.client.recording.NotReadyYetError as err:
                # It is possible that we are not finished recording yet due to
                # background processing. Try again every 1 second.
                if first_iter:
                    print('Cleaning up recording...')
                first_iter = False
                time.sleep(1.0)
                continue
            except Exception as err:
                print(f'Stop recording failed: {err}')
                break

    def _get_recording_status(self, *args):
        """Get the recording service's status."""
        status = self._recording_client.get_record_status()
        if status.is_recording:
            print('The recording service is on.')
        else:
            print('The recording service is off.')

    def _create_default_waypoint(self, *args):
        """Create a default waypoint at the robot's current location."""
        resp = self._recording_client.create_waypoint(waypoint_name='default')
        if resp.status == recording_pb2.CreateWaypointResponse.STATUS_OK:
            print('Successfully created a waypoint.')
        else:
            print('Could not create a waypoint.')

    def download_full_graph(self, *args):
        """Download the graph and snapshots from the robot."""
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print('Failed to download the graph.')
            return
        self._write_full_graph(graph)
        print(
            f'Graph downloaded with {len(graph.waypoints)} waypoints and {len(graph.edges)} edges')
        # Download the waypoint and edge snapshots.
        self._download_and_write_waypoint_snapshots(graph.waypoints)
        self._download_and_write_edge_snapshots(graph.edges)

    def _write_full_graph(self, graph):
        """Download the graph from robot to the specified, local filepath location."""
        graph_bytes = graph.SerializeToString()
        self._write_bytes(self._download_filepath, 'graph', graph_bytes)

    def _download_and_write_waypoint_snapshots(self, waypoints):
        """Download the waypoint snapshots from robot to the specified, local filepath location."""
        num_waypoint_snapshots_downloaded = 0
        for waypoint in waypoints:
            if len(waypoint.snapshot_id) == 0:
                continue
            try:
                waypoint_snapshot = self._graph_nav_client.download_waypoint_snapshot(
                    waypoint.snapshot_id)
            except Exception:
                # Failure in downloading waypoint snapshot. Continue to next snapshot.
                print(f'Failed to download waypoint snapshot: {waypoint.snapshot_id}')
                continue
            self._write_bytes(os.path.join(self._download_filepath, 'waypoint_snapshots'),
                              str(waypoint.snapshot_id), waypoint_snapshot.SerializeToString())
            num_waypoint_snapshots_downloaded += 1
            print(
                f'Downloaded {num_waypoint_snapshots_downloaded} of the total {len(waypoints)} waypoint snapshots.'
            )

    def _download_and_write_edge_snapshots(self, edges):
        """Download the edge snapshots from robot to the specified, local filepath location."""
        num_edge_snapshots_downloaded = 0
        num_to_download = 0
        for edge in edges:
            if len(edge.snapshot_id) == 0:
                continue
            num_to_download += 1
            try:
                edge_snapshot = self._graph_nav_client.download_edge_snapshot(edge.snapshot_id)
            except Exception:
                # Failure in downloading edge snapshot. Continue to next snapshot.
                print(f'Failed to download edge snapshot: {edge.snapshot_id}')
                continue
            self._write_bytes(os.path.join(self._download_filepath, 'edge_snapshots'),
                              str(edge.snapshot_id), edge_snapshot.SerializeToString())
            num_edge_snapshots_downloaded += 1
            print(
                f'Downloaded {num_edge_snapshots_downloaded} of the total {num_to_download} edge snapshots.'
            )

    def _write_bytes(self, filepath, filename, data):
        """Write data to a file."""
        os.makedirs(filepath, exist_ok=True)
        with open(os.path.join(filepath, filename), 'wb+') as f:
            f.write(data)
            f.close()

    def _update_graph_waypoint_and_edge_ids(self, do_print=False):
        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print('Empty graph.')
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = update_waypoints_and_edges(
            graph, localization_id, do_print)

    def _list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""
        self._update_graph_waypoint_and_edge_ids(do_print=True)

    def _create_new_edge(self, *args):
        """Create new edge between existing waypoints in map."""

        if len(args[0]) != 2:
            print('ERROR: Specify the two waypoints to connect (short code or annotation).')
            return

        self._update_graph_waypoint_and_edge_ids(do_print=False)

        from_id = find_unique_waypoint_id(args[0][0], self._current_graph,
                                                         self._current_annotation_name_to_wp_id)
        to_id = find_unique_waypoint_id(args[0][1], self._current_graph,
                                                       self._current_annotation_name_to_wp_id)

        print(f'Creating edge from {from_id} to {to_id}.')

        from_wp = self._get_waypoint(from_id)
        if from_wp is None:
            return

        to_wp = self._get_waypoint(to_id)
        if to_wp is None:
            return

        # Get edge transform based on kinematic odometry
        edge_transform = self._get_transform(from_wp, to_wp)

        # Define new edge
        new_edge = map_pb2.Edge()
        new_edge.id.from_waypoint = from_id
        new_edge.id.to_waypoint = to_id
        new_edge.from_tform_to.CopyFrom(edge_transform)

        print(f'edge transform = {new_edge.from_tform_to}')

        # Send request to add edge to map
        self._recording_client.create_edge(edge=new_edge)

    def _create_loop(self, *args):
        """Create edge from last waypoint to first waypoint."""

        self._update_graph_waypoint_and_edge_ids(do_print=False)

        if len(self._current_graph.waypoints) < 2:
            self._add_message(
                f'Graph contains {len(self._current_graph.waypoints)} waypoints -- at least two are '
                f'needed to create loop.')
            return False

        sorted_waypoints = sort_waypoints_chrono(self._current_graph)
        edge_waypoints = [sorted_waypoints[-1][0], sorted_waypoints[0][0]]

        self._create_new_edge(edge_waypoints)

    def _auto_close_loops_prompt(self, *args):
        print("""
        Options:
        (0) Close all loops.
        (1) Close only fiducial-based loops.
        (2) Close only odometry-based loops.
        (q) Back.
        """)
        try:
            inputs = input('>')
        except NameError:
            return
        req_type = str.split(inputs)[0]
        close_fiducial_loops = False
        close_odometry_loops = False
        if req_type == '0':
            close_fiducial_loops = True
            close_odometry_loops = True
        elif req_type == '1':
            close_fiducial_loops = True
        elif req_type == '2':
            close_odometry_loops = True
        elif req_type == 'q':
            return
        else:
            print('Unrecognized command. Going back.')
            return
        self._auto_close_loops(close_fiducial_loops, close_odometry_loops)

    def _auto_close_loops(self, close_fiducial_loops, close_odometry_loops, *args):
        """Automatically find and close all loops in the graph."""
        response = self._map_processing_client.process_topology(
            params=map_processing_pb2.ProcessTopologyRequest.Params(
                do_fiducial_loop_closure=wrappers.BoolValue(value=close_fiducial_loops),
                do_odometry_loop_closure=wrappers.BoolValue(value=close_odometry_loops)),
            modify_map_on_server=True)
        print(f'Created {len(response.new_subgraph.edges)} new edge(s).')

    def _optimize_anchoring(self, *args):
        """Call anchoring optimization on the server, producing a globally optimal reference frame for waypoints to be expressed in."""
        response = self._map_processing_client.process_anchoring(
            params=map_processing_pb2.ProcessAnchoringRequest.Params(),
            modify_anchoring_on_server=True, stream_intermediate_results=False,
            apply_gps_results=self.use_gps)
        if response.status == map_processing_pb2.ProcessAnchoringResponse.STATUS_OK:
            print(f'Optimized anchoring after {response.iteration} iteration(s).')
            # If we are using GPS, the GPS coordinates in the graph have been changed, so we need
            # to re-download the graph.
            if self.use_gps:
                print(f'Downloading updated graph...')
                self._current_graph = self._graph_nav_client.download_graph()
        else:
            print(f'Error optimizing {response}')

    def _get_waypoint(self, id):
        """Get waypoint from graph (return None if waypoint not found)"""

        if self._current_graph is None:
            self._current_graph = self._graph_nav_client.download_graph()

        for waypoint in self._current_graph.waypoints:
            if waypoint.id == id:
                return waypoint

        print(f'ERROR: Waypoint {id} not found in graph.')
        return None

    def _get_transform(self, from_wp, to_wp):
        """Get transform from from-waypoint to to-waypoint."""

        from_se3 = from_wp.waypoint_tform_ko
        from_tf = SE3Pose(
            from_se3.position.x, from_se3.position.y, from_se3.position.z,
            Quat(w=from_se3.rotation.w, x=from_se3.rotation.x, y=from_se3.rotation.y,
                 z=from_se3.rotation.z))

        to_se3 = to_wp.waypoint_tform_ko
        to_tf = SE3Pose(
            to_se3.position.x, to_se3.position.y, to_se3.position.z,
            Quat(w=to_se3.rotation.w, x=to_se3.rotation.x, y=to_se3.rotation.y,
                 z=to_se3.rotation.z))

        from_T_to = from_tf.mult(to_tf.inverse())
        return from_T_to.to_proto()

    def run(self):
        """Main loop for the command line interface."""
        while True:
            print("""
            Options:
            (0) Clear map.
            (1) Start recording a map.
            (2) Stop recording a map.
            (3) Get the recording service's status.
            (4) Create a default waypoint in the current robot's location.
            (5) Download the map after recording.
            (6) List the waypoint ids and edge ids of the map on the robot.
            (7) Create new edge between existing waypoints using odometry.
            (8) Create new edge from last waypoint to first waypoint using odometry.
            (9) Automatically find and close loops.
            (a) Optimize the map's anchoring.
            (q) Exit.
            """)
            try:
                inputs = input('>')
            except NameError:
                pass
            req_type = str.split(inputs)[0]

            if req_type == 'q':
                break

            if req_type not in self._command_dictionary:
                print('Request not in the known command dictionary.')
                continue
            try:
                cmd_func = self._command_dictionary[req_type]
                cmd_func(str.split(inputs)[1:])
            except Exception as e:
                print(e)


if __name__ == "__main__":
    # Create robot object.
    sdk = bosdyn.client.create_standard_sdk('RecordingClient')
    robot = sdk.create_robot('192.168.80.3')
    bosdyn.client.util.authenticate(robot)

    load_dotenv('.env.local')
    download_filepath = os.getenv('GRAPH_NAV_GRAPH_DOWNLOAD_FILEPATH')

    # Parse session and user name options.
    session_name = ''
    if session_name == '':
        session_name = os.path.basename(download_filepath)
    user_name = ''
    if user_name == '':
        user_name = robot._current_user

    # Crate metadata for the recording session.
    client_metadata = GraphNavRecordingServiceClient.make_client_metadata(
        session_name=session_name, client_username=user_name, client_id='RecordingClient',
        client_type='Python SDK')
    recording_command_line = RecordingInterface(robot, download_filepath, client_metadata)

    recording_command_line.start_recording()

    wasdThread()

    recording_command_line.stop_recording()

    recording_command_line.download_full_graph()