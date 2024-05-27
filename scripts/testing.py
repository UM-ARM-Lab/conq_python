import bosdyn.client
import bosdyn.client.time_sync
import bosdyn.client.robot_state

from bosdyn.client.robot_state import RobotStateClient

if __name__ == "__main__":
    while True:
        sdk = bosdyn.client.create_standard_sdk('EKF')
        robot = sdk.create_robot('192.168.80.3')

        bosdyn.client.util.authenticate(robot)

        client = robot.ensure_client(RobotStateClient.default_service_name)

        state = client.get_robot_state(timeout=1)

        velocity = state.kinematic_state.velocity_of_body_in_odom

        # Print linear velocity
        print("Linear Velocity", velocity.linear)

        # Print angular velocity
        print("Angular Velocity", velocity.angular)
