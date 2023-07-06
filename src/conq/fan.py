from bosdyn.client.power import FanControlTemperatureError


def try_reduce_fan(power_client):
    try:
        power_client.fan_power_command(10, 30, lease=None)
    except FanControlTemperatureError:
        pass
