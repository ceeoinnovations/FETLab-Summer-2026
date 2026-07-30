import struct
import socket
import requests

print("BRIDGE CLIENT LOADED", flush=True)

def _get_gateway():
    with open('/proc/net/route') as f:
        for line in f.readlines()[1:]:
            fields = line.strip().split()
            if fields[1] == '00000000':
                return socket.inet_ntoa(struct.pack('<I', int(fields[2], 16)))
    raise RuntimeError("No default gateway found in /proc/net/route")

_BRIDGE_HOST = _get_gateway()
_BRIDGE_PORT = 5000
_BRIDGE = f"http://{_BRIDGE_HOST}:{_BRIDGE_PORT}"

# Exit code callers can use to distinguish "the LEGO BLE bridge is down or stuck" from
# a generic crash - sysexits.h's EX_TEMPFAIL ("temporary failure, please retry").
EXIT_BRIDGE_BUSY = 75


class BridgeBusyError(ConnectionError):
    """The LEGO BLE bridge (ble-bridge.service, port 5000) didn't respond in time or
    isn't reachable at all.

    The most common cause: a previous run crashed/was killed without calling
    disconnect(), so the hub's BLE link is still alive from the bridge's point of view.
    The bridge is a single-central peripheral server (see ble_bridge.py's
    _drop_stale()) - a stale link blocks new /connect calls to the same hub until it
    either times out on its own or the bridge process is restarted. Wait a few seconds
    and retry, or `systemctl restart ble-bridge.service` if it persists.
    """


def _check_port(timeout=2.0):
    """Fast TCP-level probe, so a fully-dead bridge process fails immediately with a
    clear message instead of waiting out a full request timeout.

    This can't detect the "bridge is up but busy on a slow/stale connect" case - the
    bridge (ble_bridge.py) handles one request at a time and won't accept new
    connections while blocked inside one, so a stuck bridge makes this check hang too,
    same as a real request would. That case is still caught by _post()/_get() below.
    """
    try:
        with socket.create_connection((_BRIDGE_HOST, _BRIDGE_PORT), timeout=timeout):
            pass
    except OSError as e:
        raise BridgeBusyError(f"LEGO BLE bridge not reachable at {_BRIDGE_HOST}:{_BRIDGE_PORT} ({e}). Is ble-bridge.service running?") from e


def _post(path, **kw):
    try:
        return requests.post(f"{_BRIDGE}{path}", json=kw, timeout=10).json()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        raise BridgeBusyError(f"LEGO BLE bridge did not respond in time calling {path} ({e}).") from e


def _get(path, **params):
    try:
        return requests.get(f"{_BRIDGE}{path}", params=params, timeout=10).json()
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        raise BridgeBusyError(f"LEGO BLE bridge did not respond in time calling {path} ({e}).") from e


try:
    globals().update(_get('/constants'))
except Exception:
    pass


class _SensorProxy:
    def __init__(self, dev):
        self._d = dev

    @property
    def color(self):
        return _get('/sensor/read', handle=self._d._h)['color']

    @property
    def reflection(self):
        return _get('/sensor/read', handle=self._d._h)['reflection']

    @property
    def rgb(self):
        return tuple(_get('/sensor/read', handle=self._d._h)['rgb'])

    @property
    def hsv(self):
        return tuple(_get('/sensor/read', handle=self._d._h)['hsv'])


class _LeverSide:
    def __init__(self, dev, side):
        self._d, self._s = dev, side

    @property
    def angle(self):
        return _get('/controller/read', handle=self._d._h)[f'{self._s}_angle']

    @property
    def percent(self):
        return _get('/controller/read', handle=self._d._h)[f'{self._s}_percent']


class _Levers:
    def __init__(self, dev):
        self.left = _LeverSide(dev, 'left')
        self.right = _LeverSide(dev, 'right')


class _ControllerSensor:
    def __init__(self, dev):
        self._d = dev

    @property
    def leftPercent(self):
        return _get('/controller/read', handle=self._d._h)['left_percent']

    @property
    def rightPercent(self):
        return _get('/controller/read', handle=self._d._h)['right_percent']


class _Imu:
    def __init__(self, dev):
        self._d = dev

    @property
    def acceleration(self):
        return tuple(_get('/imu/read', handle=self._d._h)['acceleration'])

    @property
    def gyro(self):
        return tuple(_get('/imu/read', handle=self._d._h)['gyro'])


class _Base:
    def __init__(self, typ):
        self._typ, self._h, self.connected = typ, None, False

    def connect(self, card_color=None, card_serial=None):
        _check_port()
        kw = {'type': self._typ}
        if card_color is not None:
            kw['card_color'] = card_color
        if card_serial is not None:
            kw['card_serial'] = card_serial
        r = _post('/connect', **kw)
        self.connected = r.get('connected', False)
        if self.connected:
            self._h = r['handle']

    def disconnect(self):
        if self._h:
            _post('/disconnect', handle=self._h)
        self.connected, self._h = False, None


class SingleMotor(_Base):
    def __init__(self):
        super().__init__('SingleMotor')

    def motor_run_for_degrees(self, degrees, direction=None, speed=None):
        _post('/motor/run_for_degrees', handle=self._h, degrees=degrees, direction=direction, speed=speed)

    def motor_run_for_time(self, ms, direction=None, speed=None):
        _post('/motor/run_for_time', handle=self._h, milliseconds=ms, direction=direction, speed=speed)

    def motor_run(self, direction=None, speed=None):
        _post('/motor/run', handle=self._h, direction=direction, speed=speed)

    def motor_stop(self, end_state=None):
        _post('/motor/stop', handle=self._h, end_state=end_state)

    def motor_set_speed(self, speed):
        _post('/motor/set_speed', handle=self._h, speed=speed)

    def motor_go_to_position(self, position, direction=None, speed=None):
        _post('/motor/go_to_position', handle=self._h, position=position, direction=direction, speed=speed)


class DoubleMotor(_Base):
    def __init__(self):
        super().__init__('DoubleMotor')
        self.imu = _Imu(self)

    def motor_run_for_degrees(self, degrees, direction=None, speed=None, motor=None):
        _post('/motor/run_for_degrees', handle=self._h, degrees=degrees, direction=direction, speed=speed, motor=motor)

    def motor_run_for_time(self, ms, direction=None, speed=None, motor=None):
        _post('/motor/run_for_time', handle=self._h, milliseconds=ms, direction=direction, speed=speed, motor=motor)

    def motor_run(self, direction=None, speed=None, motor=None):
        _post('/motor/run', handle=self._h, direction=direction, speed=speed, motor=motor)

    def motor_stop(self, end_state=None):
        _post('/motor/stop', handle=self._h, end_state=end_state)

    def motor_set_speed(self, speed, motor=None):
        _post('/motor/set_speed', handle=self._h, speed=speed, motor=motor)

    def motor_go_to_position(self, position, direction=None, speed=None):
        _post('/motor/go_to_position', handle=self._h, position=position, direction=direction, speed=speed)

    def movement_move_for_degrees(self, degrees, direction=None, speed=None):
        _post('/movement/move_for_degrees', handle=self._h, degrees=degrees, direction=direction, speed=speed)

    def movement_move(self, direction=None, speed=None):
        _post('/movement/move', handle=self._h, direction=direction, speed=speed)

    def movement_move_for_time(self, ms, direction=None, speed=None):
        _post('/movement/move_for_time', handle=self._h, milliseconds=ms, direction=direction, speed=speed)

    def movement_turn_for_degrees(self, degrees, direction=None, speed=None):
        _post('/movement/turn_for_degrees', handle=self._h, degrees=degrees, direction=direction, speed=speed)

    def movement_set_speed(self, speed):
        _post('/movement/set_speed', handle=self._h, speed=speed)

    def movement_move_tank(self, left_speed, right_speed):
        _post('/movement/move_tank', handle=self._h, left_speed=left_speed, right_speed=right_speed)


class ColorSensor(_Base):
    def __init__(self):
        super().__init__('ColorSensor')
        self.sensor = _SensorProxy(self)

    def light_color(self, color, pattern=None, intensity=None):
        _post('/light/color', handle=self._h, color=color, pattern=pattern, intensity=intensity)

    def beep(self, pattern=None, frequency=None):
        _post('/sound/beep', handle=self._h, pattern=pattern, frequency=frequency)


class Controller(_Base):
    def __init__(self):
        super().__init__('Controller')
        self.sensor = _ControllerSensor(self)
        self.lever = _Levers(self)

