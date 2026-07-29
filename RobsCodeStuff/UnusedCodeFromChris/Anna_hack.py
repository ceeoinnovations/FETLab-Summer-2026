import lelib
from lelib import doubleMotor, colorSensor, controller

SERIAL = "7576"
# Connecting
dm = doubleMotor()
dm.connect(card_serial=SERIAL) # Or whatever your card is

cs = colorSensor()
cs.connect(card_serial=SERIAL) # Or whatever your card is

# You wont need the controller until the 2nd activity
c = controller()
c.connect(card_serial=SERIAL) # Or whatever your card is

curr_speed = 0
'''while True:
    light = cs.sensor.reflection
    curr_speed = 0.1 * (100 - light)
    dm.set_speed(curr_speed)
    print(light, curr_speed)
    dm.run()'''

while True:
    state = abs(c.sensor.leftPercent) > 5  or abs(c.sensor.rightPercent) > 5
    print(state)
    if state:
        dm.run(100)
    else:
        dm.run(0)