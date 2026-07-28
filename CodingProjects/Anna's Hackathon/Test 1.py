import lelib
from lelib import doubleMotor, colorSensor, controller

# Connecting
dm = doubleMotor()
dm.connect(card_serial="6325") # Or whatever your card is

cs = colorSensor()
cs.connect(card_serial="6325") # Or whatever your card is

# You wont need the controller until the 2nd activity
c = controller()
c.connect(card_serial="6325") # Or whatever your card is


# Set the speed of both motors in the double motor
dm.set_speed(speed)

# Run the double motor
dm.run()

# Stop the motor:
dm.stop()

# get the color sensor reflection reading
cs.sensor.reflection

# disconnecting
dm.disconnect()
cs.disconnect()
c.disconnect()