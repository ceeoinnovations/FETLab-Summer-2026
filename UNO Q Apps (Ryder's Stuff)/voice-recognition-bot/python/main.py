import time

from arduino.app_utils import App
from arduino.app_bricks.keyword_spotting import KeywordSpotting
from arduino.app_utils import App


#legoeducatio setup
import lelib as le
from lelib import doubleMotor

myMotor = le.doubleMotor()
myMotor.connect(card_serial="1133")

#Define functions for callbacks
myMotor.set_speed(30)

def onGo():
    myMotor.run()

def onStop():
    myMotor.stop()

def onLeft():
    myMotor.turn_left(90)

def onRight():
    myMotor.turn_right(90)
    

spotting = KeywordSpotting()
spotting.on_detect("Go", lambda: onGo)
spotting.on_detect("Stop", lambda: onStop)
spotting.on_detect("Left", lambda: onLeft)
spotting.on_detect("Right", lamda: onRight)

App.run()


def loop():
    """This function is called repeatedly by the App framework."""
    # You can replace this with any code you want your App to run repeatedly.
    


# See: https://docs.arduino.cc/software/app-lab/tutorials/getting-started/#app-run
App.run(user_loop=loop)
