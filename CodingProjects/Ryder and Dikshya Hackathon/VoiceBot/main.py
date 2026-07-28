import recognize
import lelib 
from lelib import singleMotor, doubleMotor, colorSensor, controller

dm = doubleMotor()
dm.connect(card_serial="1127") #CHANGE TO YOUR CARD'S NUMBER
dm.set_speed(30)

def on_go(label, score):
    print(f"  → GO action here (confidence: {score:.0%})")
    dm.run()



def on_left(label, score):
    print(f"  → LEFT action here (confidence: {score:.0%})")
    dm.turn_left(90)



def on_right(label, score):
    print(f"  → RIGHT action here (confidence: {score:.0%})")
    dm.turn_right(90)



def on_stop(label, score):
    print(f"  → STOP action here (confidence: {score:.0%})")
    dm.stop()



callbacks = {
    "Go":    on_go,
    "Left":  on_left,
    "Right": on_right,
    "Stop":  on_stop,
}

if __name__ == "__main__":
    while True:
        try:
            recognize.start(callbacks)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e} — restarting...")
    try:
        recognize.start(callbacks)
    except KeyboardInterrupt:
        print("\nStopped.")
