import runloop
from hub import port
import motor

# ── calibrate these for your specific robot build ──────────────────────────
TURN_90_MS   = 500   # ms for a 90° point turn
MOVE_UNIT_MS = 400   # ms per pixel unit of straight travel (steps are 1–4 px)
VELOCITY     = 300   # deg/s
# If the robot drives backward instead of forward, flip B_DIR to +1
B_DIR        = -1    # port B is mounted facing opposite to A, so needs negation
# ──────────────────────────────────────────────────────────────────────────

# First 100 words of War and Peace (same text as the plot version)
text = (
    "Well, Prince, so Genoa and Lucca are now just family estates of the "
    "Buonapartes. But I warn you, if you don't tell me that this means war, "
    "if you still try to defend the infamies and horrors of this Antichrist "
    "I really believe he is Antichrist I will have nothing more to do with "
    "you and you are no longer my friend, not even my faithful slave, as you "
    "call yourself! But how do you do? I see I have frightened you sit down "
    "and tell me all the news. It was in July, 1805, and the speaker was the "
    "well-known Anna Pavlovna Scherer, maid of honor and favorite of the "
    "Empress Marya Fedorovna. With these words she greeted Prince Vasili "
    "Kuragin, a man of high rank and importance, who was the first to arrive."
)

text_100 = " ".join(text.split()[:100])

# Convert to bits, MSB first, skipping any non-ASCII chars
bits = []
for ch in text_100:
    b = ord(ch)
    if b > 127:
        continue
    for shift in range(7, -1, -1):
        bits.append((b >> shift) & 1)

# Heading encoded as clockwise index: 0=N 1=E 2=S 3=W
# Bits 0-1 of each 4-bit group → target heading
# Bits 2-3                     → distance (1–4 px)
DIR_MAP = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}

async def pivot(turn_steps):
    """Point turn: both motors spin in opposite directions so B always moves.
    +ve = right (CW), -ve = left (CCW)."""
    duration_ms = abs(turn_steps) * TURN_90_MS
    # Right turn: A forward, B backward. Left turn: A backward, B forward.
    v = VELOCITY if turn_steps > 0 else -VELOCITY
    motor.run(port.E,  v)
    motor.run(port.F,  v)  # same raw sign = opposite robot-direction (motors face each other)
    await runloop.sleep_ms(duration_ms)
    motor.stop(port.E)
    motor.stop(port.F)

async def main():
    heading = 0  # start facing north

    for i in range(0, len(bits) - 3, 4):
        target = DIR_MAP[(bits[i], bits[i + 1])]
        dist   = (bits[i + 2] << 1 | bits[i + 3]) + 1  # 1–4

        # Shortest turn to face target (−1 = one left, +1 = one right, +2 = 180°)
        turn = (target - heading) % 4
        if turn == 3:
            turn = -1  # left 90° is shorter than right 270°
        if turn != 0:
            await pivot(turn)
        heading = target

        # Drive straight for dist pixel-units
        motor.run(port.E, VELOCITY)
        motor.run(port.F, B_DIR * VELOCITY)
        await runloop.sleep_ms(dist * MOVE_UNIT_MS)
        motor.stop(port.E)
        motor.stop(port.F)

runloop.run(main())
