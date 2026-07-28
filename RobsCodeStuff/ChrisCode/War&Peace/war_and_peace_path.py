import matplotlib.pyplot as plt
import matplotlib.collections as mc
import numpy as np

# First 100 words of War and Peace (Maude translation, ASCII-safe)
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

words = text.split()[:100]
text_100 = " ".join(words)
print(f"Text ({len(words)} words):\n  {text_100}\n")

# Convert to ASCII bytes
ascii_bytes = text_100.encode("ascii", errors="ignore")
print(f"Byte count: {len(ascii_bytes)}")

# Convert each byte to 8 bits, MSB first
bits = []
for byte in ascii_bytes:
    for shift in range(7, -1, -1):
        bits.append((byte >> shift) & 1)
print(f"Bit count:  {len(bits)}")
print(f"Steps:      {len(bits) // 4}\n")

# Bits 0-1 → direction, bits 2-3 → distance (value 0-3, +1 so always moves)
#   00 = north (+y),  01 = east (+x)
#   10 = south (-y),  11 = west (-x)
DIRS = {
    (0, 0): ( 0,  1),
    (0, 1): ( 1,  0),
    (1, 0): ( 0, -1),
    (1, 1): (-1,  0),
}

x, y = 0.0, 0.0
xs, ys = [x], [y]

for i in range(0, len(bits) - 3, 4):
    dx, dy = DIRS[(bits[i], bits[i + 1])]
    dist = (bits[i + 2] << 1 | bits[i + 3]) + 1  # 1–4 pixels
    x += dx * dist
    y += dy * dist
    xs.append(x)
    ys.append(y)

xs, ys = np.array(xs), np.array(ys)

# Color the path by progress (blue → red)
fig, ax = plt.subplots(figsize=(10, 10))
n = len(xs) - 1
cmap = plt.get_cmap("plasma")
segments = [[(xs[i], ys[i]), (xs[i+1], ys[i+1])] for i in range(n)]
lc = mc.LineCollection(segments, cmap=cmap, linewidths=1.2)
lc.set_array(np.linspace(0, 1, n))
ax.add_collection(lc)

ax.plot(xs[0],  ys[0],  "go", markersize=9, zorder=5, label="Start")
ax.plot(xs[-1], ys[-1], "rs", markersize=9, zorder=5, label="End")

ax.set_xlim(xs.min() - 10, xs.max() + 10)
ax.set_ylim(ys.min() - 10, ys.max() + 10)
ax.set_aspect("equal")
ax.set_title(
    "War and Peace — first 100 words as an ASCII bit-walk\n"
    "(bits 0-1: 00=N, 01=E, 10=S, 11=W  |  bits 2-3: distance 1–4 px)",
    fontsize=13,
)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.25)
fig.colorbar(lc, ax=ax, label="Progress (start → end)", shrink=0.6)
plt.tight_layout()
plt.savefig("war_and_peace_path.png", dpi=150, bbox_inches="tight")
print("Saved war_and_peace_path.png")
plt.show()
