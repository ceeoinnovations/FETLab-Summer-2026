#include "Arduino_RouterBridge.h"

void setup() {
  Bridge.begin();
}

void loop() {
  // All audio/Bluetooth logic lives on the Linux side (python/main.py).
  // Bridge.update() is handled automatically.
}
