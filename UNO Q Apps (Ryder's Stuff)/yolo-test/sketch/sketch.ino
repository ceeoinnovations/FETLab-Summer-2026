#include "Arduino_RouterBridge.h"

bool set_person_detected(bool detected) {
  digitalWrite(LED_BUILTIN, detected ? HIGH : LOW);
  return detected;
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Bridge.begin();
  Bridge.provide("set_person_detected", set_person_detected);
}

void loop() {
  // Bridge.update() is handled automatically
}
