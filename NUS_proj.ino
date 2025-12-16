#include <Wire.h>

#define VIB_PIN 34
#define TRIG_PIN 26
#define ECHO_PIN 27

#define MAG_ADDR 0x0D   // QMC5883L address

void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MAG_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

int16_t read16(uint8_t reg) {
  Wire.beginTransmission(MAG_ADDR);
  Wire.write(reg);
  Wire.endTransmission();
  Wire.requestFrom(MAG_ADDR, 2);
  return Wire.read() | (Wire.read() << 8);
}

float getDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0) return -1;
  return duration * 0.034 / 2.0;
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);

  pinMode(VIB_PIN, INPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // QMC5883L init
  writeReg(0x0B, 0x01);   // Reset
  writeReg(0x09, 0x1D);   // Continuous, 200Hz, 2G, OSR=512

  Serial.println("time_ms,vibration,mag_x,mag_y,mag_z,distance_cm");
}

void loop() {
  int vibration = digitalRead(VIB_PIN);

  int16_t mag_x = read16(0x00);
  int16_t mag_y = read16(0x02);
  int16_t mag_z = read16(0x04);

  float distance = getDistance();

  Serial.print(millis());
  Serial.print(",");
  Serial.print(vibration);
  Serial.print(",");
  Serial.print(mag_x);
  Serial.print(",");
  Serial.print(mag_y);
  Serial.print(",");
  Serial.print(mag_z);
  Serial.print(",");
  Serial.println(distance);

  delay(500);
}
