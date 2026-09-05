/*
  NeuroProx single-channel EMG streamer

  EMG_SAMPLE_RATE_HZ is 500 Hz: fast enough to preserve the useful bandwidth
  of typical surface EMG while remaining comfortable for an Arduino Uno ADC
  and a 115200-baud integer stream. Change EMG_ANALOG_PIN below if Patchy OUT
  is connected to a different analog input (for example, A1).
*/

const uint8_t EMG_ANALOG_PIN = A0;
const unsigned long EMG_SAMPLE_RATE_HZ = 500;
const unsigned long SAMPLE_PERIOD_US = 1000000UL / EMG_SAMPLE_RATE_HZ;
const unsigned long SERIAL_BAUD_RATE = 115200;

unsigned long nextSampleAtUs;

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);
  pinMode(EMG_ANALOG_PIN, INPUT);
  nextSampleAtUs = micros();
}

void loop() {
  const unsigned long nowUs = micros();

  // Signed subtraction keeps this comparison correct when micros() wraps.
  if ((long)(nowUs - nextSampleAtUs) >= 0) {
    nextSampleAtUs += SAMPLE_PERIOD_US;
    Serial.println(analogRead(EMG_ANALOG_PIN));
  }
}
