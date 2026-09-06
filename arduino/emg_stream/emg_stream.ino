/*
  NeuroProx single-channel EMG streamer
  + OPEN motor control

  EMG:
    A0 -> EMG sensor
    500 Hz -> USB serial

  Motor:
    D7 -> MOSFET gate
    HIGH = motor ON / opening
    LOW  = motor OFF
*/

const uint8_t EMG_ANALOG_PIN = A0;
const uint8_t MOTOR_PIN = 7;

const unsigned long EMG_SAMPLE_RATE_HZ = 500;
const unsigned long SAMPLE_PERIOD_US =
    1000000UL / EMG_SAMPLE_RATE_HZ;

const unsigned long SERIAL_BAUD_RATE = 115200;

unsigned long nextSampleAtUs;

void setup() {
  Serial.begin(SERIAL_BAUD_RATE);

  pinMode(EMG_ANALOG_PIN, INPUT);

  pinMode(MOTOR_PIN, OUTPUT);
  digitalWrite(MOTOR_PIN, LOW);

  nextSampleAtUs = micros();
}

void loop() {

  // -------------------------
  // Receive commands from Python
  // -------------------------

  if (Serial.available() > 0) {

    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "CLOSE") {
      digitalWrite(MOTOR_PIN, HIGH);
    }

    if (command == "STOP") {
      digitalWrite(MOTOR_PIN, LOW);
    }
  }


  // -------------------------
  // EMG sampling at 500 Hz
  // -------------------------

  const unsigned long nowUs = micros();

  if ((long)(nowUs - nextSampleAtUs) >= 0) {

    nextSampleAtUs += SAMPLE_PERIOD_US;

    Serial.println(analogRead(EMG_ANALOG_PIN));
  }
}