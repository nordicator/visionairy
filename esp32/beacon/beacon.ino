// VisionAiry beacon: tracks its own 3D position with an MPU-6050/6500 and streams it.
//
// Calibration ("cal"): lay the board flat and still with the phone on top, phone's top edge along the IMU's
// printed Y arrow. The board learns its gyro bias, accel scale and which way is up, and sets up the shared
// world frame: origin = this spot, Z up, Y = the Y arrow's horizontal direction (= the phone's top edge),
// X = right. The phone starts its own tracking from the same origin at the same moment.
//
// Tracking is dead reckoning: gyro -> orientation, orientation + accelerometer -> world acceleration ->
// velocity -> position. On its own that drifts within seconds, so two physical facts keep it honest:
//   - Zero-velocity updates: whenever the board is still for 80 ms its velocity is exactly zero.
//   - Free fall: once airborne the only force is gravity, so flight uses exact ballistics, not the sensor.
// "zero" re-zeroes position (e.g. with the beacon touching the phone) without redoing calibration.
//
// Every message goes out on USB serial (115200) and, when Wi-Fi is up, as a UDP broadcast on port 4210.
//   state, 30x/s:     VAS1 <uptime> <px> <py> <pz> <qw> <qx> <qy> <qz> <flags: 1 still, 2 airborne, 4 calibrated>
//   heartbeat, 4x/s:  VAB1 <uptime> <rssi> <led> <imu 0|1> <accel g> <spin dps> <airborne> <cal> <yaw> <pitch>
//   events:           VAE1 launch <uptime> <cal> <vx> <vy> <vz> <px> <py> <pz>   (m/s, m)
//                     VAE1 land <uptime> <airtime ms> <impact g> <max spin dps>
//                     VAE1 cal ok | VAE1 cal moving | VAE1 zero
// Commands (UDP to us on 4210, or serial): "cal" | "zero" | "on" | "off" | "blink" | "imu" (toggle debug stream)

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include "secrets.h"  // generated from esp32/.env by flash.sh

#define LED_ONBOARD 2
#define LED_EXTERNAL 26
#define IMU_SDA 32
#define IMU_SCL 33
#define UDP_PORT 4210
#define SERIAL_BAUD 115200  // higher rates corrupt on this board's USB chip

static const float G = 9.81f;
static const float DEG2RAD = 0.01745329f;

// Free fall: a spinning board reads a little centripetal acceleration, so "zero g" is < 0.5 g.
static const float FREEFALL_G = 0.5f;
static const uint32_t FREEFALL_MS = 30;
static const float LANDED_G = 0.8f;
static const uint32_t LANDED_MS = 40;
static const float IMPACT_G = 2.0f;

static const uint32_t SAMPLE_US = 2000;  // 500 Hz
static const int STILL_SAMPLES = 40;     // 80 ms: long enough that it's really rest, not the hand coasting
static const int CAL_SAMPLES = 400;      // 0.8 s still for calibration
static const float MAX_SPEED = 25;       // m/s: anything faster is integration gone wrong

enum LedMode { LED_ON, LED_OFF, LED_BLINK };
static const char *LED_NAMES[] = {"on", "off", "blink"};
static LedMode ledMode = LED_BLINK;  // blinking = not calibrated yet

static WiFiUDP udp;
static bool udpStarted = false;
static uint32_t lastBeat = 0, lastState = 0;

// --- IMU ---
static uint8_t imuAddr = 0;
static uint32_t lastImuProbe = 0, lastSampleUs = 0, lastImuPrint = 0;
static bool imuStream = false;
static float accelG = 1, spinDps = 0;

// Orientation: rows are the world X/Y/Z axes expressed in board coordinates, so v_world[i] = R[i] · v_board.
static float R[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
static bool calibrated = false;
static float gyroBias[3] = {0, 0, 0};
static float gScale = 1;  // corrects the accelerometer's scale so rest reads exactly 1 g

static int calLeft = 0;
static double calA[3], calW[3], calWMax;

// Dead reckoning, world frame
static float pos[3] = {0, 0, 0}, vel[3] = {0, 0, 0}, linAcc[3] = {0, 0, 0};
static int stillRun = 0;
static bool still = false;

// Throw state
static bool airborne = false;
static uint32_t lowSince = 0, highSince = 0, launchAt = 0;
static float lowVel[3], lowPos[3];  // state when free fall began = the release
static float flightMaxSpin = 0, flightPeakG = 0;

static void setLed(bool on) {
  digitalWrite(LED_ONBOARD, on);
  digitalWrite(LED_EXTERNAL, on);
}

static void updateLed(uint32_t now) {
  if (calLeft) return setLed((now / 60) % 2);  // fast flicker while calibrating
  switch (ledMode) {
    case LED_ON: setLed(true); break;
    case LED_OFF: setLed(false); break;
    case LED_BLINK: setLed((now / 500) % 2); break;
  }
}

// Serial + one UDP broadcast. For the high-rate state stream, where a lost packet doesn't matter.
static void sendLine(const char *buf) {
  Serial.println(buf);
  if (!udpStarted) return;
  udp.beginPacket(IPAddress(255, 255, 255, 255), UDP_PORT);
  udp.print(buf);
  udp.endPacket();
}

// Events go out twice over UDP (it can drop packets); the server ignores the repeat.
static void sendEvent(const char *fmt, ...) {
  char buf[128];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  sendLine(buf);
  if (!udpStarted) return;
  udp.beginPacket(IPAddress(255, 255, 255, 255), UDP_PORT);
  udp.print(buf);
  udp.endPacket();
}

// ---------------------------------------------------------------------------
// Vector helpers

static inline float dot3(const float *a, const float *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
static inline void cross3(const float *a, const float *b, float *o) {
  o[0] = a[1] * b[2] - a[2] * b[1];
  o[1] = a[2] * b[0] - a[0] * b[2];
  o[2] = a[0] * b[1] - a[1] * b[0];
}
static inline void normalize3(float *v) {
  float n = sqrtf(dot3(v, v));
  if (n > 1e-9f) { v[0] /= n; v[1] /= n; v[2] /= n; }
}

// Keep R a proper rotation: up stays as is, rebuild X and Y around it.
static void orthonormalize() {
  normalize3(R[2]);
  cross3(R[1], R[2], R[0]);  // x = y × z
  normalize3(R[0]);
  cross3(R[2], R[0], R[1]);  // y = z × x
}

// R (board -> world) as a quaternion w, x, y, z
static void quaternion(float *q) {
  float tr = R[0][0] + R[1][1] + R[2][2];
  if (tr > 0) {
    float s = sqrtf(tr + 1) * 2;
    q[0] = s / 4;
    q[1] = (R[2][1] - R[1][2]) / s;
    q[2] = (R[0][2] - R[2][0]) / s;
    q[3] = (R[1][0] - R[0][1]) / s;
  } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
    float s = sqrtf(1 + R[0][0] - R[1][1] - R[2][2]) * 2;
    q[0] = (R[2][1] - R[1][2]) / s;
    q[1] = s / 4;
    q[2] = (R[0][1] + R[1][0]) / s;
    q[3] = (R[0][2] + R[2][0]) / s;
  } else if (R[1][1] > R[2][2]) {
    float s = sqrtf(1 + R[1][1] - R[0][0] - R[2][2]) * 2;
    q[0] = (R[0][2] - R[2][0]) / s;
    q[1] = (R[0][1] + R[1][0]) / s;
    q[2] = s / 4;
    q[3] = (R[1][2] + R[2][1]) / s;
  } else {
    float s = sqrtf(1 + R[2][2] - R[0][0] - R[1][1]) * 2;
    q[0] = (R[1][0] - R[0][1]) / s;
    q[1] = (R[0][2] + R[2][0]) / s;
    q[2] = (R[1][2] + R[2][1]) / s;
    q[3] = s / 4;
  }
}

static void zeroPosition() {
  for (int i = 0; i < 3; i++) pos[i] = vel[i] = 0;
}

// ---------------------------------------------------------------------------
// MPU-6050 / 6500 over raw I2C

static bool imuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(imuAddr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool imuProbe() {
  for (uint8_t addr : {0x68, 0x69}) {
    Wire.beginTransmission(addr);
    Wire.write(0x75);  // WHO_AM_I
    if (Wire.endTransmission(false) != 0 || Wire.requestFrom(addr, (uint8_t)1) != 1) continue;
    uint8_t who = Wire.read();
    imuAddr = addr;
    imuWrite(0x6B, 0x01);  // wake up, clock from gyro PLL
    imuWrite(0x1A, 0x02);  // DLPF ~94 Hz
    imuWrite(0x1B, 0x18);  // gyro ±2000 °/s
    imuWrite(0x1C, 0x18);  // accel ±16 g
    Serial.printf("imu found at 0x%02x (WHO_AM_I 0x%02x)\n", addr, who);
    return true;
  }
  return false;
}

static bool imuRead(float *aG, float *wDps) {
  Wire.beginTransmission(imuAddr);
  Wire.write(0x3B);  // ACCEL_XOUT_H .. GYRO_ZOUT_L
  if (Wire.endTransmission(false) != 0 || Wire.requestFrom(imuAddr, (uint8_t)14) != 14) return false;
  int16_t v[7];
  for (int i = 0; i < 7; i++) v[i] = (Wire.read() << 8) | Wire.read();
  for (int i = 0; i < 3; i++) {
    aG[i] = v[i] / 2048.0f;
    wDps[i] = v[i + 4] / 16.4f;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Calibration

static void startCalibration() {
  if (!imuAddr) return sendEvent("VAE1 cal noimu");
  calLeft = CAL_SAMPLES;
  for (int i = 0; i < 3; i++) calA[i] = calW[i] = 0;
  calWMax = 0;
}

static void calibrationStep(const float *aG, const float *wDps) {
  for (int i = 0; i < 3; i++) {
    calA[i] += aG[i];
    calW[i] += wDps[i];
  }
  float w = sqrtf(dot3(wDps, wDps));
  if (w > calWMax) calWMax = w;
  if (--calLeft) return;

  float bias[3], up[3];
  for (int i = 0; i < 3; i++) {
    bias[i] = calW[i] / CAL_SAMPLES;
    up[i] = calA[i] / CAL_SAMPLES;
  }
  float bw = sqrtf(dot3(bias, bias));
  // Raw gyro includes its bias (a few °/s), so judge stillness by how far it strays from the average
  if (calWMax > bw + 15) return sendEvent("VAE1 cal moving");

  float m = sqrtf(dot3(up, up));
  gScale = 1.0f / m;
  for (int i = 0; i < 3; i++) gyroBias[i] = bias[i];
  normalize3(up);
  // World Y = the board's Y axis, flattened onto the horizontal
  float yb[3] = {0, 1, 0}, d = dot3(yb, up);
  float y[3] = {yb[0] - d * up[0], yb[1] - d * up[1], yb[2] - d * up[2]};
  normalize3(y);
  for (int i = 0; i < 3; i++) {
    R[1][i] = y[i];
    R[2][i] = up[i];
  }
  orthonormalize();
  zeroPosition();
  calibrated = true;
  ledMode = LED_ON;
  sendEvent("VAE1 cal ok");
  Serial.printf("calibrated: gyro bias %.2f %.2f %.2f dps, accel scale %.3f\n", bias[0], bias[1], bias[2], gScale);
}

// ---------------------------------------------------------------------------
// Throw detection

static void updateThrow(uint32_t now) {
  if (!airborne) {
    if (accelG < FREEFALL_G) {
      if (!lowSince) {
        lowSince = now;
        for (int i = 0; i < 3; i++) {
          lowVel[i] = vel[i];
          lowPos[i] = pos[i];
        }
      }
      if (now - lowSince >= FREEFALL_MS) {
        airborne = true;
        launchAt = lowSince;
        flightMaxSpin = spinDps;
        flightPeakG = 0;
        highSince = 0;
        // Re-run the detection window as pure ballistics from the release, so flight starts exact
        float t = (now - lowSince) / 1000.0f;
        for (int i = 0; i < 3; i++) {
          pos[i] = lowPos[i] + lowVel[i] * t;
          vel[i] = lowVel[i];
        }
        pos[2] -= 0.5f * G * t * t;
        vel[2] -= G * t;
        sendEvent("VAE1 launch %u %d %.3f %.3f %.3f %.3f %.3f %.3f", (unsigned)launchAt, calibrated, lowVel[0],
                  lowVel[1], lowVel[2], lowPos[0], lowPos[1], lowPos[2]);
      }
    } else {
      lowSince = 0;
    }
    return;
  }

  if (spinDps > flightMaxSpin) flightMaxSpin = spinDps;
  if (accelG > LANDED_G) {
    if (!highSince) highSince = now;
    if (accelG > flightPeakG) flightPeakG = accelG;
  } else {
    highSince = 0;
  }
  bool landed = (highSince && now - highSince >= LANDED_MS) || accelG > IMPACT_G;
  if (landed || now - launchAt > 6000) {
    uint32_t landAt = highSince ? highSince : now;
    airborne = false;
    lowSince = 0;
    for (int i = 0; i < 3; i++) vel[i] = 0;  // caught or hit something: it stops
    sendEvent("VAE1 land %u %u %.1f %.0f", (unsigned)landAt, (unsigned)(landAt - launchAt), flightPeakG, flightMaxSpin);
  }
}

// ---------------------------------------------------------------------------

static void imuStep(uint32_t nowUs, uint32_t now) {
  if (!imuAddr) {
    if (now - lastImuProbe > 2000) {
      lastImuProbe = now;
      if (!imuProbe()) Serial.println("imu not found on SDA 32 / SCL 33, retrying");
    }
    return;
  }
  if (nowUs - lastSampleUs < SAMPLE_US) return;
  float dt = (nowUs - lastSampleUs) * 1e-6f;
  if (dt > 0.02f) dt = SAMPLE_US * 1e-6f;  // first sample, or we stalled
  lastSampleUs = nowUs;

  float aG[3], wDps[3];
  if (!imuRead(aG, wDps)) {
    Serial.println("imu read failed, re-probing");
    imuAddr = 0;
    airborne = false;
    return;
  }
  accelG = sqrtf(dot3(aG, aG)) * gScale;
  float w[3];
  for (int i = 0; i < 3; i++) w[i] = (wDps[i] - gyroBias[i]) * DEG2RAD;
  spinDps = sqrtf(dot3(w, w)) / DEG2RAD;

  if (calLeft) calibrationStep(aG, wDps);

  // Board acceleration in m/s²
  float a[3] = {aG[0] * gScale * G, aG[1] * gScale * G, aG[2] * gScale * G};

  // Near 1 g and not spinning hard: gently pull our "up" toward what gravity says (fixes gyro tilt drift)
  if (calibrated && fabsf(accelG - 1) < 0.08f && spinDps < 60) {
    float meas[3] = {a[0], a[1], a[2]}, e[3];
    normalize3(meas);
    cross3(meas, R[2], e);
    for (int i = 0; i < 3; i++) w[i] += 0.5f * e[i];
  }

  // Integrate the gyro: each world axis (in board coords) turns by -w, i.e. r += dt * (r × w)
  for (int r = 0; r < 3; r++) {
    float c[3];
    cross3(R[r], w, c);
    for (int i = 0; i < 3; i++) R[r][i] += dt * c[i];
  }
  orthonormalize();

  // World-frame acceleration without gravity
  for (int i = 0; i < 3; i++) linAcc[i] = dot3(R[i], a);
  linAcc[2] -= G;
  bool quiet = sqrtf(dot3(linAcc, linAcc)) < 1.0f && spinDps < 25;
  stillRun = quiet ? stillRun + 1 : 0;
  still = stillRun >= STILL_SAMPLES;

  updateThrow(now);

  if (calibrated && !calLeft) {
    // In free fall the only force is gravity: use it exactly instead of the (spin-polluted) sensor
    float acc[3] = {linAcc[0], linAcc[1], linAcc[2]};
    if (airborne) acc[0] = acc[1] = 0, acc[2] = -G;
    for (int i = 0; i < 3; i++) vel[i] += acc[i] * dt;
    if (still && !airborne) vel[0] = vel[1] = vel[2] = 0;  // zero-velocity update
    float speed = sqrtf(dot3(vel, vel));
    if (speed > MAX_SPEED) for (int i = 0; i < 3; i++) vel[i] *= MAX_SPEED / speed;
    for (int i = 0; i < 3; i++) pos[i] += vel[i] * dt;
  }

  if (imuStream && now - lastImuPrint >= 100) {
    lastImuPrint = now;
    Serial.printf("imu a=%.2fg spin=%.0fdps lin=%.2f %.2f %.2f pos=%.2f %.2f %.2f %s%s\n", accelG, spinDps, linAcc[0],
                  linAcc[1], linAcc[2], pos[0], pos[1], pos[2], still ? "STILL " : "", airborne ? "AIRBORNE" : "");
  }
}

static void handleCommand(const char *cmd) {
  if (strcmp(cmd, "imu") == 0) {
    imuStream = !imuStream;
  } else if (strcmp(cmd, "cal") == 0) {
    startCalibration();
  } else if (strcmp(cmd, "zero") == 0) {
    zeroPosition();
    sendEvent("VAE1 zero");
  } else {
    for (int i = 0; i < 3; i++) {
      if (strcmp(cmd, LED_NAMES[i]) == 0) ledMode = (LedMode)i;
    }
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  pinMode(LED_ONBOARD, OUTPUT);
  pinMode(LED_EXTERNAL, OUTPUT);

  Wire.begin(IMU_SDA, IMU_SCL, 400000);
  imuProbe();

  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("VisionAiry beacon, joining \"%s\"\n", WIFI_SSID);
}

void loop() {
  uint32_t now = millis();
  imuStep(micros(), now);
  updateLed(now);

  if (WiFi.status() == WL_CONNECTED) {
    if (!udpStarted) {
      udpStarted = udp.begin(UDP_PORT);
      Serial.printf("wifi up ip=%s rssi=%d\n", WiFi.localIP().toString().c_str(), WiFi.RSSI());
    }
    int n = udp.parsePacket();
    if (n > 0) {
      char buf[16];
      int len = udp.read(buf, sizeof(buf) - 1);
      buf[len > 0 ? len : 0] = 0;
      handleCommand(buf);
    }
  } else if (udpStarted) {
    udp.stop();
    udpStarted = false;
  }

  char line[128];
  if (now - lastState >= 33) {
    lastState = now;
    float q[4];
    quaternion(q);
    int flags = (still ? 1 : 0) | (airborne ? 2 : 0) | (calibrated ? 4 : 0);
    snprintf(line, sizeof(line), "VAS1 %u %.3f %.3f %.3f %.3f %.3f %.3f %.3f %d", (unsigned)now, pos[0], pos[1], pos[2],
             q[0], q[1], q[2], q[3], flags);
    sendLine(line);
  }

  if (now - lastBeat >= 250) {
    lastBeat = now;
    // Heading / pitch of the board's Y axis in the calibrated world, so you can sanity-check calibration
    float yx = R[0][1], yy = R[1][1], yz = R[2][1];
    snprintf(line, sizeof(line), "VAB1 %u %d %s %d %.2f %.0f %d %d %.0f %.0f", (unsigned)now,
             udpStarted ? WiFi.RSSI() : 0, LED_NAMES[ledMode], imuAddr ? 1 : 0, accelG, spinDps, airborne, calibrated,
             atan2f(yx, yy) / DEG2RAD, asinf(fmaxf(-1, fminf(1, yz))) / DEG2RAD);
    sendLine(line);
  }

  static char cmd[16];
  static int ln = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      cmd[ln] = 0;
      if (ln) handleCommand(cmd);
      ln = 0;
    } else if (ln < 15) {
      cmd[ln++] = c;
    }
  }
}
