// VisionAiry XRAY: Wi-Fi CSI motion sensing through walls.
//
// The phone hotspot is the transmitter, this ESP32 is the receiver. We ping the hotspot ~100x/s
// so it keeps sending us frames, and read the Channel State Information (per-subcarrier amplitude)
// of each one. A body moving through the link scatters the signal and makes those amplitudes
// fluctuate. We measure that fluctuation ("turbulence"), compare it to an empty-room baseline,
// and call it a contact when it rises well above baseline.
//
// Serial protocol (921600 baud), one line per message:
//   S <text>                                   status
//   C <rssi> <hex amplitudes>                  one CSI frame, one byte per used subcarrier
//   M <turb> <score> <present> <pps> <cal> <thresh>   detector state, 10x/s
// Commands in: "c" recalibrate, "t<float>" set threshold, "r" toggle raw C lines.

#include <WiFi.h>
#include "esp_wifi.h"
#include "ping/ping_sock.h"
#include "secrets.h"  // generated from xray/.env by flash.sh

#define LED_PIN 2

// LLTF gives 64 subcarriers stored as [imag, real] int8 pairs: indices 0..31 are subcarriers
// 0..31, 32..63 are -32..-1. Skip DC (0), the guard bands (27..37) and index 1, which some
// frames mark invalid. That leaves 51 data subcarriers.
static const uint8_t SUB_FIRST_A = 2, SUB_LAST_A = 26, SUB_FIRST_B = 38, SUB_LAST_B = 63;
static const int NSUB = (SUB_LAST_A - SUB_FIRST_A + 1) + (SUB_LAST_B - SUB_FIRST_B + 1);

static const int WINDOW = 50;            // ~0.5 s of frames for the variance window
static const uint32_t CAL_MS = 8000;     // empty-room calibration length
static const uint32_t HOLD_MS = 1500;    // keep "present" this long after score drops
static const uint32_t REPORT_MS = 100;

struct CsiFrame {
  int8_t rssi;
  uint8_t amp[NSUB];
};

static QueueHandle_t csiQueue;
static uint8_t apMac[6];
static volatile bool apKnown = false;
static esp_ping_handle_t pingHandle = nullptr;

// Detector state
static float ring[WINDOW][NSUB];
static float sum[NSUB], sumSq[NSUB];
static int ringPos = 0, ringFill = 0;
static float turbulence = 0, scoreEma = 0, baseline = 0;
static double calSum = 0;
static uint32_t calCount = 0, calStart = 0;
static bool calibrating = true, present = false, rawStream = true;
static float threshold = 2.0f;
static uint32_t lastAbove = 0, framesThisSec = 0, pps = 0, ppsTick = 0, lastReport = 0;

// Runs in the Wi-Fi task: keep it short and hand off through the queue.
static void onCsi(void *ctx, wifi_csi_info_t *info) {
  if (!apKnown || memcmp(info->mac, apMac, 6) != 0 || info->len < 128) return;
  CsiFrame f;
  f.rssi = info->rx_ctrl.rssi;
  int k = 0;
  for (int s = 0; s < 64; s++) {
    if (!((s >= SUB_FIRST_A && s <= SUB_LAST_A) || (s >= SUB_FIRST_B && s <= SUB_LAST_B))) continue;
    float im = info->buf[2 * s], re = info->buf[2 * s + 1];
    float a = sqrtf(im * im + re * re);
    f.amp[k++] = a > 255 ? 255 : (uint8_t)a;
  }
  xQueueSend(csiQueue, &f, 0);
}

static void startCalibration() {
  calibrating = true;
  calSum = 0;
  calCount = 0;
  calStart = millis();
  present = false;
  Serial.println("S calibrating: keep the path between phone and ESP32 clear");
}

static void resetWindow() {
  memset(sum, 0, sizeof(sum));
  memset(sumSq, 0, sizeof(sumSq));
  ringPos = ringFill = 0;
}

static void startCsi() {
  wifi_csi_config_t cfg = {};
  cfg.lltf_en = true;
  cfg.htltf_en = false;
  cfg.stbc_htltf2_en = false;
  cfg.ltf_merge_en = true;
  cfg.channel_filter_en = false;
  cfg.manu_scale = false;
  cfg.shift = 0;
  esp_err_t e1 = esp_wifi_set_csi_config(&cfg);
  esp_err_t e2 = esp_wifi_set_csi_rx_cb(onCsi, nullptr);
  esp_err_t e3 = esp_wifi_set_csi(true);
  Serial.printf("S csi config=%s cb=%s enable=%s\n", esp_err_to_name(e1), esp_err_to_name(e2),
                esp_err_to_name(e3));
}

// Pinging the hotspot makes it answer us constantly, which is what gives us a steady CSI stream.
static void startPing(IPAddress gw) {
  if (pingHandle) {
    esp_ping_stop(pingHandle);
    esp_ping_delete_session(pingHandle);
    pingHandle = nullptr;
  }
  esp_ping_config_t cfg = ESP_PING_DEFAULT_CONFIG();
  IP_ADDR4(&cfg.target_addr, gw[0], gw[1], gw[2], gw[3]);
  cfg.count = ESP_PING_COUNT_INFINITE;
  cfg.interval_ms = 10;
  cfg.timeout_ms = 1000;
  cfg.data_size = 16;
  esp_ping_callbacks_t cbs = {};
  if (esp_ping_new_session(&cfg, &cbs, &pingHandle) == ESP_OK) {
    esp_ping_start(pingHandle);
    Serial.printf("S pinging %s every %d ms\n", gw.toString().c_str(), (int)cfg.interval_ms);
  } else {
    Serial.println("S ping session failed");
  }
}

static void onWifiEvent(arduino_event_id_t event, arduino_event_info_t info) {
  switch (event) {
    case ARDUINO_EVENT_WIFI_STA_GOT_IP:
      memcpy(apMac, WiFi.BSSID(), 6);
      apKnown = true;
      Serial.printf("S connected ip=%s gw=%s ch=%d rssi=%d bssid=%s\n", WiFi.localIP().toString().c_str(),
                    WiFi.gatewayIP().toString().c_str(), WiFi.channel(), WiFi.RSSI(), WiFi.BSSIDstr().c_str());
      startPing(WiFi.gatewayIP());
      resetWindow();
      startCalibration();
      break;
    case ARDUINO_EVENT_WIFI_STA_DISCONNECTED:
      apKnown = false;
      Serial.printf("S disconnected reason=%d, retrying\n", info.wifi_sta_disconnected.reason);
      break;
    default:
      break;
  }
}

static void processFrame(const CsiFrame &f) {
  // Normalise each frame by its mean amplitude so automatic gain changes don't look like motion.
  float mean = 0;
  for (int i = 0; i < NSUB; i++) mean += f.amp[i];
  mean /= NSUB;
  if (mean < 1) return;

  float *slot = ring[ringPos];
  for (int i = 0; i < NSUB; i++) {
    float v = f.amp[i] / mean;
    if (ringFill == WINDOW) {
      sum[i] -= slot[i];
      sumSq[i] -= slot[i] * slot[i];
    }
    slot[i] = v;
    sum[i] += v;
    sumSq[i] += v * v;
  }
  ringPos = (ringPos + 1) % WINDOW;
  if (ringFill < WINDOW) ringFill++;
  if (ringFill < WINDOW) return;

  // Turbulence = average standard deviation across subcarriers over the window.
  float t = 0;
  for (int i = 0; i < NSUB; i++) {
    float m = sum[i] / WINDOW;
    float var = sumSq[i] / WINDOW - m * m;
    t += var > 0 ? sqrtf(var) : 0;
  }
  turbulence = t / NSUB;

  uint32_t now = millis();
  if (calibrating) {
    calSum += turbulence;
    calCount++;
    if (now - calStart >= CAL_MS && calCount > 50) {
      baseline = calSum / calCount;
      calibrating = false;
      scoreEma = 1;
      Serial.printf("S calibrated baseline=%.5f from %u frames\n", baseline, (unsigned)calCount);
    }
    return;
  }

  float score = baseline > 0 ? turbulence / baseline : 0;
  scoreEma += 0.15f * (score - scoreEma);
  if (scoreEma > threshold) {
    present = true;
    lastAbove = now;
  } else if (present && scoreEma < threshold * 0.75f && now - lastAbove > HOLD_MS) {
    present = false;
  }
}

static void handleSerial() {
  static char buf[32];
  static int n = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c != '\n' && c != '\r' && n < 31) {
      buf[n++] = c;
      continue;
    }
    buf[n] = 0;
    if (n > 0) {
      if (buf[0] == 'c') {
        resetWindow();
        startCalibration();
      } else if (buf[0] == 't') {
        float t = atof(buf + 1);
        if (t > 1.0f && t < 50.0f) threshold = t;
        Serial.printf("S threshold=%.2f\n", threshold);
      } else if (buf[0] == 'r') {
        rawStream = !rawStream;
        Serial.printf("S raw=%d\n", rawStream);
      }
    }
    n = 0;
  }
}

void setup() {
  Serial.begin(921600);
  pinMode(LED_PIN, OUTPUT);
  delay(300);
  Serial.println("S VisionAiry XRAY booting");

  csiQueue = xQueueCreate(32, sizeof(CsiFrame));
  WiFi.onEvent(onWifiEvent);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // power save would drop most frames
  WiFi.setAutoReconnect(true);
  startCsi();
  Serial.printf("S joining \"%s\"\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
}

void loop() {
  static const char HEX_CHARS[] = "0123456789abcdef";
  static char line[16 + NSUB * 2];

  CsiFrame f;
  while (xQueueReceive(csiQueue, &f, 0) == pdTRUE) {
    framesThisSec++;
    processFrame(f);
    if (rawStream) {
      int p = snprintf(line, sizeof(line), "C %d ", f.rssi);
      for (int i = 0; i < NSUB; i++) {
        line[p++] = HEX_CHARS[f.amp[i] >> 4];
        line[p++] = HEX_CHARS[f.amp[i] & 15];
      }
      line[p] = 0;
      Serial.println(line);
    }
  }

  uint32_t now = millis();
  if (now - ppsTick >= 1000) {
    pps = framesThisSec;
    framesThisSec = 0;
    ppsTick = now;
    if (WiFi.status() != WL_CONNECTED) Serial.printf("S waiting for wifi (status %d)\n", WiFi.status());
  }
  if (now - lastReport >= REPORT_MS) {
    lastReport = now;
    Serial.printf("M %.5f %.3f %d %u %d %.2f\n", turbulence, calibrating ? 0.0f : scoreEma, present,
                  (unsigned)pps, calibrating, threshold);
  }

  // LED: solid when someone is detected, slow blink while calibrating, off when clear.
  digitalWrite(LED_PIN, present || (calibrating && (now / 250) % 2));
  handleSerial();
  delay(1);
}
