// VisionAiry bridge: a second ESP32 plugged into the laptop by USB. It joins the drone's Wi-Fi (the same network
// the beacon joins) and relays both ways, so the laptop can stay on its own Wi-Fi with internet.
//   beacon -> laptop: every VA* UDP packet on port 4210 is printed as one serial line (115200).
//   laptop -> beacon: every serial line ("cal", "zero", ...) is sent to the beacon over UDP.
// To the server this looks exactly like the beacon plugged in by USB, so server.js needs no changes.
// LED: off = no Wi-Fi, slow blink = Wi-Fi up but no beacon heard, solid = relaying.

#include <WiFi.h>
#include <WiFiUdp.h>
#include "secrets.h"  // generated from esp32/.env by flash.sh

#define LED_ONBOARD 2
#define UDP_PORT 4210
#define SERIAL_BAUD 115200  // higher rates corrupt on this board's USB chip

static WiFiUDP udp;
static bool udpStarted = false;
static IPAddress beaconIp;
static uint32_t beaconSeen = 0, lastStatus = 0;
static char cmd[64];
static size_t cmdLen = 0;

static bool beaconOnline(uint32_t now) { return beaconSeen && now - beaconSeen < 1000; }

void setup() {
  Serial.begin(SERIAL_BAUD);
  pinMode(LED_ONBOARD, OUTPUT);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // modem sleep adds 100+ ms of latency
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.printf("VisionAiry bridge, joining \"%s\"\n", WIFI_SSID);
}

void loop() {
  uint32_t now = millis();

  if (WiFi.status() == WL_CONNECTED) {
    if (!udpStarted) {
      udpStarted = udp.begin(UDP_PORT);
      Serial.printf("wifi up ip=%s drone=%s rssi=%d\n", WiFi.localIP().toString().c_str(), WiFi.gatewayIP().toString().c_str(), WiFi.RSSI());
    }
    while (udp.parsePacket() > 0) {
      char buf[256];
      int len = udp.read(buf, sizeof(buf) - 1);
      if (len <= 0 || udp.remoteIP() == WiFi.localIP()) continue;  // our own command broadcast
      while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) len--;
      buf[len] = 0;
      if (strncmp(buf, "VA", 2) != 0) continue;
      if (!beaconOnline(now)) Serial.printf("beacon at %s\n", udp.remoteIP().toString().c_str());
      beaconIp = udp.remoteIP();
      beaconSeen = now;
      Serial.println(buf);
    }
  } else if (udpStarted) {
    udp.stop();
    udpStarted = false;
  }

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c != '\n') {
      if (cmdLen < sizeof(cmd) - 1) cmd[cmdLen++] = c;
      continue;
    }
    cmd[cmdLen] = 0;
    if (cmdLen && udpStarted) {
      // Unicast once we know the beacon (Wi-Fi retries those), broadcast until then
      udp.beginPacket(beaconSeen ? beaconIp : IPAddress(255, 255, 255, 255), UDP_PORT);
      udp.print(cmd);
      udp.endPacket();
    }
    cmdLen = 0;
  }

  if (now - lastStatus >= 5000) {
    lastStatus = now;
    if (WiFi.status() != WL_CONNECTED) Serial.printf("bridge: no wifi (\"%s\")\n", WIFI_SSID);
    else if (!beaconOnline(now)) Serial.printf("bridge: wifi up rssi=%d, no beacon heard\n", WiFi.RSSI());
  }

  bool led = WiFi.status() != WL_CONNECTED ? false : beaconOnline(now) ? true : (now / 500) % 2;
  digitalWrite(LED_ONBOARD, led);
}
