#include <WiFi.h>
#include <PubSubClient.h>

// --- Configuration ---
const char* ssid        = "MATSAKONZ";
const char* password    = "First1234";
const char* mqtt_server = "broker.emqx.io";  // IP address of computer running broker (or localhost IP)
const int   mqtt_port   = 1883;
const char* mqtt_topic  = "security/threats";

#define BUZZER_PIN 18
#define IS_PASSIVE_BUZZER false // Set to true if using a passive buzzer
// Most 3-pin / module buzzers are Active-LOW (LOW turns sound ON, HIGH turns sound OFF)
#define BUZZER_ACTIVE_LOW true

#define BUZZER_ON  (BUZZER_ACTIVE_LOW ? LOW : HIGH)
#define BUZZER_OFF (BUZZER_ACTIVE_LOW ? HIGH : LOW)

WiFiClient espClient;
PubSubClient client(espClient);

void triggerAlarm() {
  Serial.println("🚨 Threat alert received! Triggering buzzer...");
  for (int i = 0; i < 3; i++) {
    if (IS_PASSIVE_BUZZER) {
      tone(BUZZER_PIN, 2000, 200); // 2000 Hz for 200ms
      delay(250);
      noTone(BUZZER_PIN);
    } else {
      digitalWrite(BUZZER_PIN, BUZZER_ON);
      delay(150);
      digitalWrite(BUZZER_PIN, BUZZER_OFF);
      delay(100);
    }
  }
  // Ensure buzzer is explicitly OFF after alarm completes
  if (!IS_PASSIVE_BUZZER) {
    digitalWrite(BUZZER_PIN, BUZZER_OFF);
  }
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived on topic: ");
  Serial.println(topic);

  // If any message received on security/threats, trigger buzzer
  triggerAlarm();
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Connecting to MQTT...");
    String clientId = "ESP32_Buzzer_" + String(random(0xffff), HEX);
    if (client.connect(clientId.c_str())) {
      Serial.println("connected!");
      client.subscribe(mqtt_topic);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying in 5 seconds");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, BUZZER_OFF);

  // Connect to Wi-Fi
  Serial.print("Connecting to Wi-Fi");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected! IP: " + WiFi.localIP().toString());

  // Configure MQTT
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(mqttCallback);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}
