#include <IBusBM.h>
#include <RoboClaw.h>
#include <WiFi.h>

// WiFi settings
const char* ssid     = "Linksys07479";
const char* password = "bs8hdvwgnx";
#define TCP_PORT 10001

// Hardware configuration
IBusBM    ibus;
RoboClaw  roboclaw(&Serial, 10000);
#define Mot1 0x80 // Left wheel motor address
#define Mot2 0x81 // Right wheel motor address

// TCP server
WiFiServer server(TCP_PORT);
WiFiClient client;
bool     computerControl = false;

#define CMD_BUFFER_SIZE 256
char     cmdBuffer[CMD_BUFFER_SIZE];
int      cmdIndex = 0;

// Debug output
bool debugOutput = true;

// Command handling
unsigned long lastCommandTime = 0;
bool motorActive = false;
unsigned long motorStopTime = 0;
bool needToStop = false;

// Computer control timeout
unsigned long lastComputerCommandTime = 0;
const unsigned long COMPUTER_TIMEOUT = 3000000; // Set very long to prevent auto-timeout

// Process commands from the computer
void processCommand(String cmd) {
  cmd.trim();
  
  // Update the last command time for timeout detection
  lastComputerCommandTime = millis();

  if (debugOutput) {
    Serial.print("Received command: ");
    Serial.println(cmd);
  }

  // Check for STOP command first - highest priority
  if (cmd.startsWith("STOP")) {
    roboclaw.ForwardBackwardM1(Mot1, 64); // Neutral position
    roboclaw.ForwardBackwardM1(Mot2, 64);
    motorActive = false;
    client.println("OK STOP");
    return;
  }

  // Check for SYS DISCONNECT command
  if (cmd.startsWith("SYS DISCONNECT")) {
    roboclaw.ForwardBackwardM1(Mot1, 64); // Stop motors
    roboclaw.ForwardBackwardM1(Mot2, 64);
    client.println("OK DISCONNECT");
    computerControl = false;
    client.stop();
    Serial.println("Computer control explicitly ended");
    return;
  }
  
  if (cmd.startsWith("SYS PING")) {
    client.println("OK PING");
    return;
  }

  if (cmd.startsWith("SYS CAL")) {
    client.println("OK SYS CAL");
    return;
  }
  
  if (cmd.startsWith("MMW !MG")) {
    client.println("OK MMW !MG");
    return;
  }
  
  if (cmd.startsWith("MMW !M")) {
    // Format: MMW !M <left> <right>
    int p1 = cmd.indexOf(' ', 6);
    int p2 = cmd.indexOf(' ', p1 + 1);
    if (p1 > 0 && p2 > 0) {
      // Extract left and right motor speed values
      String leftStr = cmd.substring(p1+1, p2);
      String rightStr = cmd.substring(p2+1);

      // Remove any fractional part
      int dotPos1 = leftStr.indexOf('.');
      if (dotPos1 >= 0) {
        leftStr = leftStr.substring(0, dotPos1);
      }

      int dotPos2 = rightStr.indexOf('.');
      if (dotPos2 >= 0) {
        rightStr = rightStr.substring(0, dotPos2);
      }

      // Convert to integer
      int leftSp = leftStr.toInt();   // -250..250
      int rightSp = rightStr.toInt(); // -250..250

      if (debugOutput) {
        Serial.print("Left motor: "); Serial.print(leftSp);
        Serial.print(", Right motor: "); Serial.println(rightSp);
      }

      // Check if this is a stop command (both motors at 0)
      if (leftSp == 0 && rightSp == 0) {
        motorActive = false;
      } else {
        motorActive = true;
        lastCommandTime = millis();
      }

      // Map to 0..127
      int m1 = map(leftSp, -250, 250, 0, 127);
      int m2 = map(rightSp, -250, 250, 0, 127);
      
      if (debugOutput) {
        Serial.print("Mapped Left: "); Serial.print(m1);
        Serial.print(", Right: "); Serial.println(m2);
      }

      // Call ForwardBackwardM1 for both addresses
      roboclaw.ForwardBackwardM1(Mot1, m1);
      roboclaw.ForwardBackwardM1(Mot2, m2);

      client.println("OK MMW !M");
    } else {
      client.println("ERROR: Invalid command format");
      if (debugOutput) {
        Serial.println("Command format error");
      }
    }
    return;
  }

  client.println("UNKNOWN COMMAND");
  if (debugOutput) {
    Serial.println("Unknown command");
  }
}

// Function declarations
int readChannel(byte channelInput, int minLimit, int maxLimit, int defaultValue);
bool readSwitch(byte channelInput, bool defaultValue);

// Define the readChannel function
int readChannel(byte channelInput, int minLimit, int maxLimit, int defaultValue) {
  uint16_t ch = ibus.readChannel(channelInput);
  if (ch < 100) return defaultValue;
  return map(ch, 1000, 2000, minLimit, maxLimit);
}

bool readSwitch(byte channelInput, bool defaultValue) {
  int intDefaultValue = defaultValue ? 100 : 0;
  int ch = readChannel(channelInput, 0, 100, intDefaultValue);
  return (ch > 50);
}

// Non-blocking WiFi startup
void setupWiFi() {
  WiFi.begin(ssid, password);
  server.begin();
  Serial.println("TCP server started on port 10001");
}

void setup() {
  // Configure static IP before connecting
  IPAddress staticIP(192, 168, 1, 104);
  IPAddress gateway(192, 168, 1, 1);
  IPAddress subnet(255, 255, 255, 0);
  WiFi.config(staticIP, gateway, subnet);
  
  Serial.begin(115200);
  Serial.println("\nESP32 Robot Controller Starting...");
  
  roboclaw.begin(115200);

  // Initialize iBus without timer
  ibus.begin(Serial2, IBUSBM_NOTIMER);

  setupWiFi();
  delay(90);
  
  // Wait for WiFi connection
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("\nWiFi connected, IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi connection failed, continuing with remote controller mode");
  }
}

void loop() {
  // Process iBus signals for remote control
  ibus.loop();

  // Handle TCP client
  if (WiFi.status() == WL_CONNECTED) {
    // Check for new clients
    WiFiClient nc = server.available();
    if (nc) {
      if (!client || !client.connected()) {
        if (client) client.stop();
        client = nc;
        client.setNoDelay(true); // Important for reducing latency
        computerControl = true;
        cmdIndex = 0;
        lastComputerCommandTime = millis(); // Reset timeout on new connection
        Serial.println("Client connected");
      } else {
        nc.stop();
      }
    }
    
    // Check if client disconnected
    if (client && !client.connected()) {
      client.stop();
      computerControl = false;
      motorActive = false;
      Serial.println("Client disconnected");
      // Stop motors on disconnect
      roboclaw.ForwardBackwardM1(Mot1, 64);
      roboclaw.ForwardBackwardM1(Mot2, 64);
    }
    
    // Check for motor timeout (200ms without new command)
    if (computerControl && motorActive && millis() - lastCommandTime > 200) {
      // If no new commands for 200ms, schedule a motor stop
      if (!needToStop) {
        needToStop = true;
        motorStopTime = millis();
      }
    }
    
    // Stop motors if scheduled and time reached
    if (needToStop && millis() >= motorStopTime) {
      roboclaw.ForwardBackwardM1(Mot1, 64);
      roboclaw.ForwardBackwardM1(Mot2, 64);
      motorActive = false;
      needToStop = false;
      Serial.println("Motor timeout - stopping");
    }
    
    // Process incoming commands
    if (computerControl && client && client.connected()) {
      while (client.available()) {
        char c = client.read();
        if (cmdIndex < CMD_BUFFER_SIZE-1) cmdBuffer[cmdIndex++] = c;
        if (c=='\n' || c=='\r') {
          if (cmdIndex > 1) { // Ensure not empty command
            cmdBuffer[cmdIndex] = '\0';
            processCommand(String(cmdBuffer));
          }
          cmdIndex = 0;
        }
      }
    }
  }

  // If not computer-controlled, use remote controller
  if (!computerControl) {
    int value = readChannel(1, 0, 126, 64);
    int turn  = map(readChannel(0, 0, 126, 64), 0, 126, -63, 63);
    if (value == 63) value = 64;

    int LSpeed = constrain(value + turn, 0, 126);
    int RSpeed = constrain(value - turn, 0, 126);

    // Drive both motors using ForwardBackwardM1
    roboclaw.ForwardBackwardM1(Mot1, LSpeed);
    roboclaw.ForwardBackwardM1(Mot2, RSpeed);
  }

  // Short delay to prevent CPU hogging
  delay(5); // Reduced from 10ms to 5ms for better responsiveness
}