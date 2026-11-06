/*
  Retrofit Monitor - ESP32 firmware, step 1: current -> features -> CSV

  What this program does, every 1 second (one "window"):
    1. Reads the CT clamp signal 1000 times (1000 samples per second).
    2. Removes the DC shift (the 1.65 V offset from the R1/R2 divider).
    3. Converts volts to amps.
    4. Calculates RMS, peak and crest factor of that window.
    5. Prints one CSV line to the Serial Monitor AND saves it on the ESP32's
       own flash memory (works with no network = offline storage, FR4/FR5).

  The ESP32 does NOT store a "state". It cannot know the state. It only measures.
  The state is predicted later from these numbers (by the rule or the ML model).

  Hardware (planned):
    CT clamp -> burden resistor (220 ohm) -> R1/R2 divider (1.65 V bias) -> GPIO34
    GPIO34 is ADC1 and input-only, which is the correct kind of pin for this.

  No hardware yet? Set SIMULATE_INPUT to 1 below. The program then makes a fake
  50 Hz current by itself, so you can test everything on Wokwi or a bare board.
*/

#include <Arduino.h>
#include <LittleFS.h>   // simple file system inside the ESP32 flash memory

// ---------------------------------------------------------------------------
// Settings (the same numbers as your Python script)
// ---------------------------------------------------------------------------
#define SIMULATE_INPUT 1          // 1 = fake signal, 0 = read the real ADC pin

const int   ADC_PIN        = 34;       // CT clamp signal comes in here
const int   SAMPLE_RATE_HZ = 1000;     // samples per second
const int   WINDOW         = 1000;     // samples per window (= 1 second)
const float BURDEN_OHM     = 220.0;    // burden resistor
const float CT_RATIO       = 2000.0;   // CT turns ratio 2000:1
const float MAINS_HZ       = 50.0;     // Sri Lanka

const char* LOG_FILE = "/rms_log.csv";   // file on the ESP32 flash

// For SIMULATE_INPUT only: the fake machine's current in amps (RMS).
// Try 0.05 (Idle), 1.8 (Wash), 2.6 (Spin), 3.6 (Overload) and watch the output.
const float SIM_AMPS = 1.8;

// ---------------------------------------------------------------------------
// Variables
// ---------------------------------------------------------------------------
const unsigned long SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;  // 1000 us

float samples_mV[WINDOW];     // one window of raw readings, in millivolts
int   sampleIndex = 0;        // how many samples we have collected so far
long  windowCount = 0;        // window number, like "window" in your Python table
unsigned long nextTick = 0;   // when the next sample is due (microseconds)

// ---------------------------------------------------------------------------
// Read ONE sample, in millivolts
// ---------------------------------------------------------------------------
float readSample_mV() {
#if SIMULATE_INPUT
  // Fake CT signal: 1650 mV bias + a 50 Hz sine wave + a little noise.
  // Same maths as generate_fake_signal.py, just going backwards from amps.
  static long n = 0;                          // sample counter
  float t = n / (float)SAMPLE_RATE_HZ;        // time in seconds
  n++;
  float amps  = SIM_AMPS * 1.41421356f * sinf(2.0f * PI * MAINS_HZ * t);  // RMS -> peak
  float volts = amps / CT_RATIO * BURDEN_OHM;                              // CT, then burden: V = I x R
  float noise = ((float)random(-300, 301)) / 100.0f;                       // about +-3 mV
  return 1650.0f + volts * 1000.0f + noise;
#else
  // Real hardware. analogReadMilliVolts() uses the factory calibration stored
  // in the chip, so it is more accurate than converting raw counts yourself.
  return (float)analogReadMilliVolts(ADC_PIN);
#endif
}

// ---------------------------------------------------------------------------
// Called once per full window: calculate features and save them
// ---------------------------------------------------------------------------
void processWindow() {
  // --- 1. Measure the DC shift of THIS window -------------------------------
  // Your Python script used a fixed 1.65 V. A real ESP32's bias is never
  // exactly 1.650 V (resistor tolerance, ADC error, temperature), so we
  // measure it: the average of a whole number of AC cycles is the DC level.
  // 1000 samples at 1 kHz = exactly 50 cycles of 50 Hz, so this works.
  float sum = 0;
  for (int i = 0; i < WINDOW; i++) sum += samples_mV[i];
  float dc_mV = sum / WINDOW;

  // --- 2. Remove the DC, convert to amps, build RMS and peak ---------------
  float sumSquares = 0;
  float peak = 0;
  for (int i = 0; i < WINDOW; i++) {
    float volts = (samples_mV[i] - dc_mV) / 1000.0f;     // AC part only, in volts
    float amps  = volts / BURDEN_OHM * CT_RATIO;          // volts -> amps
    sumSquares += amps * amps;                            // square
    if (fabsf(amps) > peak) peak = fabsf(amps);           // biggest swing, either side
  }
  float rms   = sqrtf(sumSquares / WINDOW);               // mean -> square root
  float crest = (rms > 0.0f) ? peak / rms : 0.0f;         // avoid dividing by zero

  // --- 3. Make one CSV line --------------------------------------------------
  // columns: window, time_ms, dc_v, rms_a, peak_a, crest
  // dc_v is a health check: it should stay near 1.65. If it drifts, the bias
  // circuit or the ADC has a problem.
  char line[96];
  snprintf(line, sizeof(line), "%ld,%lu,%.3f,%.3f,%.3f,%.3f",
           windowCount, millis(), dc_mV / 1000.0f, rms, peak, crest);

  Serial.println(line);                                   // to the computer

  // --- 4. Save in the flash file (offline storage) ---------------------------
  // Open, write, close each time, so a power cut loses at most one line.
  File f = LittleFS.open(LOG_FILE, FILE_APPEND);
  if (f) {
    f.println(line);
    f.close();
  } else {
    Serial.println("ERROR: cannot write the log file");
  }

  windowCount++;
}

// ---------------------------------------------------------------------------
// setup() runs once when the board starts
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(500);

  // Start the file system. 'true' formats it the first time it is used.
  if (!LittleFS.begin(true)) {
    Serial.println("ERROR: file system failed to start");
  }

  // Write the CSV header only if the file does not exist yet.
  if (!LittleFS.exists(LOG_FILE)) {
    File f = LittleFS.open(LOG_FILE, FILE_WRITE);
    if (f) {
      f.println("window,time_ms,dc_v,rms_a,peak_a,crest");
      f.close();
    }
  }

#if !SIMULATE_INPUT
  analogReadResolution(12);                  // 12-bit ADC (0..4095)
  analogSetPinAttenuation(ADC_PIN, ADC_11db); // input range about 0..3.1 V
#endif

  Serial.println("window,time_ms,dc_v,rms_a,peak_a,crest");
  nextTick = micros();
}

// ---------------------------------------------------------------------------
// loop() runs forever: take a sample exactly every 1000 microseconds
// ---------------------------------------------------------------------------
void loop() {
  // Wait until it is time for the next sample. (long) makes the comparison
  // still correct when micros() wraps around after about 71 minutes.
  while ((long)(micros() - nextTick) < 0) { }
  nextTick += SAMPLE_PERIOD_US;

  samples_mV[sampleIndex++] = readSample_mV();

  if (sampleIndex >= WINDOW) {
    processWindow();
    sampleIndex = 0;
    // Saving to flash takes a few milliseconds. Restart the timing here so we
    // do not fire many late samples in a burst to "catch up".
    nextTick = micros() + SAMPLE_PERIOD_US;
  }
}
