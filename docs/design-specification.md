# Design Specification

This document defines what the design must do and how success is measured. It answers the need described in [problem-statement.md](problem-statement.md). The problem statement says what is wrong; this document says what the design must achieve.

Status: **draft**. Values marked (assumed) are starting targets to be revised after testing or after a factory visit.

## 1. Objectives

1. Know at all times whether a machine is running, idle, or faulty.
2. Keep recording when the network is unavailable.
3. Detect abnormal machine behaviour early enough to act before a breakdown.
4. Be affordable and simple enough for a small factory to adopt.

## 2. Functional requirements

| ID | Requirement |
|----|-------------|
| FR1 | Measure the machine's electrical current without cutting or modifying its wiring. |
| FR2 | Measure the machine's vibration. |
| FR3 | Classify the machine state as Idle, Wash, Rinse, Drain, Spin, or Fault. |
| FR4 | Store timestamped readings and states locally on the device. |
| FR5 | Keep storing while the network is down, and send the stored data when it returns. |
| FR6 | Raise an alert when a fault state is detected. |
| FR7 | Show the live state, recent history, and total running and idle time on a dashboard. |

## 3. Performance criteria

| ID | Criterion | Target (assumed) | How it will be verified |
|----|-----------|------------------|-------------------------|
| PC1 | State classification accuracy | at least 90% | Held-out test set not used in training |
| PC2 | Fault detection rate | at least 90% of faults detected | Test set with inserted faults (unbalanced load, overload) |
| PC3 | False alarm rate | below 5% of normal cycles | Test set of normal cycles |
| PC4 | Time from fault start to alert | under 10 seconds | Timestamp comparison in simulation |
| PC5 | Offline storage | at least 1 hour without data loss | Simulated network cut during a run |
| PC6 | Data integrity after reconnection | 100% of stored records delivered, in order | Compare device log with dashboard records |
| PC7 | Hardware cost per machine | under LKR 8,000 | Parts list with local prices |

## 4. Constraints

**Design constraints**
- Non-invasive sensing only: the device must never require cutting or modifying mains wiring.
- Mains supply in Sri Lanka is 230 V, 50 Hz; current sensing and signal processing must account for this.
- Computation and memory are limited to what a low-cost microcontroller (ESP32 class) provides.
- Must work without a network connection.
- Must be understandable to operators with no technical training.

**Project constraints**
- Total time: about 2 weeks.
- Prototype built with simulation only (Python for signals and ML, Wokwi for the microcontroller). No physical hardware in this phase.
- Models are trained on simulated data, so real-world accuracy must be re-checked in a later phase.

## 5. Out of scope (later phase)

- Physical hardware build and calibration
- Running the model on the microcontroller itself (TinyML)
- Network messaging protocol (MQTT) and cloud storage
- Testing in a real factory

## 6. Assumptions to validate

- The six machine states (Idle, Wash, Rinse, Drain, Spin, Fault) are enough to describe a washing machine cycle.
- Current and vibration together are enough to separate normal from faulty behaviour.
- The numeric targets in section 3 are realistic. Adjust them after the first simulation results and after talking to a factory owner.
