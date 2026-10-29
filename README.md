# Offline-First Retrofit Monitor

Monitoring for old factory machines that have no data output, and that keeps working when the network drops.

## The problem (short version)

Most legacy machines in small and medium factories cannot report whether they are running, idle, or faulty, and the monitoring systems that exist often stop when the network fails. Owners cannot see where production time is lost. Full text: [docs/problem-statement.md](docs/problem-statement.md)

## Demo

_Video / GIF coming soon._

## Project structure

```
docs/          problem statement, design specification
simulation/    simulated machine signals + ML
firmware/      ESP32 code (Wokwi)
dashboard/     live machine-state dashboard
```

## Status

- [x] Problem statement
- [ ] Design specification
- [ ] Signal simulation
- [ ] Feature extraction + ML
- [ ] ESP32 firmware (offline logging)
- [ ] Dashboard
- [ ] Demo video

## Author

DimWick, Electrical Engineering student
