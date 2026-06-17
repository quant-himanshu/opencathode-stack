# Dataset Sources — OpenCATHODE Project

## VED (Vehicle Energy Dataset)
- Source: University of Michigan, Ann Arbor (Oh, Leblanc, Peng)
- Vehicles: 383 real EV/PHEV/ICE drivers, Michigan, USA
- Columns: VehId, Trip, Timestamp, GPS Lat/Long, Vehicle Speed,
  HV Battery Current/SOC/Voltage, Engine RPM, Fuel data
- Format: Weekly CSV files (data/ved/VED_*.csv)
- Note: contains NaN in battery columns for ICE-mode segments (real-world artifact)

## BMW i3 Dataset
- Source: Public BMW i3 instrumented test-vehicle dataset
- Vehicles: 70 trips (TripA01-32.csv, TripB01-38.csv)
- Columns: Time, Velocity, Battery Voltage/Current/Temperature, SoC %,
  Ambient Temperature, Heater/AirCon Power
- Includes original MATLAB read-in script (readin.m)

## Deng Dataset (BAIC EU500)
- Source: Deng et al., published Chinese commercial EV dataset
- Vehicles: 20 BAIC EU500 vehicles (#1.csv to #20.csv)
- Columns: record_time, SOC, pack_voltage, charge_current,
  max/min_cell_voltage, max/min_temperature, available_energy/capacity
- Granularity: cell-level (max/min cell voltage logged separately)
