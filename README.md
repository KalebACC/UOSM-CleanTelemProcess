# UOSM Clean Telemetry Process

## Merge d_87 and d_88 with smooth ticks

This repo now includes `merge_d87_d88.py`, which concatenates `d_87.csv` then `d_88.csv` and remaps `d_88` tick intervals to follow the same cadence pattern as `d_87`.

### Quick run

```bat
python merge_d87_d88.py
```

Output file:

- `Data/d_87_88_merged.csv`

### Optional custom paths

```bat
python merge_d87_d88.py --run87 Data/d_87.csv --run88 Data/d_88.csv --out Data/d_87_88_merged.csv
```
