# Data card

- Dataset: NASA PCoE Turbofan Engine Degradation Simulation (C-MAPSS).
- Official catalog: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- Previously used NASA-hosted archive: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
- Legacy download listed by NASA: https://data.nasa.gov/docs/legacy/CMAPSSData.zip
- This package supports FD001 and FD003 only. Both are single-operating-condition subsets; FD003 has two degradation modes.

Each trajectory file has 26 whitespace-separated columns: engine ID, cycle, three operating settings, and 21 sensor channels. Training trajectories run to failure; test trajectories stop earlier, with one endpoint RUL label per engine in the separate RUL file. Each supported subset has 100 training engines and 100 different test engines.

Raw observations are simulator output. Faults used by this package are synthetic perturbations applied to those observations. Neither the original data nor the injected faults are measurements from real damaged devices.

Download the archive through NASA, inspect its applicable use terms, then use `extract-data`. That command only accepts the six named FD001/FD003 members, rejects unsafe ZIP paths/symlinks/duplicates and refuses overwrites. It never automatically contacts NASA. The package distributes reference SHA-256 values rather than original data. Whitespace changes can preserve numerical values but change byte hashes; strict hash checking is optional and its result is explicit.

The 14 used channels are s2,s3,s4,s7,s8,s9,s11,s12,s13,s14,s15,s17,s20,s21. The numerical profile uses 30-cycle windows and 56 statistical features. Short diagnostic/test windows are padded by repeating their first observation. Every learned preprocessing object is fit within its designated training-engine boundary.

## NASA download packaging

The S3 download above contains an outer ZIP with one `CMAPSSData.zip` member. For the manual `extract-data` command, supply that inner archive. The executable `scripts/cold_start.py` handles this known wrapper, records both archive hashes, and then invokes the same safe data extractor. It never extracts arbitrary outer member paths.
