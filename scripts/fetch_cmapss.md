# Fetching the C-MAPSS data

This repository **does not redistribute** the NASA C-MAPSS data set.

## 1. Download

Official archive (NASA Prognostics Center of Excellence):

```
https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
```

## 2. Extract

Unzip and locate the `CMAPSSData/` folder. The subsets used here are FD001 and
FD003; the other two are not required.

## 3. Place the files

Copy the six files into `data/raw/`:

```
data/raw/train_FD001.txt
data/raw/test_FD001.txt
data/raw/RUL_FD001.txt
data/raw/train_FD003.txt
data/raw/test_FD003.txt
data/raw/RUL_FD003.txt
```

## 4. Verify

```bash
python -c "from riskselrul.data import cmapss; tr, te, rul = cmapss.load('FD001'); print(tr.shape, te.shape, rul.shape)"
```

Expected: the training frame has 26 columns and one row per engine cycle; `rul`
has one entry per test engine (100 for FD001 and FD003).

## Alternative data root

Instead of copying files into `data/raw/`, point the loader elsewhere:

```bash
export RISKSELRUL_DATA_ROOT=/path/to/CMAPSSData     # bash
$env:RISKSELRUL_DATA_ROOT = "D:\path\to\CMAPSSData"  # PowerShell
```

or pass `--data-root` to `riskselrul train`.

## Layout reference

Each text file is whitespace separated with 26 columns:

```
unit, cycle, op1, op2, op3, s1, s2, ..., s21
```

Strictly speaking the README of the original archive documents 26 columns for
FD001/FD003; the loader reads all columns and keeps the 14 informative sensors.
