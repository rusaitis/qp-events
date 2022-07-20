# Quasiperiodic Event Analysis

## Introduction

This package contains programs to process magnetic field data, transform between coordinate systems, perform Fourier and wavelet analysis, and compute spacecraft dwell times in latitude-LT-radial-distance bins.

## Installation

Download the archived folder or clone the repository.

### Requirements
* Python 3 (>3.9)
* Fortran compiler (e.g. GFortran) if you plan to use the magnetic field model
* NumPy (>=1.14)
* SciPy (>=0.19)
* Pandas (>=0.24)
* scipy.interpolate (>=1.0)
* scipy.signal (>=0.13)
* scipy.fftpack (>=0.13)
* scipy.weave (>=0.13)
* scipy.linalg (>=0.13)
* scipy.optimize (>=0.13)
* Matplotlib (>=2.0)

### External packages dependencies
* [`Kmag - Khurana Field Model`](https://zenodo.org/record/4080294)
Khurana, K. K., & Rusaitis, L. (n.d.). Khurana Jupiter Current Sheet Structure Model 2022. https://doi.org/10.17189/0DWH-5K25
    * To install the model as a local python package, run `setup.py` in the main directory.
    * To install the model as a system package, run `pip install -e .` in the main directory.

Tested using GNU Fortran (Homebrew GCC 11.3.0) and Python 3.9.10 running on
Mac OS 12 and ZSH shell.

## Usage
Pass -h or --help argument in the terminal to get a help with the command-line
driven interface to the program.

For example, run the following in Bash:
```bash
python3 mission_trace.py --help
```

### Optional arguments:
  **-h, --help**
  show this help message and exit
  
  **--step (default = 0.1)**
  Field line tracing step size (in planetary radius)

  **--dipole**
  Use a dipole magnetic field

  **--spacecraftPosition**
  Investigate the spacecraft position

  **--fast (default = False)**
  Enable a faster (less accurate) tracing and resonance calculation

  **--f (default = 0.09796)**
  Polar radius reduction in terms of the equatorial planetary radius

  **--maxR (default = 100)**
  Maximum distance for a field line point

  **--output**
  Custom output folder

  **--method (default = RK4)**
  Tracing method: `euler` or `RK4`

  **--year (default = 2005)**
  Year to analyze the data
  
  **--calc**
  Compute the Alfven waves
  
  **--mov**
  Generate a movie
  
  **--save**
  Save the figures instead of showing
## Program Directory Tree

The program save figures in the `output` directory. For example, an eigenfrequency plot will be created in it as "figure_eigfreq.pdf".

### PROGRAM TREE 
* mission_trace.py
  * Reads Cassini spacecract position and analyzes orbital bias
* mission_trace_reader.py
  * Processing data generated by `mission_trace.py`
* mag_fft_sweeper.py
  * Reads Cassini magnetometer data (downloaded from PDS) and computes FFT and wavelet transforms
* data_sweeper.py
  * Processes data generated by `mag_fft_sweeper.py`
* boundary_crossings.py
  * Generates a timeseries of Cassini position with respect to the magnetopshere/bow shock crossings by Jackman et al. (2019)

* output
  * event_list.txt - list of QP events
* README.md
* cassinilib
  * Core.py - general functions
  * DataPaths.py - data location and structure
  * DatetimeFunctions.py - for processing datetimes
  * Event.py - class for event properties
  * FileCheck.py - check if file exists
  * io.py - input/output routines
  * NewSignal.py - class for storing spacecraft (MAG) data
  * Plot.py - general plots
  * PlotFFT.py - FFT plots
  * PlotTimeseries.py - timeseries plots
  * SelectData.py - read and select downloaded data from PDS
  * Tic.py - simple timer
  * ToMageticCoords.py - converting data to MFA coords
  * Vector.py - class for a vector
  * Wave.py - class for storing wave properties
* wavesolver
  * configurations.py - "Basic field line, PDE solver, and model parameters"
  * fieldline.py "Field line class and associated functions to store field line data"
  * helperFunctions.py - "Miscellaneous general-use functions"
  * io.py - "Input / Output functions"
  * linalg.py - "Linear Algebra"
  * model.py - "Plasma density models, PDE functions, and dipole field model"
  * plot.py - "Plotting routines"
  * shoot.py - "Shooting Method PDE solver"
  * sim.py - "Simulation class for storing all relevant parameters for calculations"

## Examples

To read and plot Cassini position within 50 RS of Saturn:
```bash
python mission_trace.py --calc --spacecraftPosition --maxR 50 --plot
```

To plot Cassini event time to dwell time ratio for a specific local time segment with a minimum dwell time of 1 day:
```bash
python mission_trace_reader.py --LT 15 21 --plottype ratio --min_dwell 1
```

To compute the FFT and wavelet spectra for the whole mission (in Mean Field Aligned coordinates):
```bash
python mag_fft_sweeper.py --coord KSM --source mission --save --Bcoord
```

To plot the spectra for all field components and only for times Cassini was in the Magnetosphere:
```bash
python data_sweeper.py --source mission --coord KSM --plot_all_type components --location MS --save
```

To generate a timeseries of Cassini position with respect to the magnetosphere/bow shock crossings:
```bash
python boundary_crossings.py
```

## References:
[`Kmag - Khurana Field Model`](https://zenodo.org/record/4080294)
Khurana, K. K., & Rusaitis, L. (n.d.). Khurana Jupiter Current Sheet Structure Model 2022. https://doi.org/10.17189/0DWH-5K25

Khurana, K. K., Arridge, C. S., Schwarzl, H., & Dougherty, M. K. (2006). A model
of Saturn’s magnetospheric field based on latest Cassini observations. In AGU
Spring Meeting Abstracts (Vol. 2007, pp. P44A-01).

Jackman, C. M., Thomsen, M. F., & Dougherty, M. K. (2019). Survey of Saturn’s magnetopause and bow shock positions over the entire Cassini mission: boundary statistical properties and exploration of associated upstream conditions. Journal of Geophysical Research: Space Physics, 124(11), 8865–8883. https://doi.org/10.1029/2019JA026628

## License
[MIT](https://choosealicense.com/licenses/mit/)
Liutauras Rusaitis
