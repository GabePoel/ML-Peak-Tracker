# ML Peak Tracker

Currently only data generation is done. To make a bunch of data run `generate_data.py` . This creates a `generated_data` directory and fills it with 1000 randomly generated data sets. Each one is composed of the following three files:

- `*_background.csv` - Has the parameters for the big background Lorentzian.
- `*_lorentz.csv` - Has the parameters for all of the generated Lorentzians.
- `*_data.csv` - Has the actual generated data. First column is frequency and second is displacement.

All parameters are stored as row with `amplitude, frequency, FWHM, phase` in that order.

All the actual Lorentz generation happens in `generate_lorentz.py`. Import this to generate data with custom parameters or whathaveyou. Noise and other generation parameters can be toggled in `generate_data()` here.