# Lorentzian Peak Finder (now with MaChInE lEaRnInG!)

A python library for finding, fitting to, and tracking Lorentzian peaks from RUS or other measurements. Usage and documentation can be found in the [wiki](https://github.com/GabePoel/ML-Peak-Tracker/wiki). Tutorials and demo data is available in [releases](https://github.com/GabePoel/ML-Peak-Tracker/releases/tag/demo).

Note that throughout the library, the definition of Lorentzians used is the following.

![eq1](https://raw.githubusercontent.com/GabePoel/ML-Peak-Tracker/master/images/equation_1.png)

Where the individual terms are as below.

![eq2](https://raw.githubusercontent.com/GabePoel/ML-Peak-Tracker/master/images/equation_2.png)

## Installing

Just run the following to install directly from this repository.
```sh
pip3 install git+https://github.com/GabePoel/ML-Peak-Tracker#egg=peak_finder --user
```
This does not download and install the models. For those, please see [Lorentzian Models](https://github.com/GabePoel/Lorentzian-Models).