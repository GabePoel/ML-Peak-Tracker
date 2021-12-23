# Lorentzian Peak Finder (now with MaChInE lEaRnInG!)

A python library for finding, fitting to, and tracking Lorentzian peaks from RUS or other measurements. Usage and documentation can be found in the [wiki](https://github.com/GabePoel/ML-Peak-Tracker/wiki). Tutorials and demo data is available in [releases](https://github.com/GabePoel/ML-Peak-Tracker/releases/tag/demo).

Note that throughout the library, the definition of Lorentzians used is the following.

![eq1](https://raw.githubusercontent.com/GabePoel/ML-Peak-Tracker/master/images/equation_1.png)

Where the individual terms are as below.

![eq2](https://raw.githubusercontent.com/GabePoel/ML-Peak-Tracker/master/images/equation_2.png)

## Installing

Note that none of the installation methods download nor install the pre-made Lorentzian models. You need to get those separately. Please see [Lorentzian Models](https://github.com/GabePoel/Lorentzian-Models) for more information.

The **recommended** installation method is from [pypi](https://pypi.org/project/peak-finder/0.5/).

```sh
pip install peak-finder
```

But, you can also install directly from this git repository. These releases might not always be stable.

```sh
pip3 install git+https://github.com/GabePoel/ML-Peak-Tracker#egg=peak_finder --user
```
Or, if you only want to install the deltas, you can also clone this repository locally and then install using the included `local_install.sh` script. Navigate into the cloned repository and then run the following command.

```sh
sh ./peak_finder/local_install.sh
```

