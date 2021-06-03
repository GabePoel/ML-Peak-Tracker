"""
Peak Finder

A library for finding, fitting to, and tracking Lorentzian peaks from RUS or
other measurements. Tutorials and examples are available at the
`wiki <https://github.com/GabePoel/ML-Peak-Tracker/wiki>`_.

Documentation
-------------
There is some documentation available at th wiki. But, the most fine grained
information is available as docstrings provided within the code itself.

Both the wiki and docstrings assume that `utilities` is imported as `util` and
that `automatic` is imported as `auto`. Everything else is assumed to be
imported as a two letter initialism. For example, the `live_fitting`
subpackage would be imported as `lf`.

Frontend Subpackages
--------------------
utilities
    Core general purpose tools for loading/saving and manipulating RUS data.
live_fitting
    Interactive tools for fitting over imported RUS data.
automatic
    Quick and dirty machine learning models with little configurability.
sliding_window
    More detailed API for peak detecting machine learning models.
automatic_no_ml
    An alternative version of `automatic` for when TensorFlow isn't updated.

Backend Subpackages
-------------------
data_spy
    Tools for taking a quick little peak at confusing data.
track_peaks
    Automatic peak tracking given initial parameters. The original method.
models
    Definitions of existing machine learning models.
train_model
    Scripts to automatically train default machine learning models.
generate_lorentz
    Scripts to generate Lorentzians used to train machine learning models.
generate_data
    Scripts to generate complete data sets used for training.
efficient_data_generation
    Alternative tools to create large amounts of training data quickly.
background_removal
    Experimental techniques to remove background Lorentzians.
"""