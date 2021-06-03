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
automatic
    Quick and dirty machine learning models with little configurability.
automatic_no_ml
    An alternative version of `automatic` for when TensorFlow isn't updated.
live_fitting
    Interactive tools for fitting over imported RUS data.
sliding_window
    More detailed API for peak detecting machine learning models.
utilities
    Core general purpose tools for loading/saving and manipulating RUS data.

Backend Subpackages
-------------------
background_removal
    Experimental techniques to remove background Lorentzians.
classify_data
    Handling of data pre-processing.
data_spy
    Tools for taking a quick little peak at confusing data.
efficient_data_generation
    Alternative tools to create large amounts of training data quickly.
fit_lorentz
    Backend for custom fitting to different types of Lorentzians.
generate_data
    Scripts to generate complete data sets used for training.
generate_lorentz
    Scripts to generate Lorentzians used to train machine learning models.
models
    Definitions of existing machine learning models.
track_peaks
    Automatic peak tracking given initial parameters. The original method.
train_model
    Scripts to automatically train default machine learning models.
"""