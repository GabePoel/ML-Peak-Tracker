# Things That Need Doing


## Big Changes
- [ ] Migrate models to separate library/repository.
- [ ] Allow for automated model installation.
- [ ] Host latest version of `peak_finder` on Pypi again.
## Docstrings

The following files still need to be written with scikit docstring format. They generally have something written for most if not all functions/classes/methods. But, the documentation is still incomplete.
 - [x] `automatic.py`
 - [ ] `classify_data.py`
 - [ ] `data_spy.py`
 - [x] `efficient_data_generation.py`
 - [ ] `fit_lorentz.py`
 - [ ] `generate_data.py`
 - [ ] `generate_lorentz.py`
 - [x] `live_fitting.py`
    - [x] Common functions
    - [x] `_Selection`
    - [x] `_Live_Instance`
    - [x] `_Color_Selector`
    - [x] `_Point_Selector`
    - [x] `_Mistake_Selector`
    - [x] `live_selection`
    - [x] `color_selection`
    - [x] `point_selection`
    - [x] `mistake_selection`
 - [ ] `models.py`
 - [ ] `sliding_window.py`
 - [x] `train_model.py`
 - [x] `utilities.py`
 - [ ] `scripts/train_simple_class.py`
 - [ ] `scripts/train_tight_wiggle.py`
 - [ ] `scripts/train_wide_wiggle.py`

 ## Bug Fixes

 The following scripts don't respond well to running in the main PATH. They instead need to be run from their default directory. Also, their connection to the model they train from needs to be updated.
 - [ ] `scripts/train_simple_count.py`
 - [ ] `scripts/train_tight_wiggle.py`
 - [ ] `scripts/train_wide_wiggle.py`

 And these ones need to be fixed or removed.
 - [ ] `scripts/multi_class_train.py`
 - [ ] `scripts/train_simple_count.py`

 ## Other Stuff
 - [ ] Write guide on training models.
 - [x] Rewrite peak _tracker_.
 - [x] Update this todo list because it's so out of date lol 🙁

## Tests.
 - [ ] Write tests for _everything_. 🙁
 - [ ] Add tooltips to live sessions.
