**<span style="color:purple">live&#95;fitting.mistake&#95;selection</span>_(data_files, parameters=None, path=None, method='lm')_**


Interactive Lorentzian editor over a range of temperatures.


#### Parameters
* data_files : <b><i>list</i></b>  List of data files to process and display.
* parameters : <b><i>arr, optional</i></b>  Any pre-determined Lorentzian parameters to continue working from. The
	parameters returned by `color_selection` work for this purpose. This
	exists so you can save your work during the fitting process and load it
	again later.
* path : <b><i>str, optional</i></b>  File path to use for autosaving.
* method : <b><i>{'trf', 'dogbox', 'lm'}, optional</i></b>  Fitting method to use by the :func: `<scipy.optimize.least_squares>`
	backend for fitting to Lorentzians. See the documentation on that for
	more details. You generally do not have to change this.

#### Returns
<b><i>arr</i></b>  The 3D parameter array of the Lorentzians fitted from the inputted data
	files.
	    - Axis 0 determines which sweep.
	    - Axis 1 determines which Lorentzian.
	    - Axis 2 is the parameters of the given Lorentzian.