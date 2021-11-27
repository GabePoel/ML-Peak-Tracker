**<span style="color:purple">live&#95;fitting.color&#95;selection</span>_(data_files, x_res=1000, y_res=100, cmap='viridis', parameters=None, method='lm', no_out=False)_**


Interactive Lorentzian tracker over a range of temperatures. Displays the
peaks over all sweeps as a topographical color plot.


#### Parameters
* data_files : <b><i>list</i></b>  List of data files to process and display.
* x_res : <b><i>int, optional</i></b>  Initial resolution of the displayed x-axis. Default is `1000` but can
	be set higher or lower depending on what sort of graphics your computer
	can handle. The maximum value is the length of the frequency array in
	each data file. The x-axis is the frequency axis.
* y_res : <b><i>int, optional</i></b>  Initial resolution of the displayed y-axis. Default is `100` but can be
	set higher or lower depending on what sort of graphics your computer
	can handle. The maximum value is the length of the list of data files
	inputted into `data_files`. The y-axis just corresponds to the sweeps
	in the order that they're given in the inputted list of data files. For
	a linear change in temperature this means the values displayed are
	proportional to the temperature. But, they are not the temperature
	itself.
* cmap : <b><i>str, optional</i></b>  Initial colormap used for the displayed plot. Defaults to `viridis` but
	accepts any colormap that would be usable by `matplotlib`. This can be
	changed from a selection of other colormaps during the interactive
	plotting process.
* parameters : <b><i>arr, optional</i></b>  Any pre-determined Lorentzian parameters to continue working from. The
	parameters returned by `color_selection` work for this purpose. This
	exists so you can save your work during the fitting process and load it
	again later.
* method : <b><i>{'trf', 'dogbox', 'lm'}, optional</i></b>  Fitting method to use by the :func: `<scipy.optimize.least_squares>`
	backend for fitting to Lorentzians. See the documentation on that for
	more details. You generally do not have to change this.
* no_out : <b><i>bool, optional</i></b>  Makes the default zoom slightly scaled down along the y-axis. This
	is to make it easier to use the rectangle zoom tool over a region
	without accidentially cutting out any data you need to fit over.

#### Returns
<b><i>arr</i></b>  The 3D parameter array of the Lorentzians fitted from the inputted data
	files.
	    - Axis 0 determines which sweep.
	    - Axis 1 determines which Lorentzian.
	    - Axis 2 is the parameters of the given Lorentzian.