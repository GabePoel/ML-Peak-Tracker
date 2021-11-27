**<span style="color:purple">live&#95;fitting.point&#95;selection</span>_(data_files, params=None, fs=[])_**


An interactive session to help with finding redundant peaks between several
RUS sweeps at the same given temperature.


#### Parameters
* data_files : <b><i>list</i></b>  List of `data_files` at the same temperature.
* params : <b><i>arr, optional</i></b>  A 3D Lorentzian parameter array correpsonding to the parameters for
	the given data_files. Also accepts a list of 2D Lorentzian parameter
	arrays.
* fs : <b><i>list, optional</i></b>  Any already chosen frequencies to put markers out and include in the
	export.

#### Returns
<b><i>list</i></b>  The final list of chosen frequencies.