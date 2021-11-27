**<span style="color:purple">live&#95;fitting.live&#95;selection</span>_(data_file, params=None, vline=None)_**


Interactive peak selector at a single temperature.


#### Parameters
* data_file : <b><i>Data_File</i></b>  The `Data_File` that with all the data to be looked at.
* params : <b><i>arr, optional</i></b>  A 2D Lorentzian parameter array with any parameters to start with.
	This allows you to pause your work and continue later.
* vline : <b><i>float, optional</i></b>  A specific frequency to mark with a vertical green line. This is done
	for integration with the interactive color plot primarily.

#### Returns
<b><i>arr</i></b>  A 2D Lorentzian parameter array of all the peaks selected within the
	interactive session.