# ML Peak Tracker

## What Works

The program is currently broken down into three steps:

1. Figure out *how many* clusters of Lorentzians there are. This is how many multi-Lorentz fits will be needed.
2. Figure out *where* those Lorentzian clusters are.
3. Figure out how many Lorentzians are within a given cluster.

After all this is done, fitting can easily happen. Versions of the first and third step are done and semi-functional. The current models can consistently get above 90% accuracy with figuring out how many clusters there are. But, they have yet to crack 70% for figuring out how many are clumped together at any one point. No attempt has yet been made at figuring out where the clusters are.

## Data Generation

To make a bunch of data run `generate_data.py` . This creates a `generated_data` directory and fills it with 1000 randomly generated data sets. Each one is composed of the following three files:

- `*_background.csv` - Has the parameters for the big background Lorentzian.
- `*_lorentz.csv` - Has the parameters for all of the generated Lorentzians.
- `*_data.csv` - Has the actual generated data. First column is frequency and second is displacement.

All parameters are stored as row with `amplitude, frequency, FWHM, phase` in that order.

All the actual Lorentz generation happens in `generate_lorentz.py`. Import this to generate data with custom parameters or whathaveyou. Noise and other generation parameters can be toggled in `generate_data()` here.

As I've set it up here, the in and out of phase components of the Lorentzian are defined as follows:

$$A_\text{in}=\frac{A_0}{2\Gamma}\frac{f-f_0}{\left(\frac{f-f_0}{\Gamma}\right)^2+\frac{1}{4}},\quad A_{\text{out}}=\frac{A_0}{4}\frac{1}{\left(\frac{f-f_0}{\Gamma}\right)^2+\frac{1}{4}}$$ 

Where the full version is then the following:

$$A(f,\theta)=A_\text{in}\cos(\theta)+A_\text{out}\sin(\theta)$$

(The LaTeX probably won't appear properly on GitHub. But it should render properly in any other markdown viewer.)