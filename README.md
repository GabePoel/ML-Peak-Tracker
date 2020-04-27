# ML Peak Tracker

## Message for Sylvia

Please run `scripts/count_hidden_layer_1024.py` for as long as you can. It passively generates and trains from data. The model it makes is `models/count_hidden_layer_1024`. The script will save the model periodically, so you can start and stop the script as need be. After a bunch of training, send me the `models/count_hidden_layer_1024` or push it here if that's easier. Thanks!

## What Works

Run the code in `train_test.ipynb` to see a more or less completely working Lorentzian peak generation model in action!

## Data Generation

There are several types of data generation that work now. The simplest types are called `single` and `simple` and they can be found in `efficient_data_generation.py`. Currently, all the models rely on these two types of data generation. But, it's also possible to generate an entire complete set of data as detailed below.

### Full Data Set Generation

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