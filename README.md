## Prerequisites

The following Python libraries are used:

* numpy (linear algebra)
* scikit-image (loading images, gaussian noise)
* scipy (sparse algebra, DFT)
* pyproximal (TV proximal operator)
* opencv-python (cascade classifiers)
* cvxpy (l1-minimization)

Installation with e.g.
```
pip install numpy scikit-image scipy pyproximal opencv-python cvxpy
```

## Compressed Sensing, 12.1

Files are available in the `sheet12_ex1` subdirectory. The report can be found
in `reports/CS_Sheet12.pdf`.

### Program layout

| Program         | Description                          |
|:----------------|:-------------------------------------|
| `main.py`       | Runs all experiments                 |
| `algorithms.py` | Implementation of greedy algorithms  |
| `plot.py`       | Generate plots for report            |
| `sensor.py`     | Generate random and Fourier matrices |

`main.py` uses the default values `--n=2**7`, `--m=2**6`, `--tol=1e-6`,
`--n_trials=100` and runs all algorithms. 

A subset of algorithms can be specified with the `--problem` argument, using
comma-separated values and a problem index as defined in the table below.

| Index | Algorithm       |
|-------|-----------------|
| 1     | `basis_pursuit` |
| 2     | `OMP`           |
| 3     | `MP`            |
| 4     | `IHT`           |
| 5     | `CoSaMP`        |
| 6     | `BT`            |
| 7     | `HTP`           |
| 8     | `SP`            |

### Data

All trials are stored in `.json` files, following the format:

```
<matrix>_<rows>_<columns>_<num_trials>_fre_<method>.json
```

Each JSON file contains the average recovery error and CPU time per iteration, indexed by sparsity:

```
"error": [
  2.2753024811660305e-12,  # s = 1
  2.933767862840058e-12,   # s = 2
  ...
]
```

```
"cputime": [
  0.28070760991000004,  # s = 1
  0.28951821852000015,  # s = 2
]
```

## Compressed Sensing, 12.2

Files are available in the `sheet12_ex2` subdirectory. The report can be found
in `reports/CS_Sheet12.pdf`.

### Program layout

| Program               | Description                                |
|-----------------------|--------------------------------------------|
| `main.py`             | Face recognition with varying parameters   |
| `dataset.py`          | Sampling, cropping, and resizing of images |
| `face_recognition.py` | Dictionary and soft thresholding           |
| `fista.py`            | FISTA schemes (1D)                         |

`main.py` uses the default values `--tol=1e-4`, `--sigma=1`, `--max-iter=10000`,
`--method=fista_mod` and `--robust=False`. The number of images to which the
face recognition problem is applied has to be specified, e.g. `main.py 5`. 

When `--robust` is specified, images have Gaussian noise added (with default
mean 0 and variance 0.02, controlled by the `--robust-mean` and `--robust-var`
arguments) and the robust face recognition problem is applied. The dictionaries
A and B = [A I] are written out to MatrixMarket files, for the face recognition
and robust face recognition problem, respectively.

### Data

The image data is contained in the following directories:

| Directory           | Description                                     |
|---------------------|-------------------------------------------------|
| `data_all`          | LFW dataset                                     |
| `data_cropped`      | LFW dataset after face recognition and resizing |
| `data_training`     | Samples from cropped dataset                    |
| `data_verification` | Samples not contained in `data_training`        |

For each image, the following files are generated after the algorithm terminates:

| File                                              | Description                   |
|---------------------------------------------------|-------------------------------|
| `img<num>_input.jpg`                              | Input image                   |
| `img<num>_<method>_<sigma>_<tol>_recovered.jpg`   | Recovered image Bx*           |
| `img<num>_<method>_<sigma>_<tol>_recognized0.jpg` | Column of B indexed by x*_[0] |
| `img<num>_<method>_<sigma>_<tol>_recognized1.jpg` | Column of B indexed by x*_[1] |
| `img<num>_<method>_<sigma>_<tol>_recognized2.jpg` | Column of B indexed by x*_[2] |
| `img<num>_<method>_<sigma>_<tol>.json`            | Result data                   |

With `--robust`, a `_robust` suffix is added to `<method>` images and `_noisy`
to input images. The result data contains the solution and objective difference
norms, the solution x*, the number of iterations, and the indices of x* sorted
in descending magnitude.

## Convex Optimization, Image Denoising

Files are available in the `denoising` directory. The report can be found
in `reports/denoising.pdf`

### Program layout

| Program               | Description                                |
|-----------------------|--------------------------------------------|
| `main.py`             | FISTA TV denoising with varying parameters |
| `fista.py`            | FISTA schemes (2D)                         |
| `fista_restart.py`(*) | FISTA restarting schemes (2D)              |
| `plot.py`             | Plots used for report                      |

`main.py` uses the default parameters `--tol=1e-6`, `--sigma=0.06`,
`--max-iter=5000` for the iteration, and `--noise-mode=gaussian`,
`--noise-var=0.01`, `--noise-mean=0` for the noisy images.

(*) Restarting schemes were not included in the final comparison.

### Data

The image data is contained in the `stylegan2` directory.

For each image, the following files are generated after the algorithm terminates:

| File                                                  | Description                  |
|-------------------------------------------------------|------------------------------|
| `image<num>.png`                                      | Input image                  |
| `image<num>_<noise_mode>.png`                         | Input image with added noise |
| `image<num>_<noise_mode>_<method>_<sigma>_<tol>.png`  | Denoised image               |
| `image<num>_<noise_mode>_<method>_<sigma>_<tol>.json` | Denoised image (result data) |

The result data contains the solution and objective difference norms, the
solution x*, the number of iterations, and the indices of x* sorted in
descending magnitude.
