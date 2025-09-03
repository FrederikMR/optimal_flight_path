# Optimal Flight Path under Jet Streams

<img width="569" height="292" alt="earth_chart" src="https://github.com/user-attachments/assets/d55e1770-7c54-4673-8436-811c2927b37c" />

This repository is a "just for fun"-project that aims to compute the time-minimizing flight paths given jet streams.

Computing the time-minizming the path between two points on a sphere corresponds to finding the geodesics on a sphere, which have a well-known closed-form expression. However, if a flight is moving under the influence of a jet stream, simply computing geodesics will not the time-minimizing paths. This repository computes the time minimizing flight path between two points given a jet stream by computing geodesics under a time-dependent Finsler metric. We consider the Riemannian background metric induced by the WGS48 Earth Model corresponding to an ellipsoid, and a time-dependent force field acting on the earth model corresponding to the jet stream. Under certain regularity this can be seen as a time-dependent Finsler metric following the work of [1].

Geodesics under a time-dependent Finsler metric can be found using the algorithm prooposed in [2]. In the special case that the force field is zero, or is not time-dependent this corresponds to the algorithm in [3].

## Installation

The code can be setup using the environment env.yaml.

```
conda env create -f env.yaml
conda activate flight_model
```

## Data

The code automatically downloads the jet stream data using The Climate Data Store (CDS) Application Program Interface (API). See for license, details and how to setup up the API.

## Code Structure

The following shows the structure of the code. All general implementations of geometry and optimization algorithms can be found in the "geometry" folder for both the Riemannian and Finsler case.

    .
    ├── main.py                                     # Computes the time-minimizing curve given initial parameters between two points on earth for a static jet stream.
    ├── gui_interface.py                            # A graphical user interface for computing time-minizming curves between two points.
    ├── flight_model/manifold.py                    # Contains the WGS48 Earth Model
    ├── flight_model/conversion.py                  # Conversion from coordinates to floates
    ├── flight_model/metrics.py                     # The time-dependent Finsler metric for the Earth model
    ├── flight_model/interpolation.py               # Bilinear interpolation for the jet stream data in JAX
    ├── flight_model/geodesics.py                   # The algorithm from [2] used to compute the time-minimizing flight path.
    ├── flight_model/plotting.py                    # Contains generic plot functions
    ├── flight_model/download_jet_streams.py        # Downloads the jet stream data using The Climate Data Store (CDS) Application Program Interface (API). See for license and details.
    ├── flight_ui_background_image.png              # A background image taken from Microsoft Flight Simulater for the GUI
    └── README.md

## Future work

In the current version the optimal flight path can be found without a jet stream or with a static jet stream. Future work can include:
* Constructing a model that estimates the time evolution of the jet stream. This can be added directly to the code as is.
* Incorporating constraints into the model, e.g., places where it is not allowed to fly

## Disclaimer

The model is a simple representation of computing optimal flight paths given jet streams and furhter investigation of the method and other considerations are needed to ensure that this is indeed the time-minimizing flight path. The code is free to use. This code is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement.
In no event shall the author be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the code or the use or other dealings in the code.
Use of this code is entirely at your own risk.

## References

```
[1] L. Piro, E. Tang, R. Golestanian, Physical Review Research 3 (2), 023125
[2] S. Markvorsen, E. Pendás-Recondo, F. Rygaard, arXiv preprint arXiv:2508.07274, 2025
[3] F. Rygaard, S. Hauberg, https://arxiv.org/abs/2505.05961, 2025
```
