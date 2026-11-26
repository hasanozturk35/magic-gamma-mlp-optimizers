# Dataset: MAGIC Gamma Telescope

`magic04.csv` is the unmodified UCI file (no header, 19 020 rows, 11 columns).

- **Source:** R. Bock (2004), *MAGIC Gamma Telescope*, UCI Machine Learning
  Repository, <https://doi.org/10.24432/C52C8B>
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
  redistributed here with attribution.

The data are Monte Carlo simulated images of air showers recorded by a
ground-based Cherenkov telescope. The task is to separate primary gamma rays
(signal) from cosmic-ray hadrons (background).

| Column | Description |
|---|---|
| `fLength` | major axis of the shower ellipse [mm] |
| `fWidth` | minor axis of the ellipse [mm] |
| `fSize` | log10 of the sum of all pixel contents [#photons] |
| `fConc` | ratio of the sum of the two highest pixels to `fSize` |
| `fConc1` | ratio of the highest pixel to `fSize` |
| `fAsym` | distance from highest pixel to centre, projected on the major axis [mm] |
| `fM3Long` | 3rd root of the third moment along the major axis [mm] |
| `fM3Trans` | 3rd root of the third moment along the minor axis [mm] |
| `fAlpha` | angle of the major axis with the vector to the origin [deg] |
| `fDist` | distance from origin to the centre of the ellipse [mm] |
| `class` | `g` = gamma (12 332 rows), `h` = hadron (6 688 rows) |

`src/load_magic04.m` checks the column count, that every feature is a finite
number and that every label is `g` or `h`, and stops with an error otherwise.
