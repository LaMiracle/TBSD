# Image Texture Analysis Toolkit

A Python toolkit for image texture analysis, including background removal, texture basis creation, periodicity checking, anomaly detection, and visualization.

## Project Structure

- `benchmarks/`: Benchmark scripts for performance evaluation (FFT, RPCA, SSD).
- `numerical_exp/`: Numerical experiments including anomaly confirmation and curve closing.
- `test_data/`: Test data directory with subfolders like `hole/`.
- `testPhase/`: Testing phase scripts, including decomposition.
- `train_data/`: Training data with `real-world/` and `simulation/` subfolders.
- `trainPhase/`: Training phase scripts including background removal (`bgRemove.py`), main training (`main.py`), periodicity check (`perioCheck.py`), and texture basis creation (`texBasisCreate.py`).
- `visualization/`: Visualization utilities for plotting PNG images (`pngPlot.py`).

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd image-texture-analysis-toolkit
   ```

2. Install dependencies:
   ```
   pip install numpy scipy matplotlib opencv-python
   ```
   (Adjust based on actual requirements; check individual scripts for imports.)

## Usage

- For training: Run `python trainPhase/main.py`
- For testing: Run scripts in `testPhase/`
- For experiments: Execute scripts in `numerical_exp/`
- For benchmarks: Run scripts in `benchmarks/`
- For visualization: Use `visualization/pngPlot.py`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact [song-j23@mails.tsinghua.edu.cn].