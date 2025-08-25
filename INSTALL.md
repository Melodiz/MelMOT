# Installation Guide for MelMOT

This guide will help you install and set up MelMOT on your system.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

## Installation Options

### Option 1: Install from Source (Recommended for Development)

1. **Clone the repository**
   ```bash
   git clone https://github.com/Melodiz/MelMOT.git
   cd MelMOT
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e .
   ```

### Option 2: Install Dependencies Only

1. **Clone the repository**
   ```bash
   git clone https://github.com/Melodiz/MelMOT.git
   cd MelMOT
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Verification

After installation, verify that everything works:

1. **Run tests**
   ```bash
   python run_tests.py
   ```

2. **Run example**
   ```bash
   python examples/simple_tracking.py
   ```

3. **Test CLI**
   ```bash
   python -m melmot.cli --help
   ```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're in the virtual environment
2. **CUDA not found**: Install PyTorch with CUDA support if needed
3. **YOLO model download fails**: Check internet connection and try again

### Getting Help

- Check the README.md for detailed usage instructions
- Review the example configuration files in `melmot/config/`
- Run tests to identify specific issues

## Next Steps

1. **Configure your setup**: Copy and modify `melmot/config/example_setup.yaml`
2. **Prepare your data**: Organize your video files and camera configurations
3. **Run tracking**: Use the CLI or Python API to start tracking

## Research Usage

For researchers and students:

1. **Run the examples**
   ```bash
   python examples/simple_tracking.py
   ```

2. **Modify configurations**
   Edit the YAML files in `melmot/config/` to experiment with different parameters

3. **Extend the algorithms**
   The modular structure makes it easy to implement new tracking or Re-ID approaches
