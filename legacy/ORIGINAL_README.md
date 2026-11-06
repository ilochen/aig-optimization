# DRL_Final
Final project for Deep Reinforcement Learning 2025@NTU

## install
pip install -r requirements.txt -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
sudo apt-get install doxygen
cd to abc_py
把pybind11 拉到abc_py下
sudo apt-get update
sudo apt-get install libboost-all-dev
mkdir build 
cd build 
cmake .. with path of your abc
example cmake .. -DABC_DIR=/home/dereklin1205/University/DRL/DRL_Final/abc_rl/abc_rl_dependency/abc


# HOW TO USE RL ENVIRONMENT

## 📁 Project Structure

    ├── abc/                         # Cloned ABC logic synthesis tool
    ├── abc.history                  # ABC command history (optional)
    ├── helper_functions.py          # Internal utility functions (⚠️ Do not use directly)
    ├── requirements.txt             # Python dependencies
    ├── rl_environment.py            # Main environment entry point (✅ use this)
    ├── runtime_results/             # Output directory
    │   └── current.aig              # Example AIG file
    └── testbenches/                 # Benchmark circuits
        ├── 2025_IWLS_Contest_Benchmarks_020425/
        └── EPFL/


### 1. Clone the ABC Library

You must clone and build the ABC logic synthesis tool in the `abc/` folder:

    git clone https://github.com/berkeley-abc/abc.git abc
    cd abc
    make
    cd ..

### 2. Install Python Dependencies

Use the provided `requirements.txt` to install the required packages:

    pip install -r requirements.txt

### 3. Run the Environment

Use `rl_environment.py` as the main entry point. It includes a simple demo in the `main()` function:

    python rl_environment.py

## ⚠️ Notes

- Do not modify or run `helper_functions.py` directly. This file contains internal utilities and currently has unresolved issues.
- Only use the `rl_environment.py` interface for interaction and extension.




