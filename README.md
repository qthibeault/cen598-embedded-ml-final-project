# Smart Appliance Conversion

CEN598 Embedded Machine Learning Fall 2023

## About

This project contains several programs to record and monitor the status of a
washer and dryer. The purpose of this project is to detect when either
appliance is on or off and send notifications when washing or drying cycles
complete.

The program `monitor` is a python script that can be run using a python
interpreter to collect IMU data from both IMU sensors using the I2C bus. 

The `predictor` module is a native python module that wraps the Tensorflow-Lite
models used for detection. This is implemented as a module rather than using
the tensorflow python library because of the difficulties in compiling the
Tensorflow-Lite python wheel for the armv6 architecture of the Raspberry Pi
Zero W.

The `notifier` program also collects IMU data and records it into a buffer,
which is used to compute the 4 features that are provided to the `predictor`
module to determine the state of each appliance.

## Building

The `monitor` program can be run without any setup.

The `predictor` module is built using CMake. From the project's root
directory run the command `cmake -S predictor -B build` to configure
the project. Then, execute the command `cmake --build build` to build
the native module.

The `notifier` program depends on the `predictor` module. Once the module
has been compiled, copy the module archive, which should be named
`predictor-XX.so` to your python interpreter's site-packages directory to
"install" it.

## Running

To execute the `monitor` program run the command `python3 src/monitor.py` from
the command line.

The execute the `notifier` program run the command `python3 src/notifier.py`.
You can optionally provide a CSV file to the program that contains the IMU data
by using the recording flag like so:
`python3 src/notifier.py --recording recording.csv`.
