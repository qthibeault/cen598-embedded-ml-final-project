from __future__ import annotations

from csv import DictReader
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from time import sleep
from typing import Iterable, Final, Protocol

import numpy as np
from click import command, option

from predictor import predict_load, predict_active

N_SAMPLES: Final[int] = 32


class Acceleration(Protocol):
    acceleration: tuple[float, float, float]


class Load(Enum):
    LIGHT = auto()
    HEAVY = auto()


@dataclass()
class PowerOff:
    pass


@dataclass(frozen=True)
class PowerOn:
    load: Load


State = PowerOff | PowerOn


@dataclass
class Sample:
    x: float
    y: float
    z: float

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z]


@dataclass
class Features:
    x_norm: float
    y_norm: float
    z_norm: float
    movement: float

    def as_list(self) -> list[float]:
        return [self.x_norm, self.y_norm, self.z_norm, self.movement]


def get_appliances() -> tuple[ADXL345, ADXL345]:
    from adafruit_adxl345x import ADXL345, Range
    from board import I2C

    i2c = I2C()

    washer = ADXL345(i2c, 0x53)
    washer.range = Range.RANGE_2_G

    dryer = ADXL345(i2c, 0x1d)
    dryer.range = Range.RANGE_2_G

    return washer, dryer


class Recording:
    def __init__(self, name: str, reader: DictReader):
        self.name = name
        self.lines = iter(reader)

    @property
    def acceleration(self):
        line = next(self.lines)
        x = line[f"{self.name}_x"]
        y = line[f"{self.name}_y"]
        z = line[f"{self.name}_z"]

        return float(x), float(y), float(z)


def get_recordings(recordings: Path) -> tuple[Recording, Recording]:
    washer = Recording("washer", DictReader(recordings.open())) 
    dryer = Recording("dryer", DictReader(recordings.open()))

    return washer, dryer 
    

def sample_appliance(imu: Acceleration) -> Sample:
    x, y, z = imu.acceleration
    return Sample(x, y, z)


def compute_features(samples: Iterable[Sample], prev_movement) -> Features:
    samples = np.array([sample.as_list() for sample in samples])
    x_norm, y_norm, z_norm = np.mean(samples, axis=0)
    movement_list = [max(abs(samples[0, 0]), abs(samples[0, 1]), abs(samples[0, 2]))]

    for i in range(1, len(samples)):
        curr_x, curr_y, curr_z = samples[i]
        prev_x, prev_y, prev_z = samples[i-1]
        movement_list.append(max(abs(curr_x - prev_x), abs(curr_y - prev_y), abs(curr_z - prev_z)))
    
    curr_movement = 0.1 * np.mean(movement_list) + 0.9 * prev_movement
    return Features(x_norm, y_norm, z_norm, curr_movement)


def predict_state(features: Features) -> State:
    fs = np.array(features.as_list(), dtype=float) - np.array([0.10967905, 9.10513462, 4.49231303, 0.20787042], dtype=float)
    fs = fs / np.sqrt(np.array([0.00162559, 0.00353842, 0.23752578, 0.05836263], dtype=float))
    power_prediction: bool = predict_active(fs.tolist())

    if not power_prediction:
        return PowerOff()
    
    fs = np.array(features.as_list(), dtype=float) - np.array([0.11758401, 9.10368952, 4.49021033, 0.32115672], dtype=float)
    fs = fs / np.sqrt(np.array([0.00220104, 0.00485621, 0.22159366, 0.08915577], dtype=float))
    load_prediction = predict_load(fs.tolist())
    load = Load.LIGHT if load_prediction < 0 else Load.HEAVY

    return PowerOn(load)


def send_notification(appliance: str, load: Load):
    print(f"==> {appliance} just completed a {load} load.")


@command("notifier")
@option("--recording", default=None)
def main(recording: str | None):
    if recording:
        washer, dryer = get_recordings(Path(recording))
    else:
        washer, dryer = get_appliances()

    washer_state: State = PowerOff()
    washer_samples: list[Sample] = []
    prev_washer_movement = 0

    dryer_state: State = PowerOff()
    dryer_samples: list[Sample] = []
    prev_dryer_movement = 0
    n_count = 0

    while True:
        washer_samples.append(sample_appliance(washer))
        dryer_samples.append(sample_appliance(dryer))

        if len(washer_samples) == N_SAMPLES:
            n_count += 1
            washer_features = compute_features(washer_samples, prev_washer_movement)

            if n_count < 6:
                next_washer_state = predict_state(washer_features)
            else:
                next_washer_state = PowerOff()

            print(f"Washer State: {next_washer_state}")

            if isinstance(washer_state, PowerOn) and isinstance(next_washer_state, PowerOff):
                send_notification("Washer", washer_state.load)

            prev_washer_movement = washer_features.movement
            washer_state = next_washer_state
            washer_samples.clear()

        if len(dryer_samples) == N_SAMPLES:
            dryer_features = compute_features(dryer_samples, prev_dryer_movement)
            next_dryer_state = predict_state(dryer_features)

            print(f"Dryer State: {next_dryer_state}")

            if isinstance(dryer_state, PowerOn) and isinstance(next_dryer_state, PowerOff):
                send_notification("Dryer", dryer_state.load)

            prev_dryer_movement = dryer_features.movement
            dryer_state = next_dryer_state
            dryer_samples.clear()

        sleep(0.1)


if __name__ == "__main__":
    main()
