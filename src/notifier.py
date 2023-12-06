from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from time import sleep
from typing import Iterable, Final

from board import I2C
from adafruit_adxl345 import ADXL345, Range

from predictor import predict_load, predict_power
import numpy as np

N_SAMPLES: Final[int] = 32


class Load(Enum):
    LIGHT = auto()
    HEAVY = auto()


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


def sample_appliance(imu: ADXL345) -> Sample:
    x, y, z = imu.acceleration
    return Sample(x, y, z)


def compute_features(samples: Iterable[Sample], prev_movement) -> Features;
    samples = np.array([sample.as_list() for sample in samples])
    x_norm, y_norm, z_norm = np.mean(samples, axis=0)
    movement_list = [max(abs(samples[0, 0]), abs(samples[0, 1]), abs(samples[0, 2]))]
    for i in range(1, len(samples)):
        curr_x, curr_y, curr_z = samples[i]
        prev_x, prev_y, prev_z = samples[i-1]
        movement_list.append(max(abs(curr_x - prev_x), abs(curr_y - prev_y), abs(curr_z - prev_z)))
    
    curr_movement = 0.1 * np.mean(movement_list) + 0.9 * prev_movement
    


def predict_state(features: Features) -> State:
    fs = features.as_list()
    power_prediction: float = predict_power(fs)

    if power_prediction < 0:
        return PowerOff()
    
    load_prediction = predict_load(fs)
    load = Load.LIGHT if load_prediction < 0 else Load.HEAVY

    return PowerOn(load)


def send_notification(appliance: str, load: Load):
    print(f"==> {appliance} just completed a {load} load.")


def main():
    i2c = I2C()
    washer = ADXL345(i2c, 0x53)
    washer.range = Range.RANGE_2_G
    washer_state: State = PowerOff()
    washer_samples: list[Sample] = []
    prev_washer_movement = 0

    dryer = ADXL345(i2c, 0x1d)
    dryer.range = Range.RANGE_2_G
    dryer_state: State = PowerOff()
    dryer_samples: list[Sample] = []
    prev_dryer_movement = 0

    while True:
        washer_samples.append(sample_appliance(washer))
        dryer_samples.append(sample_appliance(dryer))

        if len(washer_samples) == N_SAMPLES:
            washer_features = compute_features(washer_samples, prev_washer_movement)
            next_washer_state = predict_state(washer_features)

            if isinstance(washer_state, PowerOn) and isinstance(next_washer_state, PowerOff):
                send_notification("Washer", washer_state.load)

            prev_washer_movement = washer_features.movement
            washer_state = next_washer_state
            washer_samples.clear()

        if len(dryer_samples) == N_SAMPLES:
            dryer_features = compute_features(dryer_samples, prev_dryer_movement)
            next_dryer_state = predict_state(dryer_features)

            if isinstance(dryer_state, PowerOn) and isinstance(next_dryer_state, PowerOff):
                send_notification("Dryer", dryer_state.load)

            prev_dryer_movement = dryer_features.movement
            dryer_state = next_dryer_state
            dryer_samples.clear()

        sleep(0.1)


if __name__ == "__main__":
    pass
