import time
import board
import adafruit_adxl34x

WASHER_ADDR = 0x53
DRYER_ADDR = 0x1d


def main():
    i2c = board.I2C()
    washer = adafruit_adxl34x.ADXL345(i2c, WASHER_ADDR)
    washer.range = adafruit_adxl34x.Range.RANGE_2_G

    dryer = adafruit_adxl34x.ADXL345(i2c, DRYER_ADDR)
    dryer.range = adafruit_adxl34x.Range.RANGE_2_G

    print("time, washer_x, washer_y, washer_z, dryer_x, dryer_y, dryer_z")

    while True:
        washer_x, washer_y, washer_z = washer.acceleration
        dryer_x, dryer_y, dryer_z = dryer.acceleration
        t = time.time_ns()

        print(f"{t},{washer_x:.5f},{washer_y:.5f},{washer_z:.5f},{dryer_x:.5f},{dryer_y:.5f},{dryer_z:.5f}")
        time.sleep(0.1)


if __name__ == "__main__":
    main()
