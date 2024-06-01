import random
import time

# with replacing the logic in these functions with the embedded code for raspberry pi pico


class Smartwatch:
    def __init__(self):
        self.battery_level = 100  # Initialize battery level to 100%
        self.accelerometer_enabled = True
        self.gyroscope_enabled = True

    def get_battery_level(self):
        # Simulate getting the battery level (replace with actual code)
        self.battery_level = random.randint(0, 100)
        return self.battery_level
    
    def enable_accelerometer(self, enable):
        # Simulate enabling/disabling the accelerometer (replace with actual code)
        self.accelerometer_enabled = enable
    
    def enable_gyroscope(self, enable):
        # Simulate enabling/disabling the gyroscope (replace with actual code)
        self.gyroscope_enabled = enable

    def run_sleep_mode(self):
        while True:
            battery_level = self.get_battery_level()
            mode = self.determine_mode(battery_level)
            print("Battery level:", battery_level, "% - Mode:", mode)
            time.sleep(5)  # Adjust delay based on update frequency

    def determine_mode(self, battery_level):
        if battery_level < 10:
            self.enable_accelerometer(False)
            self.enable_gyroscope(False)
            mode = "Low Power Mode, and the accelerometer and gyroscope are off."
        elif battery_level < 30:
            self.enable_accelerometer(True)
            self.enable_gyroscope(True)
            mode = "Delayed Mode"
        else:
            self.enable_accelerometer(True)
            self.enable_gyroscope(True)
            mode = "Normal Mode"
        return mode

# Create an instance of the Smartwatch class
smartwatch = Smartwatch()

# Run the sleep mode functionality
smartwatch.run_sleep_mode()
