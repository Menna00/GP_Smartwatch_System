'''
import random
import time

class Smartwatch:
    def __init__(self):
        self.battery_level = 100  # Initialize battery level to 100%
        self.accelerometer_enabled = True
        self.gyroscope_enabled = True

    def get_battery_level(self):
        # Simulate getting the battery level (replace with actual code)
        self.battery_level = random.randint(0, 100)
        print("Battery level:", self.battery_level, "%")
        
    
    def enable_accelerometer(self, enable):
        # Simulate enabling/disabling the accelerometer (replace with actual code)
        if enable:
            print("Accelerometer enabled")
        else:
            print("Accelerometer disabled")
        self.accelerometer_enabled = enable
    
    def enable_gyroscope(self, enable):
        # Simulate enabling/disabling the gyroscope (replace with actual code)
        if enable:
            print("Gyroscope enabled")
        else:
            print("Gyroscope disabled")
        self.gyroscope_enabled = enable

    def run_sleep_mode(self):
        if self.battery_level < 10:
            print("Battery level is below 10%. Entering low power mode.")
            self.enable_accelerometer(False)
            self.enable_gyroscope(False)
            delay = 30
        elif self.battery_level < 30:
            print("Battery level is below 30%. Using sensors with a delay.")
            delay = 10
        else:
            print("Battery level is above 30%. Using sensors with full power.")
            delay = 0

        # Simulate sensor readings with delay
        while True:
            self.get_battery_level()
            time.sleep(delay)  # Adjust delay based on battery level
            

# Create an instance of the Smartwatch class
smartwatch = Smartwatch()

# Run the sleep mode functionality
smartwatch.run_sleep_mode()

'''

import random
import time

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
