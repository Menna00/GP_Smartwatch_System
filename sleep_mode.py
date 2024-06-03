import random
import time
import statistics

class Smartwatch:
    def __init__(self):
        self.battery_level = 100  # Initialize battery level to 100%
        self.accelerometer_enabled = True
        self.gyroscope_enabled = True
        self.accelerometer_readings = []
        self.gyroscope_readings = []

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

    def get_sensor_reading(self):
        # Simulate getting sensor readings (replace with actual code)
        return (random.uniform(-300, 300), random.uniform(-300, 300), random.uniform(-300, 300))  # Simulating readings between -300 and 300


    def update_sensor_readings(self):
        # Update accelerometer and gyroscope readings
        accel_reading = self.get_sensor_reading()
        gyro_reading = self.get_sensor_reading()
        
        self.accelerometer_readings.append(accel_reading)
        self.gyroscope_readings.append(gyro_reading)
        
        # Keep only the last 10 readings
        if len(self.accelerometer_readings) > 10:
            self.accelerometer_readings.pop(0)
        if len(self.gyroscope_readings) > 10:
            self.gyroscope_readings.pop(0)

    def check_idle_mode(self):
        # Check the variance of the last 10 readings
        if len(self.accelerometer_readings) == 10 and len(self.gyroscope_readings) == 10:
            accel_variance = sum(statistics.variance([reading[i] for reading in self.accelerometer_readings]) for i in range(3)) / 3
            gyro_variance = sum(statistics.variance([reading[i] for reading in self.gyroscope_readings]) for i in range(3)) / 3
            print(f"Accelerometer Variance: {accel_variance}, Gyroscope Variance: {gyro_variance}")
            if accel_variance < 30000 and gyro_variance < 30000:  # Adjusted threshold for the range of -300 to 300
                return True
        return False

    def run_sleep_mode(self):
        while True:
            battery_level = self.get_battery_level()
            self.update_sensor_readings()
            idle_mode = self.check_idle_mode()
            mode = self.determine_mode(battery_level, idle_mode)
            print("Battery level:", battery_level, "% - Mode:", mode)
            time.sleep(3)  # Adjust delay based on update frequency

    def determine_mode(self, battery_level, idle_mode):
        if battery_level < 10:
            self.enable_accelerometer(False)
            self.enable_gyroscope(False)
            mode = "Low Power Mode, and the accelerometer and gyroscope are off."
            # and delay the other sensors by 40 for example
            
        elif battery_level < 30:
            self.enable_accelerometer(True)
            self.enable_gyroscope(True)
            mode = "Delayed Mode due to low battery. Delay by 10."
            # here is the code for delay the sensors readings by 10 for example
            # or whatever you think will be efficient
            
            if idle_mode:
                mode = "Delayed Mode for the battery level and the idle condition. Delay by 20."
                # here is the code for delaying by another 10 because of the idle condition
            
        else:
            self.enable_accelerometer(True)
            self.enable_gyroscope(True)
            mode = "Normal Mode"
            if idle_mode:
                mode = "Delayed Mode due to idle condition. Delay by 10."
                # here is the code for delaying by 10 because of the idle condition
        return mode

# Create an instance of the Smartwatch class
smartwatch = Smartwatch()

# Run the sleep mode functionality
smartwatch.run_sleep_mode()
