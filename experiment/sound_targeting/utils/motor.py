import serial
import time


class AdafruitStepper:
    """
    Class to control two stepper motors.
    When the Arduino is switched on, the current motor position is defined as origin (0,0).
    All x,y target values are absolute values w.r.t. this origin.
    The class is tied to serial commands defined in Accel_MultiStepper_Serial.ino

    :param port: string, e.g. 'COM3' for windows or '/dev/ttyACM0' under linux
        # under Linux:
        # 1. find PORTNAME: dmesg | grep tty
        # 2. give permission: sudo chmod 666 /dev/PORTNAME

    Example:
        To move to absolute step position (x,y) = (3445,56), the following serial command is sent:
        Serial Out: 0000344500000056

        The Arduino is expected to only send bytes whenever all motors have moved to the target position.
    """

    def __init__(self, port):
        self.ser = serial.Serial(port, 9600, timeout=1)  # Linux, check for port
        time.sleep(4)
        self.goto(0, 0)

    def goto(self, x, y):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        # turn into serial command
        x = int(x)
        y = int(y)
        cmd = str(x)[-8:].zfill(8) + str(y)[-8:].zfill(8)
        # send
        print(cmd)
        self.ser.write(cmd.encode())
        # wait for any feedback signal
        while not self.ser.in_waiting:  # stops upon feedback
            time.sleep(.1)
        self.ser.reset_input_buffer()

    def goto_cm(self, xcm, ycm):
        x, y = self.cm2step(xcm, ycm)
        self.goto(x, y)

    def cm2step(self, xcm, ycm):
        f = 1596.01  # empirical
        xsteps = int(f * xcm)
        ysteps = int(f * ycm)
        return xsteps, ysteps

    def __del__(self):
        # Return back to origin upon destruction
        self.goto(0, 0)
        self.ser.close()
