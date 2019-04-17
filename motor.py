import RPi.GPIO as gpio
import time

def stop():
        print('Stop')
        gpio.output(13, False)
        gpio.output(12, False)
        gpio.output(16, False)
        gpio.output(15, False)
        time.sleep(0.05)

def motor_init():
	gpio.setmode(gpio.BOARD)
	gpio.setup(12, gpio.OUT)
	gpio.setup(13, gpio.OUT)
	gpio.setup(15, gpio.OUT)
	gpio.setup(16, gpio.OUT)
	print('Motors initiated')

def forward():
	print('Forward')
	gpio.output(13, True)
        gpio.output(15, False)
        gpio.output(16, True)
        gpio.output(12, False)
        time.sleep(0.05)
	stop()

def backwards():
	print('Reverse')
	gpio.output(13, False)
        gpio.output(15, True)
        gpio.output(16, False)
        gpio.output(12, True)
        time.sleep(0.2)
	stop()	

def left():
	print('Left')
	gpio.output(16, True)
        gpio.output(12, False)
	time.sleep(0.05)
	gpio.output(16, False)
	stop()

def right():
	print('Right')
	gpio.output(13, True)
        gpio.output(15, False)
	time.sleep(0.05)
	gpio.output(13, False)
	stop()

print('===================================================')	 
print('===================================================')

def kinhsh(char):	
		if (char == "w"):
			forward()
		if (char  == "a"):
                        left()
		if (char == "d"):
                        right()
		if (char== "s"):
                        backwards()  
