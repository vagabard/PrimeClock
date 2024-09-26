import functools
import math
import time

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter as FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.projections.polar import RadialLocator
from matplotlib.ticker import FixedLocator
from sympy import sieve
from sympy.physics.quantum.circuitplot import pyplot
from sympy.polys.numberfields.utilities import is_int

plt.rcParams["figure.subplot.left"] = 0.05  # the left side of the subplots of the figure
plt.rcParams["figure.subplot.right"] = .95  # the right side of the subplots of the figure
plt.rcParams["figure.subplot.bottom"] = 0.05  # the bottom of the subplots of the figure
plt.rcParams["figure.subplot.top"] = .95  # the top of the subplots of the figure

#plt.rcParams["polaraxes.grid"]

FRAMES_PER_NUMBER = 128
MAX_NUMBER_TO_CHECK = 50


plotCenter = (0, 0)
background_image = None
lines_to_draw_dynamic = []
lines_to_draw_static = []
prime_indicator_line = None
last_prime_found = 0
indicator_to_annotator = {}

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    print(backend)
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WxAgg':
	    print("test")
        #f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

class PrimeFixedLocator(FixedLocator):
    primeTicks = []
    oldVmax = 0

    def __init__(self, locs):
	    super().__init__(locs,None)

    def __call__(self):
        return self.tick_values()

    def tick_values(self):
        """
        Return the locations of the ticks.

        .. note::

            Because the values are fixed, vmin and vmax are not used in this
            method.

        """
        vmin, vmax = self.axis.get_view_interval()

        if (vmax > self.oldVmax):
            self.primeTicks = list(sieve.primerange(1, vmax))
            self.oldVmax = vmax
        return self.primeTicks
    
    
 #This is for speed.  An array of radian points is computed once.   After than the array is accessed and the nearest point is used
 #rather than allocating more arrays and calculating more appoximations
def convertRadiansToArrayIndex(angleToStartFrom, radianAngle):
	if (angleToStartFrom > 2 * np.pi):
		angleToStartFrom = angleToStartFrom % (2 * np.pi)
	if (angleToStartFrom < 0):
		angleToStartFrom = 2*np.pi
	if (radianAngle > 2 * np.pi):
		radianAngle = radianAngle % (2 * np.pi)
	if radianAngle < angleToStartFrom:
		radianAngle = radianAngle + 2 * np.pi
	#	integerIndex = math.ceil(radianAngle * FRAMES_PER_NUMBER / (2 * np.pi)) % (theta.size)
	#	integerIndex = int (radianAngle * FRAMES_PER_NUMBER / (2 * np.pi)) % (theta.size)
	integerIndex = int((radianAngle / (4 * np.pi)) * (theta.size))
	return integerIndex


class CircleCycleForOnePrime:
	primeNumber = 0
	center = (0, 0)
	
	logScaling = False
	cycleRadius = 0
	currentIndicatorValue = 0
	RING_WIDTH = 2
	INDICATOR_WIDTH =RING_WIDTH * 1.5
	INDICATOR_LENGTH = None
	ANGLE_TO_INDICATE_ZERO = 0 #  if this is zero the conversion has problems, fix this sometime
	INDICATOR_ARC_UNIT_RATIO = .3
	zeroRegionArc = None
	nonzeroRegionArc = None
	indicatorArc = None
	unitSlice = None
	axesForCycle = None
	ZERO_START_FOR_INDICATOR_ARC = ANGLE_TO_INDICATE_ZERO
	
	def __init__(self, center, primeNumber, axes):
		self.center = center
		self.primeNumber = int(primeNumber)
		self.setScaling()
		self.axesForCycle = axes
		self.constructPrimeModCircleRegions()
	
	#		self.setIndicator(0)
	
	def setScaling(self, minimumRadius=1):
		self.cycleRadius = self.primeNumber
		if (self.cycleRadius < minimumRadius):
			self.cycleRadius = minimumRadius + 2
	
	def constructPrimeModCircleRegions(self):
		global lines_to_draw_static
		
		# divide a circle into primeNumber slices
		self.unitSlice = 2 * math.pi / self.primeNumber
		self.ZERO_START_FOR_INDICATOR_ARC = self.ANGLE_TO_INDICATE_ZERO - self.INDICATOR_ARC_UNIT_RATIO * self.unitSlice / 2
		self.ZERO_END_FOR_INDICATOR_ARC = self.ZERO_START_FOR_INDICATOR_ARC + self.INDICATOR_ARC_UNIT_RATIO * self.unitSlice
		zeroStartAngle = self.ANGLE_TO_INDICATE_ZERO - (self.unitSlice / 2)
		zeroRegionBeginIndex = convertRadiansToArrayIndex(0, zeroStartAngle)
		zeroEndAngle = self.ANGLE_TO_INDICATE_ZERO + (self.unitSlice / 2)
		zeroRegionEndIndex = convertRadiansToArrayIndex(zeroStartAngle, zeroEndAngle)
		zeroRegionPoints = theta[zeroRegionBeginIndex:zeroRegionEndIndex + 1]
		self.zeroRegionArc = Line2D(zeroRegionPoints,
		                            self.cycleRadius * radius[zeroRegionBeginIndex:zeroRegionBeginIndex + len(
			                            zeroRegionPoints)],
		                            linewidth=CircleCycleForOnePrime.RING_WIDTH,
		                            color='red', animated=False)
		self.axesForCycle.add_line(self.zeroRegionArc)
		lines_to_draw_static.append(self.zeroRegionArc)

		#this was for a green arc to indicate the nonzero region of mods
#		safeRegionEndIndex = convertRadiansToArrayIndex(zeroEndAngle, zeroStartAngle)
#		nonzeroRegionPoints = theta[zeroRegionEndIndex:safeRegionEndIndex + 1]
#		self.nonzeroRegionArc = Line2D(nonzeroRegionPoints,
#		                               self.cycleRadius * radius[zeroRegionEndIndex:zeroRegionEndIndex + len(
#			                               nonzeroRegionPoints)],
#		                               linewidth=CircleCycleForOnePrime.RING_WIDTH,
#		                               color='green')
#		self.axesForCycle.add_line(self.nonzeroRegionArc)
#		lines_to_draw_static.append(self.nonzeroRegionArc)
#		self.axesForCycle.autoscale_view()
	
	def constructPrimeModCircleIndicator(self):
		global lines_to_draw_dynamic
		global prime_indicator_line
		# divide a circle into primeNumber slices
		indicatorZeroBeginIndex = convertRadiansToArrayIndex(0,
		                                                     5 * np.pi / 4)  # convertRadiansToArrayIndex(0, self.ZERO_START_FOR_INDICATOR_ARC)
		indicatorZeroEndIndex = convertRadiansToArrayIndex(indicatorZeroBeginIndex,
		                                                   7 * np.pi / 4)  # convertRadiansToArrayIndex(self.ZERO_START_FOR_INDICATOR_ARC, self.ZERO_END_FOR_INDICATOR_ARC)
		indicatorPoints = theta[indicatorZeroBeginIndex:indicatorZeroEndIndex + 1]
		radiusPoints = self.cycleRadius * radius[indicatorZeroBeginIndex:indicatorZeroBeginIndex + len(indicatorPoints)]
		self.indicatorArc = Line2D(indicatorPoints, radiusPoints,
		                           linewidth=CircleCycleForOnePrime.INDICATOR_WIDTH,
		                           color='blue', animated=True)
		self.axesForCycle.add_line(self.indicatorArc)
		lines_to_draw_dynamic.append(self.indicatorArc)
		prime_indicator_line = Line2D([0,0], [0,self.cycleRadius],linewidth=CircleCycleForOnePrime.INDICATOR_WIDTH,color='green',animated=True)
		self.axesForCycle.add_line(prime_indicator_line)
		lines_to_draw_dynamic.append(prime_indicator_line)
	
	def is_number_multiple_of_base(self, testNumber):
		return ((testNumber % self.primeNumber) == 0)
	
	def setIndicator(self, newIndicatorValue=None):
		if self.indicatorArc is None:
			self.constructPrimeModCircleIndicator()
		if self.INDICATOR_LENGTH is None:
			self.INDICATOR_LENGTH = convertRadiansToArrayIndex(0,(self.unitSlice) * self.INDICATOR_ARC_UNIT_RATIO)
			if self.INDICATOR_LENGTH <5:
				self.INDICATOR_LENGTH = 5
			self.indicatorArc.set_ydata(self.cycleRadius * radius[0:self.INDICATOR_LENGTH])
		if (newIndicatorValue is not None):
			# convert the remainder to a percent
			self.currentIndicatorValue = (newIndicatorValue % self.primeNumber)
			if (newIndicatorValue.is_integer()):
				arcLabel = self.primeNumber,'  mods ',newIndicatorValue,' to ',  self.currentIndicatorValue
				self.indicatorArc.set_label(arcLabel)
		startingRegion = self.ZERO_START_FOR_INDICATOR_ARC + self.unitSlice * self.currentIndicatorValue
		startingIndex = convertRadiansToArrayIndex(0, startingRegion)
		endingIndex = startingIndex+self.INDICATOR_LENGTH
		self.indicatorArc.set_xdata(theta[startingIndex:endingIndex])


def setCountNumberDisplayed(currentNumber):
	countingString = '{:.2f}'.format(currentNumber)
	numberDisplay.set(text=countingString)


def create_background_image():
	global lines_to_draw_static
	global background_image
	for one_line in lines_to_draw_static:
		ax.draw_artist(one_line)
	ax.autoscale_view()
	fig.canvas.flush_events()
	pyplot.pause(.01)
	# get copy of entire figure (everything inside fig.bbox) sans animated artist
	background_image = fig.canvas.copy_from_bbox(fig.bbox)

def set_prime_indicator_green(prime_so_far):
	if (prime_so_far):
		prime_indicator_line.set_color('green')
	else:
		prime_indicator_line.set_color('red')

def animate_helper(frameNumber, cycleContainer):
	global last_prime_found
	currentNumber = startingValue + frameNumber / FRAMES_PER_NUMBER
	# only checxk for a new cycle if this is an integer
	currentNumberNotDivisibleByAnyCycle = True
	found_a_prime = False
	should_prime_indicator_be_green = True
	
	for oneCycle in cycleContainer:
		oneCycle.setIndicator(currentNumber)
		# check if current number is divisible by cycle prime
		currentNumberNotDivisibleByAnyCycle = currentNumberNotDivisibleByAnyCycle and not oneCycle.is_number_multiple_of_base(currentNumber)
		should_prime_indicator_be_green = should_prime_indicator_be_green and not oneCycle.is_number_multiple_of_base(int(currentNumber+.5))
	# if current number isn't divisible by any cycle, it's a prime
	if (currentNumber.is_integer() and  currentNumberNotDivisibleByAnyCycle):
		found_a_prime = True
		currentNumber = int(currentNumber)
		last_prime_found=currentNumber
		# got a new prime
		newCycle = CircleCycleForOnePrime(plotCenter, currentNumber, ax)
		cycles.append(newCycle)
		newCycle.setIndicator(currentNumber)
	
	setCountNumberDisplayed(currentNumber)
	set_prime_indicator_green(should_prime_indicator_be_green) # or (int(currentNumber) == last_prime_found))
	if (frameNumber % 100 == 0):
		print(frameNumber)
		
	return lines_to_draw_dynamic, currentNumberNotDivisibleByAnyCycle, found_a_prime

def animate(frameNumber, cycleContainer):
	animate_helper(frameNumber,cycleContainer)
	return lines_to_draw_dynamic


def handleOneFrame(frameNumber, cycleContainer):
	global background_image
	lines,numberDivisibleByCycle, found_prime = animate_helper(frameNumber, cycleContainer)
	if found_prime:
		create_background_image()
	fig.canvas.restore_region(background_image)
	for oneAnimatedObject in lines_to_draw_dynamic:
		ax.draw_artist(oneAnimatedObject)
	# show the result to the screen, this pushes the updated RGBA buffer from the
	# renderer to the GUI framework so you can see it
	fig.canvas.blit(fig.bbox)
	fig.canvas.flush_events()
	return lines_to_draw_dynamic


#, numberDisplay)


# def getTextLocation():
#	textCoorinates = (ax.get_xlim()[0], ax.get_ylim()[1])
#	print(textCoorinates, ' ' , ax.get_xlim(), ' ' , ax.get_ylim())
#	return textCoorinates
halfTheta = np.linspace(0, 2 * np.pi, FRAMES_PER_NUMBER)
theta = np.append(halfTheta, halfTheta + 2 * np.pi)
radius = np.full(theta.shape, 1)

#matplotlib.use('TkAgg')
matplotlib.use('QtAgg')
#matplotlib.use('WxAgg')


fig = plt.figure()#figsize=(10.24,7.68))
#move_figure(fig,2800,1300)
ax = fig.add_subplot(projection='polar')
primeLocator = PrimeFixedLocator([2])# FixedLocator([2])
radialLoc = RadialLocator(primeLocator)
radialLoc._axes = ax
ax.set_rlabel_position(157.5)
ax.set_theta_offset(3*np.pi / 2)
ax.yaxis.set_major_locator(radialLoc)


numberDisplay = ax.annotate('2',(np.pi, 1),  animated=True) #ax.text(np.pi, 1, '2', transform=ax.transData, animated=True)
lines_to_draw_dynamic.append(numberDisplay)
plt.ioff()
plt.show(block=False)

cycles = []
twoCycle = CircleCycleForOnePrime(plotCenter, 2, ax)
cycles.append(twoCycle)
create_background_image()
twoCycle.setIndicator(2)

ax.draw_artist(twoCycle.indicatorArc)
fig.canvas.blit(fig.bbox)
fig.canvas.flush_events()
plt.pause(0.001)
fig.canvas.restore_region(background_image)

startingValue = 2
anim = functools.partial(animate, cycleContainer=cycles)
numberOfFrames = FRAMES_PER_NUMBER * MAX_NUMBER_TO_CHECK
#ani = animation.FuncAnimation(fig, anim, numberOfFrames, save_count=numberOfFrames, repeat=False, )

#ani.save(filename='primeClock50.mp4', writer=FFMpegWriter(fps=30 ))

for frameNumber in range(numberOfFrames):
	handleOneFrame(frameNumber, cycles)
	plt.pause(.001)

MAX_NUMBER_TO_CHECK = 100
numberOfFrames = FRAMES_PER_NUMBER * MAX_NUMBER_TO_CHECK
#startTime = time.time()
fmpgWriter = FFMpegWriter(fps=50)
#ani = animation.FuncAnimation(fig, anim, numberOfFrames, repeat=False, blit=True, interval=1)
#ni.save(filename='primeClock100.mp4',writer=fmpgWriter)
#endTime = time.time()
#print("time taken ",(endTime-startTime))
# MAX_NUMBER_TO_CHECK = 250
# numberOfFrames = FRAMES_PER_NUMBER * MAX_NUMBER_TO_CHECK
# ani = animation.FuncAnimation(fig, anim, numberOfFrames, save_count=numberOfFrames, repeat=False)
# ani.save(filename='primeClock250.mp4', fps='20')

# MAX_NUMBER_TO_CHECK = 500
# numberOfFrames = FRAMES_PER_NUMBER * MAX_NUMBER_TO_CHECK
# ani = animation.FuncAnimation(fig, anim, numberOfFrames, save_count=numberOfFrames, repeat=False)
# ani.save(filename='primeClock500.mp4', fps='20')

# plt.show()
# put this here so ani doesn't get garbage collected
ani
