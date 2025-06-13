#   Copyright [2025] [Kenneth Horne,Jr]

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import functools
import math
from collections.abc import Sequence

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter as FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator as FixedLocator
from mpl_toolkits.mplot3d.proj3d import transform
from numpy.ma.core import append
from sympy import sieve, primerange

# plt.rcParams["animation.html"] = "jshtml"
plt.rcParams["animation.html"] = "html5"

plt.rcParams["figure.subplot.left"] = 0.01  # the left side of the subplots of the figure
plt.rcParams["figure.subplot.right"] = .99  # the right side of the subplots of the figure
plt.rcParams["figure.subplot.bottom"] = 0.01  # the bottom of the subplots of the figure
plt.rcParams["figure.subplot.top"] = .99  # the top of the subplots of the figure

FRAMES_PER_NUMBER = 8
MAX_NUMBER_TO_CHECK = 7

plotCenter = (0, 0)
global theta
global radius


def convertRadiansToArrayIndex(angleToStartFrom, radianAngle):
	if (angleToStartFrom > 2*np.pi):
		angleToStartFrom = angleToStartFrom % (2*np.pi)
	if (radianAngle > 2*np.pi):
		radianAngle=radianAngle % (2*np.pi)
	if radianAngle < angleToStartFrom:
		radianAngle = radianAngle+2*np.pi
#	integerIndex = math.ceil(radianAngle * FRAMES_PER_NUMBER / (2 * np.pi)) % (theta.size)
#	integerIndex = int (radianAngle * FRAMES_PER_NUMBER / (2 * np.pi)) % (theta.size)
	integerIndex = int ((radianAngle / (4 * np.pi)) * (theta.size))
	return integerIndex

class CircleCycleForOnePrime:
	primeNumber = 0
	center = (0, 0)
	
	logScaling = False
	cycleRadius = 0
	currentIndicatorValue = 0
	RING_WIDTH = 2
	INDICATOR_WIDTH = RING_WIDTH * 2
	ANGLE_TO_INDICATE_ZERO = 3*np.pi/2
	INDICATOR_ARC_UNIT_RATIO = .8
	zeroRegionArc = None
	nonzeroRegionArc = None
	indicatorArc = None
	patches = []
	lines = []
	modIndicatorZero = False
	unitSlice = None
	axesForCycle = None
	ZERO_START_FOR_INDICATOR_ARC = ANGLE_TO_INDICATE_ZERO
	
	def __init__(self, center, primeNumber, axes, logScaling):
		self.center = center
		self.primeNumber = int(primeNumber)
		self.setLogScaling(logScaling)
		self.axesForCycle = axes
		self.constructPrimeModCircleRegions()
		self.setIndicator(0)
	
	def setLogScaling(self, useLogScaling, minimumRadius=1):
		self.logScaling = useLogScaling
		self.cycleRadius = self.primeNumber
		if (self.logScaling):
			self.cycleRadius = math.log(self.primeNumber)
		if (self.cycleRadius < minimumRadius):
			self.cycleRadius = minimumRadius + 2
	
	def constructPrimeModCircleRegions(self):
		# divide a circle into primeNumber slices
		self.unitSlice = 2 * math.pi / self.primeNumber
		self.ZERO_START_FOR_INDICATOR_ARC = self.ANGLE_TO_INDICATE_ZERO - self.INDICATOR_ARC_UNIT_RATIO * self.unitSlice / 2
		self.ZERO_END_FOR_INDICATOR_ARC = self.ZERO_START_FOR_INDICATOR_ARC + self.INDICATOR_ARC_UNIT_RATIO * self.unitSlice
		zeroStartAngle = self.ANGLE_TO_INDICATE_ZERO - (self.unitSlice / 2)
		zeroRegionBeginIndex = convertRadiansToArrayIndex(0, zeroStartAngle)
		zeroEndAngle = self.ANGLE_TO_INDICATE_ZERO + (self.unitSlice / 2)
		zeroRegionEndIndex = convertRadiansToArrayIndex(zeroStartAngle, zeroEndAngle)
		safeRegionEndIndex = convertRadiansToArrayIndex(zeroEndAngle, zeroStartAngle)
		zeroRegionPoints = theta[zeroRegionBeginIndex:zeroRegionEndIndex + 1]
		self.zeroRegionArc = Line2D(zeroRegionPoints,
		                            self.cycleRadius * radius[zeroRegionBeginIndex:zeroRegionBeginIndex + len(
			                            zeroRegionPoints)],
		                            linewidth=CircleCycleForOnePrime.RING_WIDTH,
		                            color='red')
		self.axesForCycle.add_line(self.zeroRegionArc)
		nonzeroRegionPoints = theta[zeroRegionEndIndex:safeRegionEndIndex + 1]
		self.nonzeroRegionArc = Line2D(nonzeroRegionPoints,
		                               self.cycleRadius * radius[zeroRegionEndIndex:zeroRegionEndIndex + len(
			                               nonzeroRegionPoints)],
		                               linewidth=CircleCycleForOnePrime.RING_WIDTH,
		                               color='green')
		self.axesForCycle.add_line(self.nonzeroRegionArc)
		indicatorZeroBeginIndex = convertRadiansToArrayIndex(0, self.ZERO_START_FOR_INDICATOR_ARC)
		indicatorZeroEndIndex = convertRadiansToArrayIndex(self.ZERO_START_FOR_INDICATOR_ARC,
		                                                   self.ZERO_END_FOR_INDICATOR_ARC)
		indicatorPoints = theta[indicatorZeroBeginIndex:indicatorZeroEndIndex + 1]
		self.indicatorArc = Line2D(indicatorPoints,
		                           self.cycleRadius * radius[indicatorZeroBeginIndex:indicatorZeroBeginIndex + len(
			                           indicatorPoints)],
		                           linewidth=CircleCycleForOnePrime.INDICATOR_WIDTH,
		                           color='blue')
		self.axesForCycle.add_line(self.indicatorArc)
		
		self.lines = [self.zeroRegionArc, self.nonzeroRegionArc, self.indicatorArc]
	
	def setIndicator(self, newIndicatorValue=None):
		if (newIndicatorValue is not None):
			# convert the remainder to a percent
			self.currentIndicatorValue = (newIndicatorValue % self.primeNumber)
			self.modIndicatorZero = (self.currentIndicatorValue == 0)
		startingRegion = self.ZERO_START_FOR_INDICATOR_ARC + self.unitSlice * self.currentIndicatorValue
		
		self.indicatorArc.set_xdata(theta[convertRadiansToArrayIndex(0,startingRegion):convertRadiansToArrayIndex(
			startingRegion, startingRegion + (self.unitSlice) * self.INDICATOR_ARC_UNIT_RATIO)])
		self.indicatorArc.set_ydata(self.cycleRadius * radius[convertRadiansToArrayIndex(
			0,startingRegion):convertRadiansToArrayIndex(startingRegion,
			startingRegion + (self.unitSlice) * self.INDICATOR_ARC_UNIT_RATIO)])


def setCountNumberDisplayed(currentNumber):
	countingString = '{:.2f}'.format(currentNumber)
	numberDisplay.set_text(countingString)


def animate(frameNumber, cycleContainer):
	#	if (frameToPauseUntil <= frameNumber):
	currentNumber = startingValue + frameNumber / FRAMES_PER_NUMBER
	# only checxk for a new cycle if this is an integer
	currentNumberIsIntegerAndNotDivisibleByAnyCycle = (currentNumber.is_integer())
	
	linelist = []
	for oneCycle in cycleContainer:
		oneCycle.setIndicator(currentNumber)
		# check if current number is divisible by cycle prime
		currentNumberIsIntegerAndNotDivisibleByAnyCycle = currentNumberIsIntegerAndNotDivisibleByAnyCycle and not oneCycle.modIndicatorZero
		#linelist.extend(oneCycle.lines)
		ax.autoscale_view()
	# if current number isn't divisible by any cycle, it's a prime
	if (currentNumberIsIntegerAndNotDivisibleByAnyCycle):
		#		ax.tick_params(labelsize=computedLabelSize)
		# got a new prime
		newCycle = CircleCycleForOnePrime(plotCenter, currentNumber, ax, False)
		cycles.append(newCycle)
#		currentTicks = ax.yaxis.get_major_locator().locs
#		ax.yaxis.set_major_locator(FixedLocator(append(currentTicks,currentNumber)))
		#linelist.extend(newCycle.lines)
		fig.canvas.flush_events()
	
	setCountNumberDisplayed(currentNumber)
	return LineCollection(linelist)
	        #, numberDisplay)


# def getTextLocation():
#	textCoorinates = (ax.get_xlim()[0], ax.get_ylim()[1])
#	print(textCoorinates, ' ' , ax.get_xlim(), ' ' , ax.get_ylim())
#	return textCoorinates
halfTheta = np.linspace(0, 2 * np.pi, FRAMES_PER_NUMBER)
theta = np.append(halfTheta, halfTheta+2*np.pi)
radius = np.full(theta.shape, 1)

plt.autoscale(enable=True, axis='both')
fig = plt.figure()
ax = plt.subplot(projection='polar')
#ax.yaxis.set_major_locator(FixedLocator(np.fromiter(primerange(MAX_NUMBER_TO_CHECK), int)))

numberDisplay = ax.text(np.pi, 1, '2', transform=ax.transData)
#transProjectionAffine  transScale transShift

#ax.add_artist(numberDisplay)

cycles = []
twoCycle = CircleCycleForOnePrime(plotCenter, 2, ax, False)
cycles.append(twoCycle)

ax.autoscale_view()
startingValue = 2

anim = functools.partial(animate, cycleContainer=cycles)
numberOfFrames = FRAMES_PER_NUMBER * MAX_NUMBER_TO_CHECK
ani = animation.FuncAnimation(fig, anim, numberOfFrames, save_count=numberOfFrames, repeat=False)
ani.save(filename='primeClock50.mp4', writer=FFMpegWriter(fps=20))
# plt.show(block=True)

# MAX_NUMBER_TO_CHECK = 100
# numberOfFrames = FRAMES_PER_NUMBER * MAX_NUMBER_TO_CHECK
# ani = animation.FuncAnimation(fig, anim, numberOfFrames, save_count=numberOfFrames, repeat=False)
# ani.save(filename='primeClock100.mp4', fps='20')

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
