   # Copyright [2025] [Kenneth Horne,Jr]
   #
   # Licensed under the Apache License, Version 2.0 (the "License");
   # you may not use this file except in compliance with the License.
   # You may obtain a copy of the License at
   #
   #   http://www.apache.org/licenses/LICENSE-2.0
   #
   # Unless required by applicable law or agreed to in writing, software
   # distributed under the License is distributed on an "AS IS" BASIS,
   # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   # See the License for the specific language governing permissions and
   # limitations under the License.

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
from matplotlib.text import OffsetFrom
from matplotlib.ticker import FixedLocator, NullLocator, NullFormatter
from sympy import sieve, primepi

#plt.rcParams["figure.subplot.left"] = 0.05  # the left side of the subplots of the figure
plt.rcParams["figure.subplot.right"] = .95  # the right side of the subplots of the figure
plt.rcParams["figure.subplot.bottom"] = 0.05  # the bottom of the subplots of the figure
plt.rcParams["figure.subplot.top"] = .90  # the top of the subplots of the figure

#plt.rcParams["polaraxes.grid"]

SHOW_LIVE_ANIMATION = False
WRITE_MOVIE_FILE = True
NEW_RINGS_APPEAR_IN_CENTER = False # 2 would be the outermost ring, then 3 inside that, then 5,etc
#it should look better that way, but getting the spaces right is a pain
FRAMES_PER_NUMBER = 128 #how many sections to break each ring into
MAX_NUMBER_TO_CHECK = 50

plotCenter = (0, 0)
background_image = None
lines_to_draw_dynamic = []
lines_to_draw_static = []
prime_indicator_line = None
last_prime_found = 0
indicator_to_annotator = {}
annotator_mod_result_list_for_layout = []


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
	old_max = 0
	
	def __init__(self, locs):
		super().__init__(locs, None)
	
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
		if (vmin <= vmax):
			if (vmax > self.old_max):
				self.primeTicks = list(sieve.primerange(1, vmax))
				self.old_max = vmax
		else:
			#we have an inverted axis
			if (vmin > self.old_max):
				self.primeTicks = list(sieve.primerange(1, vmin))
				self.old_max = vmin
		return self.primeTicks


#This is for speed.  An array of radian points is computed once.   After than the array is accessed and the nearest point is used
#rather than allocating more arrays and calculating more appoximations
def convertRadiansToArrayIndex(angleToStartFrom, radianAngle):
	if (angleToStartFrom > 2 * np.pi):
		angleToStartFrom = angleToStartFrom % (2 * np.pi)
	if (angleToStartFrom < 0):
		angleToStartFrom = 2 * np.pi
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
	RING_WIDTH = 1
	INDICATOR_WIDTH = RING_WIDTH * 1.5
	INDICATOR_LENGTH_IN_ARRAY_INDEX = None
	ANGLE_TO_INDICATE_ZERO = 0  #  if this is zero the conversion has problems, fix this sometime
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
		index_within_prime_numbers = primepi(self.primeNumber)
		self.cycleRadius = self.primeNumber

	def constructPrimeModCircleRegions(self):
		global lines_to_draw_static
		
		# divide a circle into primeNumber slices
		self.unitSlice = 2 * math.pi / self.primeNumber
		zeroStartAngle = self.ANGLE_TO_INDICATE_ZERO - (self.unitSlice / 2)
		zeroRegionBeginIndex = convertRadiansToArrayIndex(0, zeroStartAngle)
		zeroEndAngle = self.ANGLE_TO_INDICATE_ZERO + (self.unitSlice / 2)
		zeroRegionEndIndex = convertRadiansToArrayIndex(zeroStartAngle, zeroEndAngle)
		zeroRegionPoints = theta[zeroRegionBeginIndex:zeroRegionEndIndex + 1]

#		self.zeroRegionArc = Line2D(zeroRegionPoints,self.cycleRadius * radius[zeroRegionBeginIndex:zeroRegionBeginIndex + len(zeroRegionPoints)],linewidth=CircleCycleForOnePrime.RING_WIDTH,color='red', animated=False)
#		self.axesForCycle.add_line(self.zeroRegionArc)
#		lines_to_draw_dynamic.append(self.zeroRegionArc)
	
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
		global indicator_to_annotator
		global lines_to_draw_dynamic
		global annotator_mod_result_list_for_layout
		self.compute_indicator_length()
		# divide a circle into primeNumber slices
		ZERO_START_FOR_INDICATOR_ARC = self.ANGLE_TO_INDICATE_ZERO - (self.unitSlice / 2)# * self.INDICATOR_ARC_UNIT_RATIO
		indicatorZeroBeginIndex = convertRadiansToArrayIndex(0,ZERO_START_FOR_INDICATOR_ARC)
		indicatorZeroEndIndex = indicatorZeroBeginIndex+self.INDICATOR_LENGTH_IN_ARRAY_INDEX #convertRadiansToArrayIndex(ZERO_START_FOR_INDICATOR_ARC, ZERO_END_FOR_INDICATOR_ARC)
		indicatorPoints = theta[indicatorZeroBeginIndex:indicatorZeroEndIndex]
		radiusPoints = self.cycleRadius * radius[0:len(indicatorPoints)]
		self.indicatorArc = Line2D(indicatorPoints, radiusPoints,
		                           linewidth=CircleCycleForOnePrime.INDICATOR_WIDTH,
		                           color='blue', animated=True)
#		self.indicatorArc.set_ydata(self.cycleRadius * radius[0:self.INDICATOR_LENGTH])
		self.axesForCycle.add_line(self.indicatorArc)
		lines_to_draw_dynamic.append(self.indicatorArc)
		if (prime_indicator_line == None):
			prime_indicator_line = Line2D([0, 0], [0, self.cycleRadius],
		                              linewidth=CircleCycleForOnePrime.INDICATOR_WIDTH, color='green',
		                              animated=True)
			self.axesForCycle.add_line(prime_indicator_line)
			lines_to_draw_dynamic.append(prime_indicator_line)
		else:
			prime_indicator_line.set_ydata([0,self.cycleRadius])
		label_text_1 = "{0:>4}=".format(self.primeNumber)
		annotation_row = - 3 * len(indicator_to_annotator)
		coords_to_use = OffsetFrom(numberDisplay, (1, -.25))  #'figure fraction'
		label_annotation = ax.annotate(label_text_1, xy=(14, annotation_row), xycoords=coords_to_use, ha='right', va='center')
		#skip space for the beginning of label , the length of the label, and how many digits the prime number is
		mod_result_xcoord = label_annotation.xyann[0] + .03 + .01 * len(str(self.primeNumber))
		mod_result = ax.annotate('0', xy=(mod_result_xcoord, annotation_row), xycoords=coords_to_use, ha='right', va='center', animated=True)
		annotator_mod_result_list_for_layout.append((label_annotation, mod_result))
		lines_to_draw_dynamic.append(mod_result)
		
		indicator_to_annotator[self.indicatorArc] = mod_result
		layout_legend()
	
	def is_number_multiple_of_base(self, testNumber):
		return ((testNumber % self.primeNumber) == 0)
	
	def setIndicator(self, newIndicatorValue=None):
		global indicator_to_annotator
		if self.indicatorArc is None:
			self.constructPrimeModCircleIndicator()
		if (newIndicatorValue is not None):
			# convert the remainder to a percent
			self.currentIndicatorValue = (newIndicatorValue % self.primeNumber)
			if (newIndicatorValue.is_integer()):
				arcLabel = self.primeNumber, '  mods ', newIndicatorValue, ' to ', self.currentIndicatorValue
				self.indicatorArc.set_label(arcLabel)
		startingRegion = (self.ZERO_START_FOR_INDICATOR_ARC - self.unitSlice/2) + self.unitSlice * self.currentIndicatorValue
		startingIndex = convertRadiansToArrayIndex(0, startingRegion)
		endingIndex = startingIndex + self.INDICATOR_LENGTH_IN_ARRAY_INDEX
		self.indicatorArc.set_xdata(theta[startingIndex:endingIndex])
		mod_result_to_display = '{:>3.1f}'.format(self.currentIndicatorValue)
		if ((self.currentIndicatorValue > (self.primeNumber-.5) or self.currentIndicatorValue < .5) and int(newIndicatorValue) != self.primeNumber):
			indicator_to_annotator[self.indicatorArc].set(color='red')
		else:
			indicator_to_annotator[self.indicatorArc].set(color='green')
		
		indicator_to_annotator[self.indicatorArc].set(text=mod_result_to_display)

	def compute_indicator_length(self):
		if self.INDICATOR_LENGTH_IN_ARRAY_INDEX is None:
			#			self.INDICATOR_LENGTH = convertRadiansToArrayIndex(0, (self.unitSlice) * self.INDICATOR_ARC_UNIT_RATIO)
			self.INDICATOR_LENGTH_IN_ARRAY_INDEX = convertRadiansToArrayIndex(0, self.unitSlice)
			# set a minimum size to make sure we can see the indicator
			if self.INDICATOR_LENGTH_IN_ARRAY_INDEX < 5:
				self.INDICATOR_LENGTH_IN_ARRAY_INDEX = 5


def layout_legend():
	global annotator_mod_result_list_for_layout
	span_of_labels_in_points = len(annotator_mod_result_list_for_layout) * 10
	starting_location = span_of_labels_in_points / 2
	percent_of_labels_done = 0.0
	for label, result in annotator_mod_result_list_for_layout:
		label_text = label.get_text()
		result_text = result.get_text()
		label_x_offset = 30
		result_x_offset = label_x_offset+ 4  * len(label_text)
		y_offset = starting_location - percent_of_labels_done * span_of_labels_in_points
		label.set_position((label_x_offset, y_offset))
		result.set_position((result_x_offset, y_offset))
		percent_of_labels_done = percent_of_labels_done + (1 / len(annotator_mod_result_list_for_layout))


# skip space for the beginning of label , the length of the label, and how many digits the prime number is
#mod_result_xcoord = label_annotation.xyann[0] + .03 + .01 * len(str(self.primeNumber))
#mod_result = ax.annotate('0', xy=(mod_result_xcoord, annotation_row), xycoords=coords_to_use, animated=True)
#annotator_mod_result_list_for_layout.append((label_annotation, mod_result))


def setCountNumberDisplayed(currentNumber):
	countingString = '{:.1f}'.format(currentNumber)
	numberDisplay.set(text=countingString)
	backgroundNumberDisplay.set(text=countingString)


def create_background_image():
	global lines_to_draw_static
	global background_image
	for one_line in lines_to_draw_static:
		ax.draw_artist(one_line)
	ax.autoscale_view()
	fig.canvas.flush_events()
	plt.pause(.001)
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
		currentNumberNotDivisibleByAnyCycle = currentNumberNotDivisibleByAnyCycle and not oneCycle.is_number_multiple_of_base(
			currentNumber)
		should_prime_indicator_be_green = should_prime_indicator_be_green and (
					not oneCycle.is_number_multiple_of_base(int(currentNumber + .5)) or oneCycle.primeNumber == int(
				currentNumber))
	# if current number isn't divisible by any cycle, it's a prime
	if (currentNumber.is_integer() and currentNumberNotDivisibleByAnyCycle):
		found_a_prime = True
		currentNumber = int(currentNumber)
		last_prime_found = currentNumber
		# got a new prime
		newCycle = CircleCycleForOnePrime(plotCenter, currentNumber, ax)
		cycles.append(newCycle)
		newCycle.setIndicator(currentNumber)
		ax.autoscale_view()
	
	setCountNumberDisplayed(currentNumber)
	set_prime_indicator_green(should_prime_indicator_be_green)  # or (int(currentNumber) == last_prime_found))
	if (frameNumber % 100 == 0):
		print(frameNumber)
	
	return lines_to_draw_dynamic, currentNumberNotDivisibleByAnyCycle, found_a_prime


def animate(frameNumber, cycleContainer):
	animate_helper(frameNumber, cycleContainer)
	return lines_to_draw_dynamic


def handleOneFrame(frameNumber, cycleContainer):
	global background_image
	lines, numberDivisibleByCycle, found_prime = animate_helper(frameNumber, cycleContainer)
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

def layout_angle_ticks():
	ax.xaxis.set_major_locator(NullLocator())
	ax.xaxis.set_major_formatter(NullFormatter())
	

halfTheta = np.linspace(0, 2 * np.pi, FRAMES_PER_NUMBER)
theta = np.append(halfTheta, halfTheta + 2 * np.pi)
radius = np.full(theta.shape, 1)

#matplotlib.use('TkAgg')
matplotlib.use('QtAgg')
#matplotlib.use('WxAgg')


fig = plt.figure(figsize=(10.24, 7.68))
ax = fig.add_subplot(projection='polar')
if (NEW_RINGS_APPEAR_IN_CENTER):
	ax.invert_yaxis()

#explaination_text = "Each ring is for a prime number.   The indicator arc sweeps around to show the result of the current number mod by the prime. Zero is at the bottom. When no ring modulates the current number, the line at the bottom is green, indicating a prime candidate."
#fig.suptitle(explaination_text, y=.92, ha='center', va='bottom', wrap=True)
primeLocator = PrimeFixedLocator([2])  # FixedLocator([2])
radialLoc = RadialLocator(primeLocator)
radialLoc._axes = ax
ax.yaxis.set_major_locator(radialLoc)
ax.set_rlabel_position(157.5)
ax.set_theta_offset(3 * np.pi / 2)
layout_angle_ticks()

numberDisplay = ax.annotate('2', (.001, .5), xycoords='figure fraction',
                            animated=True)  #ax.text(np.pi, 1, '2', transform=ax.transData, animated=True)
backgroundNumberDisplay = ax.text(0.5,0.5, '2', transform=ax.transAxes, fontsize=100,ha='center', va='center', alpha=0.15)


offset_mod_label = OffsetFrom(numberDisplay, (0, 0))
ax.annotate('mod', (0, 13), xycoords=offset_mod_label)
ax.annotate('by', (0, -9), xycoords=offset_mod_label)
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
numberOfFrames = FRAMES_PER_NUMBER * MAX_NUMBER_TO_CHECK

if SHOW_LIVE_ANIMATION:
	for frameNumber in range(numberOfFrames):
		handleOneFrame(frameNumber, cycles)
		plt.pause(.001)

if WRITE_MOVIE_FILE:
	#MAX_NUMBER_TO_CHECK = 100
	numberOfFrames = FRAMES_PER_NUMBER * MAX_NUMBER_TO_CHECK
	anim = functools.partial(animate, cycleContainer=cycles)
	#startTime = time.time()
	fmpgWriter = FFMpegWriter(fps=60)
	ani = animation.FuncAnimation(fig, anim, numberOfFrames, repeat=False, blit=True, interval=1)
	filename='primeClock'+str(MAX_NUMBER_TO_CHECK)+'.gif'
	ani.save(filename=filename, writer=fmpgWriter)
#endTime = time.time()
#print("time taken ",(endTime-startTime))


# put this here so ani doesn't get garbage collected
#ani
