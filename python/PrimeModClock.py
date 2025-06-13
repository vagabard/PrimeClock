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

import matplotlib.pyplot as plt
import numpy as np
import math
import functools
import matplotlib.animation as animation
import matplotlib.text as matText
import math

from matplotlib.ticker import StrMethodFormatter as StrMethodFormatter
from matplotlib.collections import PatchCollection
from matplotlib.patches import Arc as patchArc
from matplotlib.ticker import FixedLocator as FixedLocator
from matplotlib.animation import FFMpegWriter as FFMpegWriter
from sympy import sieve, primepi

#%matplotlib auto
# plt.rcParams["animation.html"] = "jshtml"
plt.rcParams["animation.html"] = "html5"

plt.rcParams["figure.subplot.left"]=   0  # the left side of the subplots of the figure
plt.rcParams["figure.subplot.right"]=  1    # the right side of the subplots of the figure
plt.rcParams["figure.subplot.bottom"]= 0   # the bottom of the subplots of the figure
plt.rcParams["figure.subplot.top"]=    1   # the top of the subplots of the figure



FRAMES_PER_NUMBER = 40
MAX_NUMBER_TO_CHECK = 50
plotCenter = (0, 0)


class PrimeFixedLocator(FixedLocator):
    primeTicks = []
    oldVmax = 0

    def __init__(self):
        self.locs = []

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


class CircleCycleForOnePrime:
    primeNumber = 0
    center = (0, 0)

    logScaling = False
    radius = 0
    currentIndicatorValue = 0
    RING_WIDTH = 2
    INDICATOR_WIDTH = RING_WIDTH * 2
    ANGLE_TO_INDICATE_ZERO = 270.0
    zeroWedge = None
    nonZeroWedge = None
    indicatorWedge = None
    patches = []
    modIndicatorZero = False
    unitSlice = None
    axesForCycle = None

    def __init__(self, center, primeNumber, axes, logScaling):
        self.center = center
        self.primeNumber = int(primeNumber)
        self.setLogScaling(logScaling)
        self.axesForCycle = axes
        self.constructPrimeModCirclePatches()
        self.setIndicator(0)

    def setLogScaling(self, useLogScaling, minimumRadius=0):
        self.logScaling = useLogScaling
        self.radius = 2 * self.primeNumber
        if (self.logScaling):
            self.radius = math.log(self.primeNumber)
        if (self.radius < minimumRadius):
            self.radius = minimumRadius + 2

    def constructPrimeModCirclePatches(self):
        # divide a circle into primeNumber slices
        self.unitSlice = 360.0 / self.primeNumber
        #		zeroRegionBegin= -(self.unitSlice/2)
        zeroRegionBegin = 0
        zeroRegionEnd = self.unitSlice

        self.zeroWedge = patchArc(self.center, self.radius, self.radius, theta1=zeroRegionBegin, theta2=zeroRegionEnd,
                                  linewidth=CircleCycleForOnePrime.RING_WIDTH,
                                  angle=CircleCycleForOnePrime.ANGLE_TO_INDICATE_ZERO - (self.unitSlice / 2),
                                  color='red')
        self.axesForCycle.add_patch(self.zeroWedge)

        self.nonZeroWedge = patchArc(self.center, self.radius, self.radius, theta1=zeroRegionEnd,
                                     theta2=zeroRegionBegin, linewidth=CircleCycleForOnePrime.RING_WIDTH,
                                     angle=CircleCycleForOnePrime.ANGLE_TO_INDICATE_ZERO - (self.unitSlice / 2),
                                     color='green')
        self.axesForCycle.add_patch(self.nonZeroWedge)

        self.indicatorWedge = patchArc(self.center, self.radius, self.radius, theta1=-self.unitSlice / 5,
                                       theta2=self.unitSlice / 5, linewidth=CircleCycleForOnePrime.INDICATOR_WIDTH,
                                       angle=CircleCycleForOnePrime.ANGLE_TO_INDICATE_ZERO - (self.unitSlice / 2),
                                       color='blue')
        self.axesForCycle.add_patch(self.indicatorWedge)
        patches = [self.zeroWedge, self.nonZeroWedge, self.indicatorWedge]

    def setIndicator(self, newIndicatorValue=None):
        if (newIndicatorValue != None):
            # convert the remainder to a percent
            self.currentIndicatorValue = (newIndicatorValue % self.primeNumber)
            self.modIndicatorZero = (self.currentIndicatorValue == 0)
        translatedFromValueToAngle = CircleCycleForOnePrime.ANGLE_TO_INDICATE_ZERO - (
                    self.unitSlice / 2) + self.unitSlice * self.currentIndicatorValue
        self.indicatorWedge.set_angle(translatedFromValueToAngle)


def setCountNumberDisplayed(currentNumber):
    countingString = '{:2g}'.format(currentNumber)
    numberDisplay.set_text(countingString)


def animate(frameNumber, cycleContainer):
    #	if (frameToPauseUntil <= frameNumber):
    currentNumber = startingValue + frameNumber / FRAMES_PER_NUMBER
    # only checxk for a new cycle if this is an integer
    currentNumberIsIntegerAndNotDivisibleByAnyCycle = (currentNumber.is_integer())

    patchlist = []
    for oneCycle in cycleContainer:
        oneCycle.setIndicator(currentNumber)
        # check if current number is divisible by cycle prime
        currentNumberIsIntegerAndNotDivisibleByAnyCycle = currentNumberIsIntegerAndNotDivisibleByAnyCycle and not oneCycle.modIndicatorZero
        patchlist.extend(oneCycle.patches)
        ax.autoscale_view()
    # if current number isn't divisible by any cycle, it's a prime
    if (currentNumberIsIntegerAndNotDivisibleByAnyCycle):
        computedLabelSize = 15
        if (math.log(primepi(currentNumber)) > 1):
            computedLabelSize = 5 + (math.log(primepi(currentNumber)) + 10) / math.log(primepi(currentNumber))
        ax.tick_params(labelsize=computedLabelSize)
        # got a new prime
        newCycle = CircleCycleForOnePrime(plotCenter, currentNumber, ax, False)
        cycles.append(newCycle)
        patchlist.extend(newCycle.patches)
        fig.canvas.flush_events()
        frameToPauseUntil = frameNumber + FRAMES_PER_NUMBER

    setCountNumberDisplayed(currentNumber)
    return PatchCollection(patchlist), numberDisplay


# def getTextLocation():
#	textCoorinates = (ax.get_xlim()[0], ax.get_ylim()[1])
#	print(textCoorinates, ' ' , ax.get_xlim(), ' ' , ax.get_ylim())
#	return textCoorinates

plt.autoscale(enable=True, axis='both')
fig = plt.figure()
ax = plt.subplot()
#ax.set_xmargin(0)
#ax.set_ymargin(0)
ax.tick_params(direction='in', pad=10, labelsize=15)

ax.tick_params(axis='x', pad=15, labelsize=10)

ax.xaxis.set_major_formatter(StrMethodFormatter("   {x}"))
xSpineLeft = ax.spines['bottom']
xSpineLeft.set_position('zero')
xSpineRight = ax.spines['top']
xSpineRight.set_position('zero')
ySpineTop = ax.spines['left']
ySpineTop.set_position('zero')
ySpineBottom = ax.spines['right']
ySpineBottom.set_position('zero')

ax.xaxis.set_major_locator(PrimeFixedLocator())

ax.yaxis.set_major_locator(PrimeFixedLocator())

# Build a rectangle in axes coords
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

numberDisplay = ax.text(0, 1, '2', fontsize='large', transform=ax.transAxes)
ax.add_artist(numberDisplay)

cycles = []
twoCycle = CircleCycleForOnePrime(plotCenter, 2, ax, False)
cycles.append(twoCycle)

ax.autoscale_view()
startingValue = 2

anim = functools.partial(animate, cycleContainer=cycles)
MAX_NUMBER_TO_CHECK = 50
numberOfFrames = FRAMES_PER_NUMBER * MAX_NUMBER_TO_CHECK
ani = animation.FuncAnimation(fig, anim, numberOfFrames, save_count=numberOfFrames, repeat=False)
ani.save(filename='primeClock50.mp4', writer=FFMpegWriter(fps=20))

MAX_NUMBER_TO_CHECK = 100
numberOfFrames = FRAMES_PER_NUMBER * MAX_NUMBER_TO_CHECK
ani = animation.FuncAnimation(fig, anim, numberOfFrames, save_count=numberOfFrames, repeat=False)
# ani.save(filename='primeClock100.mp4', fps='20')

MAX_NUMBER_TO_CHECK = 250
numberOfFrames = FRAMES_PER_NUMBER * MAX_NUMBER_TO_CHECK
ani = animation.FuncAnimation(fig, anim, numberOfFrames, save_count=numberOfFrames, repeat=False)
# ani.save(filename='primeClock250.mp4', fps='20')

MAX_NUMBER_TO_CHECK = 500
numberOfFrames = FRAMES_PER_NUMBER * MAX_NUMBER_TO_CHECK
ani = animation.FuncAnimation(fig, anim, numberOfFrames, save_count=numberOfFrames, repeat=False)
# ani.save(filename='primeClock500.mp4', fps='20')

# plt.show()
# put this here so ani doesn't get garbage collected
ani

