import math

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from matplotlib.projections.polar import PolarTransform

global theta
global radius

def convertRadiansToArrayIndex(radianAngle):
	return math.ceil(radianAngle * 100) % theta.size


def drawModRingForNumber(numberForRing):
	ZERO_ANGLE = 3 * math.pi / 2
	unitSlice = 2 * math.pi / numberForRing
	zeroRegionBeginIndex = convertRadiansToArrayIndex(ZERO_ANGLE - (unitSlice / 2))
	zeroRegionEndIndex = convertRadiansToArrayIndex(ZERO_ANGLE + (unitSlice / 2))
	safeRegionEndIndex = convertRadiansToArrayIndex(2*np.pi+ZERO_ANGLE - (unitSlice / 2))
	
	zeroRegionArc = Line2D(theta[zeroRegionBeginIndex:zeroRegionEndIndex],
	                       numberForRing * radius[zeroRegionBeginIndex:zeroRegionEndIndex], linewidth=5,
	                       color='red')
	
	nonZeroRegionArc = Line2D(theta[zeroRegionEndIndex:safeRegionEndIndex],
	                          numberForRing * radius[zeroRegionEndIndex:safeRegionEndIndex], linewidth=5,
	                          color='green')
	indicatorArc = Line2D(theta[zeroRegionBeginIndex:zeroRegionEndIndex],
	                       numberForRing * radius[zeroRegionBeginIndex:zeroRegionEndIndex], linewidth=8,
	                       color='blue')
	
#	print(theta[zeroRegionEndIndex:safeRegionEndIndex])
	ax.add_line(nonZeroRegionArc)
	ax.add_line(zeroRegionArc)
	ax.add_line(indicatorArc)
	return indicatorArc

def rotateLine(lineToRotate, angle) :
#	trans1 = Affine2D().rotate_around(0, 0, angle)
	lineToRotate.set_xdata(lineToRotate.get_xdata()+angle)


theta = np.arange(0 , 4*np.pi, .01)
radius = np.full(theta.shape, 1)
fig, ax = plt.subplots(subplot_kw={'polar': 'True'})




print('trans data  ',ax.transData, '  \n also  ', ax.get_transform())
#ax.add_patch(rectLine)
#ax.add_line(linePol)
indic5 = drawModRingForNumber(5)
indic11 = drawModRingForNumber(11)
ax.autoscale_view()
plt.show(block=False)
plt.pause(2)
rotateLine(indic11,5*np.pi/8)
plt.pause(5)
