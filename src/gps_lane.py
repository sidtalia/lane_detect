from pyroutelib3 import Router # Import the router
import numpy as np
from scipy.ndimage import gaussian_filter  # this needs to be installed.
import math as m
import matplotlib.pyplot as plt

def calcposNED(lat, lon, latReference, lonReference):
	earthRadius = 6378145.0
	lat /= 57.3
	lon /= 57.3
	latReference /= 57.3
	lonReference /= 57.3
	posNEDr = np.zeros(3)
	Y = earthRadius * (lat - latReference)
	X = earthRadius * np.cos(latReference) * (lon - lonReference)
	return X, Y


def find_route(start_lat,start_lon,end_lat,end_lon, home_lat, home_lon):
	router = Router("car") # Initialise it
	start = router.findNode(start_lat, start_lon) # Find start and end nodes
	end = router.findNode(end_lat, end_lon)

	status, route = router.doRoute(start, end) # Find the route - a list of OSM nodes
	print(status)
	if status == 'success':
		print("found route")
		routeLatLons = list(map(router.nodeLatLon, route)) # Get actual route coordinates
		# print(routeLatLons)
	Y = np.array([i[0] for i in routeLatLons])
	X = np.array([i[1] for i in routeLatLons])
	start_Y = Y[0]
	start_X = X[0]

	X, Y = calcposNED(np.copy(Y), np.copy(X),home_lat, home_lon)

	# find mid points between all consecutive points twice, filter(smoothen) and repeat 3 times
	for i in range(3):
		Y = np.dstack((Y[:-1],Y[:-1] + np.diff(Y)/2.0)).ravel()
		Y = np.dstack((Y[:-1],Y[:-1] + np.diff(Y)/2.0)).ravel()
		X = np.dstack((X[:-1],X[:-1] + np.diff(X)/2.0)).ravel()
		X = np.dstack((X[:-1],X[:-1] + np.diff(X)/2.0)).ravel()

		Y = gaussian_filter(Y,sigma=1)
		X = gaussian_filter(X,sigma=1)

	# print(X,Y)
	plt.scatter(X,Y)
	plt.axis('equal')
	plt.show()

	return X,Y


if __name__ == '__main__':
	# find_route(28.547376, 77.183635, 28.544931, 77.194203, 28.544728, 77.183394)  # from white castle intersection to reception
	find_route(28.545313, 77.180849, 28.545861, 77.179583, 28.544728, 77.183394)  # test road