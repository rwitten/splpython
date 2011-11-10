import numpy
import scipy

def contains_descriptor(x, y, x_min, x_max, y_min, y_max):
	return (x >= x_min) and (y >= y_min) and (x < x_max) and (y < y_max)

def get_bboxes_containing_descriptor(x, y, h):
	bcd = numpy.zeros(5, int)
	bcd[0] = contains_descriptor(x, y, h.x_min, h.x_max, h.y_min, h.y_max)
	bcd[1] = contains_descriptor(x, y, h.x_min, float(h.x_min + h.x_max) / float(2.0), h.y_min, float(h.y_min + h.y_max) / float(2.0))
	bcd[2] = contains_descriptor(x, y, float(h.x_min + h.x_max) / float(2.0), h.x_max, h.y_min, float(h.y_min + h.y_max) / float(2.0))
	bcd[3] = contains_descriptor(x, y, h.x_min, float(h.x_min + h.x_max) / float(2.0), float(h.y_min + h.y_max) / float(2.0), h.y_max)
	bcd[4] = contains_descriptor(x, y, float(h.x_min + h.x_max) / float(2.0), h.x_max, float(h.y_min + h.y_max) / float(2.0), h.y_max)
	return bcd
