#ifndef FILTERS_H
#define FILTERS_H

#include "images.h"


void compute_histogram(const Image *src, int hist[256]);
int otsu_treshold(const Image *src);
Image *treshold(const Image *src, int t);


#endif