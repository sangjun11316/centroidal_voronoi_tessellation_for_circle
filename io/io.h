#ifndef IO_H
#define IO_H

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <algorithm>

#define IOPREC 10

void dump_points(float *ref,   int    ref_nb,
                 float *query, int    query_nb,
                 int dim, int iter);

void dump_area(int  *area,  int  ref_nb,
               int iter);

void dump_knn(float *knn_dist, int *knn_index,
              int query_nb, int dim, int k, int iter);

#endif
