#!/usr/bin/env python

from __future__ import with_statement

import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib import ticker, cm, colors
import numpy as np
import sys
import os
import glob
import re

import argparse

import vtk
from vtk.util import numpy_support

def main():
    parser = argparse.ArgumentParser(description = 'postproc to Centroidal Voronoi Tessellation CUDA implementation')
    parser.add_argument('--itarget', type=int, nargs=1, required=False, default=0,  help='target point to watch')

    itarget = parser.parse_args().itarget

    file_ref   = "points_ref.dat"
    file_area  = "area_ref.dat"
    file_query = "points_query.dat"
    file_knn   = "knn_query.dat"

    ########################
    #       READ DATA      #
    ########################
    #--- read ref's
    try:
        with open(file_ref, mode='rt', encoding='utf-8') as fid:
            data_ref = fid.read()
    except EnvironmentError:
        print("Failed to open {}".format(file_ref))

    data_ref = data_ref.split('\n')

    n_ref   = int(data_ref[0])
    dim_ref = int(data_ref[1])

    x_ref   = np.zeros((n_ref, dim_ref))
    for ipt in range(n_ref):
        line = data_ref[2+ipt].split()
        for d in range(dim_ref):
            x_ref[ipt,d] = float(line[d])

    #--- read area
    try:
        with open(file_area, mode='rt', encoding='utf-8') as fid:
            data_area = fid.read()
    except EnvironmentError:
        print("Failed to open {}".format(file_area))

    data_area = data_area.split('\n')

    area_ref  = np.zeros(n_ref)
    for ipt in range(n_ref):
        line = data_area[ipt].split()
        area_ref[ipt] = float(line[0])

    #--- read query's
    try:
        with open(file_query, mode='rt', encoding='utf-8') as fid:
            data_query = fid.read()
    except EnvironmentError:
        print("Failed to open {}".format(file_query))

    data_query = data_query.split('\n')

    n_query   = int(data_query[0])
    dim_query = int(data_query[1])

    # check dimension consistency
    if dim_ref != dim_query:
        raise ValueError("dim_ref != dim_qeury")

    dim = dim_ref # redundant, but just to consistent with the main code

    x_query   = np.zeros((n_query, dim_query))
    for ipt in range(n_query):
        line = data_query[2+ipt].split()
        for d in range(dim_query):
            x_query[ipt,d] = float(line[d])

    #--- read knn data (of query's)
    try:
        with open(file_knn, mode='rt', encoding='utf-8') as fid:
            data_knn = fid.read()
    except EnvironmentError:
        print("Failed to open {}".format(file_knn))

    data_knn = data_knn.split('\n')

    if int(data_knn[0]) != n_query:
        raise ValueError("n_knn != n_query")

    if int(data_knn[1]) != dim:
        raise ValueError("dim_knn != dim")

    knn_k = int(data_knn[2])

    knn_index = np.zeros(n_query, dtype=np.int32)
    knn_dist  = np.zeros(n_query)

    for ipt in range(n_query):
        line = data_knn[3+2*ipt  ].split()
        knn_index[ipt] = int(line[0])

        line = data_knn[3+2*ipt+1].split()
        knn_dist[ipt] = float(line[0])

    #--- make area_query # this is not necessary
    area_query  = np.zeros(n_query)
    for ipt in range(n_query):
        area_query[ipt] = area_ref[knn_index[ipt]]

    ########################
    #      WRITE VTK       #
    ########################
    # refs
    Points = vtk.vtkPoints()
    Vertices = vtk.vtkCellArray()
    for i in range(n_ref):
        _id = Points.InsertNextPoint(x_ref[i,0], x_ref[i,1], 0.)
        Vertices.InsertNextCell(1)
        Vertices.InsertCellPoint(_id)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(Points)
    polyData.SetVerts(Vertices)

    pointData = polyData.GetPointData()
    area_ref_vtk = numpy_support.numpy_to_vtk(area_ref)
    area_ref_vtk.SetName("area_ref")
    pointData.AddArray(area_ref_vtk)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName("vtk_ref.vtk")
    writer.SetInputData(polyData)
    writer.Write()

    # query's
    Points = vtk.vtkPoints()
    Vertices = vtk.vtkCellArray()
    for i in range(n_query):
        _id = Points.InsertNextPoint(x_query[i,0], x_query[i,1], 0.)
        Vertices.InsertNextCell(1)
        Vertices.InsertCellPoint(_id)

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(Points)
    polyData.SetVerts(Vertices)

    pointData = polyData.GetPointData()
    area_ref_vtk = numpy_support.numpy_to_vtk(area_query)
    area_ref_vtk.SetName("area_query")
    pointData.AddArray(area_ref_vtk)
    knn_index_vtk = numpy_support.numpy_to_vtk(knn_index)
    knn_index_vtk.SetName("knn_index")
    pointData.AddArray(knn_index_vtk)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName("vtk_query.vtk")
    writer.SetInputData(polyData)
    writer.Write()

    ########################
    #         PLOT         #
    ########################
    '''
    #fig, ax = plt.subplots(figsize = (15,13))
    fig, ax = plt.subplots(figsize = (13,13))
    #ax.scatter(x_query[:,0], x_query[:,1], c=knn_index, s=1)
    #ax.scatter(x_query[:,0], x_query[:,1], c=area_query, s=1)
    im = ax.scatter(x_query[:,0], x_query[:,1], c=area_ref[knn_index], s=1)
    fig.colorbar(im)
    ax.scatter(x_ref[:,0], x_ref[:,1], c='k', s=10)

    #itarget = 0
    #ax.scatter(x_query[itarget,0], x_query[itarget,1], s=100, fc='none', ec='r')
    #for ik in range(knn_k):
    #    index = knn_index[itarget]
    #    print("index {}".format(index))
    #    ax.scatter(x_ref[index,0], x_ref[index,1], s=100, fc='none', ec='g')

    #ax.set_xlim([-0.03, -0.01])
    #ax.set_ylim([-0.47, -0.45])

    plt.show()
    '''

if __name__=='__main__':
    main()
