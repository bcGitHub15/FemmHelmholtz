#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 16:40:32 2020

ScalarField

A class for manipulating regularly sampled field data such as
might be created by a finite element program. The data are first
represented by three rectangular arrays, x, y, and z, all with the
same dimensions. x and y contain the coordinates of the sampling
points and z the sampled values.

The class can display itself in various ways on 2-D and 3-D plots.
It allows for polynomial smoothing in one or two dimensions.
It supports differentiation in either direction without altering the
size of the array, though the edge values will not be as accurate as
those in the body of the array.
It would be nice also to support SOR smoothing of the interior of the
array.

In this version modifiers, such as Fitxxx and Diffxxx methods, return
a new ScalarField sharing the old x and y arrays bit with the newly
modified z array.
The tools to perform the individual operations hav been retained,
renamed to do<oldName> and now both take and return simple numpy
arrays. They are used to implement the external operations.

PolyDiff<i> methods have been improved to poly fit in BOTH directions,
but only differentiate in one. PolyFit smooths in both directions.

I have also define short aliases for the plotting functions and shortened
the class name to ScalarField.

This version pulls out the Grid base class, providing the sizes and
the coordinate arrays. It adds support for FEMME grids that are
extracted from FEMME models, not from files, and so have to know
their grid structure before reading the data.

@author: bcollett
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as p3
import femm

class Grid:
    #
    #   To build a grid you need to specify a rectangular
    #   region of space and to specify increments in the two
    #   directions.
    #   I have set up default values so that at minimum you
    #   need to specify a top right point. In that case you will
    #   get 101 points in each direction.
    #
    def __init__(self, xhigh, yhigh, xlow=None, ylow=None, dx=None, dy=None):
        if xlow==None:
            xlow=0.0
        if ylow==None:
            ylow=0.0
        if dx==None:
            dx=(xhigh-xlow)/100
        if dy==None:
            dy=(yhigh-ylow)/100
        x = np.arange(xlow, xhigh+dx, dx)
        y = np.arange(ylow, yhigh+dy, dy)
        self.BuildFromXY(x,y)
    
    def BuildFromXY(self, newx, newy):
        self.ncol = newx.size
        self.nrow = newy.size
        self.x = np.empty((self.nrow, self.ncol))
        for row in range(self.nrow):
            self.x[row,:]=newx
        self.y = np.empty((self.nrow, self.ncol))
        for col in range(self.ncol):
            self.y[:,col]=newy

    def PlotAsSurface(self, zvals, axes=None, stride=10):
        if axes == None:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')
        axes.plot_surface(self.x,self.y,zvals,
                          cmap=cm.coolwarm, linewidth = 0,
                          rstride=stride,cstride=stride)
        return axes

    def PlotAsWireframe(self, zvals, axes=None):
        if axes == None:
            fig = plt.figure()
            axes = fig.add_subplot(111)
        axes.plot_wireframe(self.x,self.y,zvals)
        return axes

    def ContourPlot(self, zvals, nline=10, axes=None):
        if axes == None:
            fig,ax = plt.subplots()
#            fig = plt.figure()
#            axes = fig.add_subplot(111, projection='3d')
        else:
            ax = axes
        ax.pcolormesh(self.x,self.y,zvals,cmap=cm.coolwarm, shading='auto')
        CS = ax.contour(self.x,self.y,zvals,levels=nline)
        ax.clabel(CS, inline=True, fontsize=10)
        return ax

    def PlotAsImage(self, zvals, axes=None):
        if axes == None:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')
        axes.imshow(zvals)
        return axes

    def doDiffx(self, src):
        dudx = np.zeros_like(src)
        dudx[:,0] = (src[:,1]-src[:,0])/(self.x[0,1]-self.x[0,0])
        dudx[:,-1] = (src[:,-1]-src[:,-2])/(self.x[0,-1]-self.x[0,-2])
        dudx[:,1:-2] = (src[:,2:-1]-src[:,0:-3])/(self.x[0,2:-1]-self.x[0,0:-3])
        return dudx

    def doDiffy(self,src):
        dudy = np.zeros_like(src)
        dudy[0,:] = (src[1,:]-src[0,:])/(self.y[1,0]-self.y[0,0])
        dudy[1:-2,:] = (src[2:-1,:]-src[0:-3,:])/(self.y[2:-1,:]-self.y[0:-3,:])
        dudy[-1,:] = (src[-1,:]-src[-2,:])/(self.y[-1,0]-self.y[-2,0])
        return dudy

    def doPolyFitx(self,src,degree=4):
        transp = np.transpose(src)
        params=np.polyfit(self.x[0,:], transp, degree)
        fitu = np.zeros_like(transp)
        for i in range(self.nrow):
            fitu[:,i] = np.polyval(params[:,i],self.x[0,:])
        ftransp = np.transpose(fitu)
        return ftransp

    def doPolyFity(self,src,degree=4):
        fitu = np.zeros_like(src)
        params=np.polyfit(self.y[:,0], src, degree)
        for i in range(self.ncol):
            fitu[:,i] = np.polyval(params[:,i],self.y[:,0])
        return fitu

    def doPolyDiffx(self,src,degree=4):
        transp = np.transpose(src)
        params=np.polyfit(self.x[0,:], transp, degree)
        fitu = np.zeros_like(transp)
        for i in range(self.nrow):
            fitu[:,i] = np.polyval(np.polyder(params[:,i]),self.x[0,:])
        ftransp = np.transpose(fitu)
        return ftransp

    def doPolyDiffy(self,src,degree=4):
        params=np.polyfit(self.y[:,0], src, degree)
        fitu = np.zeros_like(src)
        for i in range(self.ncol):
            fitu[:,i] = np.polyval(np.polyder(params[:,i]),self.y[:,0])
        self.u = fitu
        return fitu

#
#   The scalar field adds a single field value which it treats as
#   a scalar field and calls it u.
#   ScalarFields provide a form of 2-step creation. While you can create
#   ScalarFields from ScalarFields with methods such as Diffx you can
#   also create them de-novo and fill the data in later, either using
#   one of the load methods or directly modifying ScalarField.z yourself.
#
class ScalarField(Grid):
    def __init__(self,srcField=None):
        if isinstance(srcField, Grid):
            self.nrow = srcField.nrow
            self.ncol = srcField.ncol
            self.x = srcField.x
            self.y = srcField.y
            if isinstance(srcField, ScalarField):
                self.u = srcField.u.copy()
            else:
                self.u = np.zeros_like(self.x)
        else:
            self.nrow = 1
            self.ncol = 1
            self.x = np.zeros((1,1))
            self.y = np.zeros((1,1))
            self.u = np.zeros((1,1))

    def Install(self, newu):
        if np.shape(newu) == np.shape(self.u):
            self.u = newu
            return True
        return False

    def LoadComsol(self, name, uindex=3):
        s00 = np.loadtxt(name,skiprows=9)
        self.nrow = 201
        self.ncol = 101
        if s00[:,1].size != self.nrow * self.ncol:
            return False
        newShape = (self.nrow, self.ncol)
        self.x = np.reshape(s00[:,1], newShape)
        self.y = np.reshape(s00[:,2], newShape)
        self.u = np.reshape(s00[:,uindex], newShape)
        return True

    #
    #   Loads z from x or y component of an _open_ FEMME model 
    #   rather than a file.
    #   If you want to use this then you should set up the
    #   underlying Grid before calling LoadFemme.
    #   AAAGH, his FEMM system does not _have_ a model. There is
    #   just a global instance of femm so I have to go that way.
    #
#    def LoadFemmx(self, model):
    def LoadFemmx(self):
        for row in range(self.nrow):
            for col in range(self.ncol):
#                bx,by = model.mo_getb(self.x[row, col], self.y[row,col])
                bx,by = femm.mo_getb(self.x[row, col], self.y[row,col])
                self.u[row,col] = bx
        return True

    def LoadFemmy(self):
        for row in range(self.nrow):
            for col in range(self.ncol):
                bx,by = femm.mo_getb(self.x[row, col], self.y[row,col])
                self.u[row,col] = by
        return True

    def Splot(self, strpass=10, axpass=None):
        return self.PlotAsSurface(self.u, axpass, strpass)

    def Wplot(self, axpass=None):
        return self.PlotAsWireFrame(axpass)

    def Cplot(self, npass=10, axpass=None):
        return self.ContourPlot(self.u, npass)

    def Iplot(self, figpass=None):
        return self.PlotAsImage(figpass)

    def Diffx(self):
        newf = ScalarField(self)
        newf.u = self.doDiffx(self.u)
        return newf

    def Diffy(self):
        newf = ScalarField(self)
        newf.u = self.doDiffy(self.origz)
        return newf

    def PolyFitx(self,degpass=4):
        newf = ScalarField(self)
        newf.u = self.doPolyFitx(self.u, degpass)
        return newf

    def PolyFity(self,degpass=4):
        newf = ScalarField(self)
        newf.u = self.doPolyFity(self.u, degpass)
        return newf
       
    def PolyFit(self,degpass=4):
        newf = ScalarField(self)
        temp = self.doPolyFity(self.u, degpass)
        newf.u = self.doPolyFitx(temp, degpass)
        return newf
       
    def PolyDiffx(self,degpass=4):
        newf = ScalarField(self)
        temp = self.doPolyFity(self.u, degpass)
        newf.u = self.doPolyDiffx(temp, degpass)
        return newf

    def PolyDiffy(self,degpass=4):
        newf = ScalarField(self)
        temp = self.doPolyFitx(self.u, degpass)
        newf.u = self.doPolyDiffy(temp, degpass)
        return newf

#
#   The vector field adds a second component which it names v
#   and it treats the two as the x and y components of a 2-D vector
#   field with u as the x component and v as the y.
#
class VectorField(ScalarField):
    def __init__(self,srcField=None):
        if isinstance(srcField, Grid):
            self.nrow = srcField.nrow
            self.ncol = srcField.ncol
            self.x = srcField.x
            self.y = srcField.y
            if isinstance(srcField, ScalarField):
                self.u = srcField.u.copy()
            else:
                self.u = np.zeros_like(self.x)
            if isinstance(srcField, VectorField):
                self.v = srcField.v.copy()
            else:
                self.v = np.zeros_like(self.x)
        else:
            self.nrow = 1
            self.ncol = 1
            self.x = np.zeros((1,1))
            self.y = np.zeros((1,1))
            self.u = np.zeros((1,1))
            self.v = np.zeros((1,1))
    #
    #   Read z from B components of an _open_ FEMME model 
    #   rather than a file.
    #   If you want to use this then you should set up the
    #   underlying Grid before calling LoadFemme.
    #   AAAGH, his FEMM system does not _have_ a model. There is
    #   just a global instance of femm so I have to go that way.
    #
    def ReadFemm(self):
        for row in range(self.nrow):
            for col in range(self.ncol):
                bx,by = femm.mo_getb(self.x[row, col], self.y[row,col])
                self.u[row,col] = bx
                self.v[row,col] = by
            print(',',end='')
        return True

    def VPlot(self, stride=1, axes=None):
        if axes == None:
            fig = plt.figure()
            axes = fig.add_subplot(111)
        axes.quiver(self.x[0::stride,0::stride], self.y[0::stride,0::stride],
                    self.u[0::stride,0::stride],self.v[0::stride,0::stride])

    def StreamPlot(self, stride=1, axes=None):
        if axes == None:
            fig,ax = plt.subplots()
#            fig = plt.figure()
#            axes = fig.add_subplot(111, projection='3d')
        else:
            ax = axes
        CS = ax.streamplot(self.x,self.y,self.u, self.v)
#        ax.clabel(CS, inline=True, fontsize=10)
        return ax

    def Norm2(self):
        newf = ScalarField(self)
        newf.u = self.u * self.u + self.v * self.v
        return newf        

    def Mag(self):
        newf = ScalarField(self)
        newf.u = np.sqrt(self.u * self.u + self.v * self.v)
        return newf
    
    def WriteOn(self, fname):
        dlen = self.x.size
        oneArray = np.empty((dlen, 4))
        oneArray[:,0] = self.x.reshape(dlen)
        oneArray[:,1] = self.y.reshape(dlen)
        oneArray[:,2] = self.u.reshape(dlen)
        oneArray[:,3] = self.v.reshape(dlen)
        np.savetxt(fname, oneArray, header=f'{self.nrow},{self.ncol}')
    
    def LoadFemmFile(self, fname):
        data = np.loadtxt(fname)
        self.x = data[:,0].reshape(self.nrow, self.ncol)
        self.y = data[:,1].reshape(self.nrow, self.ncol)
        self.u = data[:,2].reshape(self.nrow, self.ncol)
        self.v = data[:,3].reshape(self.nrow, self.ncol)
        
