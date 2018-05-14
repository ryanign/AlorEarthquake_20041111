#!/usr/bin/env python

import os
import sys
import math as mt
from mpl_toolkits.basemap import Basemap as Basemap
#from shapely.geometry import Polygon
import matplotlib as mpl
#from matplotlib.patches import Polygon
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.colors import rgb2hex
#from shapely.geometry import Point
from pyproj import Geod
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d
from matplotlib import colors,colorbar
from obspy.geodetics.base import gps2dist_azimuth
from obspy.geodetics import locations2degrees
from obspy.core import Trace, Stream
from scipy.optimize import nnls
from scipy import interpolate
import pickle
#from netCDF4 import Dataset as NetCDFFile
from matplotlib import colors,colorbar
from matplotlib.colors import rgb2hex
# Probably want to switch all points/polys to this (but they're not 3D)
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point as ShapelyPoint
from scipy.interpolate import UnivariateSpline



ffipy_dir = os.environ['HOME'] + '/RSES-u5484353/Academics/PhD_research/02_HistoricalTsunamis/1992_Flores/new_work_20170919/inversion/seismic/philscode'
if not ffipy_dir in sys.path:
    sys.path.append(ffipy_dir)
from ffi import kiksyn,read_velmod,raypgeom,delay
import ffi
from dcs_python import coseismic_matrix as _coseismic_matrix

def Poly_new(verts):
    codes = [Path.MOVETO]
    for vert in verts[1:-1]:
        codes.append(Path.LINETO)
    codes.append(Path.CLOSEPOLY)
    return Path(verts, codes)

class Point(object):
    def __init__(self,x,y,z=0.):
        self.x = x
        self.y = y
        self.z = z
        
    def __str__(self):
        return 'x(longitude)=%8.3f, y(latitude)=%8.3f, z(depth)=%7.2f' % \
                (self.x,self.y,self.z)
    def copy(self):
        return Point(self.x,self.y,self.z)
        
def rectangular_polygon(strike,dip,length,width,center=None,origin=None,\
            Verbose=False,basemap=None):
    if center == None and origin == None:
        raise Exception, 'Must specify either origin or center: none supplied'
    if center != None and origin != None:
        raise Exception, 'Must specify one of origin or center: both supplied'

    g = Geod(ellps='WGS84')
    dw = width*mt.cos(mt.radians(dip))
    dz = mt.sin(mt.radians(dip))*width

    if center != None:
        # Now that center is defined, calculate Polygon, starting at lower left corner
        # and proceeding clockwise
        (tmpx,tmpy,tmpbaz) = g.fwd(center.x, center.y,  strike+90., 500.*dw)
        (tmpx,tmpy,tmpbaz) = g.fwd(  tmpx,   tmpy, strike-180.,     500.*length)
        llcrnr = Point(tmpx,tmpy,center.z+0.5*dz)
    else:
        llcrnr = origin
        
    if Verbose:
        print 'lower left corner at (%8.3f,%7.3f)' % (tmpx,tmpy)         
    (tmpx,tmpy,bazm) = g.fwd(llcrnr.x,llcrnr.y,strike,1000.*length)
    lrcrnr = Point(tmpx,tmpy,llcrnr.z)
    if Verbose:
        print 'lower right corner at (%8.3f,%7.3f)' % (tmpx,tmpy)         
    (tmpx,tmpy,tmpbaz) = g.fwd(lrcrnr.x,lrcrnr.y,strike-90.,1000.*dw)
    urcrnr = Point(tmpx,tmpy,lrcrnr.z-dz)
    if Verbose:
        print 'upper right corner at (%8.3f,%7.3f)' % (tmpx,tmpy)         
    (tmpx,tmpy,tmpbaz) = g.fwd(urcrnr.x,urcrnr.y,bazm,1000.*length)
    ulcrnr = Point(tmpx,tmpy,urcrnr.z)
    if Verbose:
        print 'upper left corner at (%8.3f,%7.3f)' % (tmpx,tmpy)         
    #print '1 ',ulcrnr[0],ulcrnr[1]
    poly = []
    for xyz in [llcrnr,lrcrnr,urcrnr,ulcrnr,llcrnr]:
        if basemap != None:
            poly.append(basemap(xyz.x,xyz.y))
        else:
            poly.append(xyz)           
    return poly

class Fault():
    '''
    An earthquake fault class. 
    A Fault object can be initalized with geometry specified via the fault_spec
    array, or as a list of subfaults whose geometry has been set elsewhere.
    Input:
       fault_spec   A numpy array with elements set accoring to:
          fault_spec[0]  = hypocenter (lon,lat,dep) as Point object
          fault_spec[1]  = fault strike (deg)
          fault_spec[2]  = fault dip (deg)
          fault_spec[3]  = dip != 0.: h_top, depth to top of fault (km)
                               == 0.: interpreted as indicating hypocenter at:
                                   -1.,fault edge in strike+90. direction,
                                    0., center, 
                                    1., fault edge in strike-90. direction
          fault_spec[4]  = xdrct (-1.<->1.);   Rupture directivity: 
                                                 -1., unilateral along-strike,                                               
                                                  0., bilateral                                  
                                                  1., unilateral anti-strike                                                                        
          fault_spec[5]  = Faul length (km)
          fault_spec[6]  = Fault width (km)
       subfaults    A list of subfault structures
       velmod       A velocity model in the format used by kiksyn, which can 
                    be read from a file using read_velmod 
                    
       Initialization sets the whole earth raypath geometry appropriate for 
       the hypocenter depth. 
              
    '''

    def __init__(self, fault_spec=None, subfaults=None, velmod=None, Verbose=False):
        # Instantiate a Geod object for coordinate projections
        self.g = Geod(ellps='WGS84')
        if fault_spec != None:
            if len(fault_spec) != 7:
                raise Exception, 'Fault_spec array must have lenght 7: hypo,strike,dip,h_top,xdrct,length,width'
            self.hypo = fault_spec[0]
            (self.strike,self.dip,self.h_top,self.xdrct,self.length,self.width) =  fault_spec[1:] 
            self.h_bottom = self.h_top+self.width*mt.sin(mt.radians(self.dip))
            # Sanity check that hypocenter depth above bottom & below top of fault
            if self.dip != 0. and (self.hypo.z > self.h_bottom or self.hypo.z < self.h_top):
                raise Exception, 'Hypocenter depth (%g) is outside fault depth range (%g-%g)' % (self.hypo.z,self.h_top,self.h_bottom)
            print     self.hypo.x,self.hypo.y,self.hypo.z,self.h_top,self.strike,self.dip,self.xdrct
            ## Calculate hyocentre in fault coordinates
            # Move from hypocenter lat,lon to base of fault
            if self.dip == 90.:
                dist = 0.
            if self.dip == 0.: # Horizontal fault 
                self.hypo_y = 0.5*self.width*(1.+self.h_top)
                dist = self.hypo_y
            else:
                self.hypo_y = (self.h_bottom - self.hypo.z)/mt.sin(mt.radians(self.dip))
                dist = (self.h_bottom - self.hypo.z)/mt.tan(mt.radians(self.dip))
            # Calculate lon,lat of hypocenter projectred to base of fault
            lon,lat,azm = self.g.fwd(self.hypo.x,self.hypo.y,self.strike+90.,1000.*dist)
            print lon,lat,self.hypo.x,self.hypo.y
            # Move back along strike to fault origin
            dist = -0.5*(self.xdrct+1.0)*self.length
            self.hypo_x = -dist
            lon,lat,azm = self.g.fwd(lon,lat,self.strike,1000.*dist)
            self.origin = Point(lon,lat,self.h_bottom)
            self.shape = 'Rectangular'
            print 'Fault origin at %g,%g,%g' % (self.origin.x,self.origin.y,self.origin.z)
            # Set body wave raypath geometries for Kikuchi reflectivity 
            # (assumes 1066A)
            # Set geometrical spreading and slowness arrays for 
            ir = 1
            if self.hypo.z > 33.: ir = 0
            self.raypaths = {}
            gp,ps,v0 = raypgeom(self.hypo.z,ir,1) 
            self.raypaths[ 'P'] = (gp,ps,v0)
            gp,ps,v0 = raypgeom(self.hypo.z,ir,2) 
            self.raypaths['SH'] = (gp,ps,v0)
            self.subfaults = []
        elif subfaults != None:
            self.shape = None
            self.subfaults = subfaults
        else:          
            raise Exception, 'Fault class requires list of subfaults or fault_spec array'
        # Set velmod 
        self.velmod  = velmod
        self.ruptime = None
        self.vrmax   = None
        self.ntw    = None
        
    def copy(self):
        if self.shape != 'Rectangular':
            raise ValueError,'Can only handle faults with shape = Rectangular'
        fault = Fault([self.hypo,self.strike,self.dip,self.h_top,self.xdrct,
                    self.length,self.width])
        fault.g                 = Geod(ellps='WGS84')
        fault.h_bottom          = self.h_bottom   
        fault.sbf_len           = self.sbf_len  
        fault.sbf_wid           = self.sbf_wid  
        fault.hypo_x            = self.hypo_x 
        fault.hypo_y            = self.hypo_y
        if self.Tessellation  == 'Rectangular':
            fault.sbf_nx        = self.sbf_nx
            fault.sbf_ny        = self.sbf_ny
            fault.Tessellation  = 'Rectangular'
            if hasattr(self,'Ordering'):
                fault.Ordering  = self.Ordering
        fault.ntw               = self.ntw
        fault.origin            = self.origin.copy()
        fault.raypaths          = self.raypaths.copy()
        if self.ruptime != None:
            fault.ruptime       = self.ruptime.copy()
        fault.shape             = self.shape
        if self.velmod != None:            
            fault.velmod            = self.velmod.copy()
        fault.vrmax             = self.vrmax
        fault.subfaults = [] 
        for sbf in self.subfaults:
            fault.subfaults.append(sbf.copy())
        return fault
     
    def geographic_limits(self,xpnd=0.5):
        lnmn = 400.; lnmx =  -400.; ltmn = 100.; ltmx =  -100.
        if self.shape == 'Rectangular':
            llcrnr = self.origin
            lon,lat,baz = self.g.fwd(self.origin.x,self.origin.y,self.strike,1000.*self.length)
            lrcrnr = Point(lon,lat,self.origin.z)
            lon,lat,baz = self.g.fwd(self.origin.x,self.origin.y,self.strike-90.,1000.*self.width*mt.cos(mt.radians(self.dip)))
            ulcrnr = Point(lon,lat,self.h_top)
            lon,lat,baz = self.g.fwd(ulcrnr.x,ulcrnr.y,self.strike,1000.*self.length)
            urcrnr = Point(lon,lat,self.h_top)
            for point in (llcrnr,lrcrnr,ulcrnr,urcrnr):
                lnmn = min(lnmn,point.x)
                lnmx = max(lnmx,point.x)
                ltmn = min(ltmn,point.y)
                ltmx = max(ltmx,point.y)
        elif self.subfaults != None:
            for sbf in self.subfaults:
                for point in sbf.polygon:
                    lnmn = min(lnmn,point.x)
                    lnmx = max(lnmx,point.x)
                    ltmn = min(ltmn,point.y)
                    ltmx = max(ltmx,point.y)
        else:
            raise Exception, "Fault must have shape='Rectangular' or subfaults != None"
        lllon = 0.5*(lnmn+lnmx)-xpnd*(lnmx-lnmn)
        lllat = 0.5*(ltmn+ltmx)-xpnd*(ltmx-ltmn)
        urlon = 0.5*(lnmn+lnmx)+xpnd*(lnmx-lnmn)
        urlat = 0.5*(ltmn+ltmx)+xpnd*(ltmx-ltmn)
        return lllon,urlon,lllat,urlat
    
    def getRuptime(self):
        return self.ruptime
        
    def slipinit(self,nt,dt,vrmax,ntw,HKT=False):
        #self.ruptime = np.linspace(0.,(nt-1)*dt,nt)
        self.vrmax   = vrmax
        self.ntw    = ntw
        # Calculate slip
        if HKT:
            zeroslip = np.zeros((ntw,2))
            ## Use of rmin is for compatibility with HKT
            rmin = self.length
            sbf_times = [[] for x in range(len(self.subfaults))]
            for sbf in self.subfaults:
                if sbf.hypo_distance < rmin and sbf.hypo_distance > 0.:
                    rmin = sbf.hypo_distance
            rmin *= 1.25
            ### Do it just like HKT
            v1 = ntw*dt
            igj = [None for x in range(0,len(self.subfaults))]
            for time in self.ruptime:
                r0=time*vrmax+rmin
                r1=r0-v1*vrmax
                r1=max(r1,0.0)
                for sbf in self.subfaults:
                    r=mt.hypot(sbf.hypo_deltaz,sbf.hypo_distance)
                    if r <= r0 and r >= r1:
                        sbf_times[sbf.index].append(time)
                        if igj[sbf.index] == None:
                            igj[sbf.index] = [[(time,r1,r0)],\
                                            (sbf.cntr.x,sbf.cntr.y,sbf.cntr.z,r)]
                        else:
                            igj[sbf.index][0].append((time,r1,r0))
            for sbf in self.subfaults:
                sbf_time = sbf_times[sbf.index]
                sbf.setSlip(sbf_time,zeroslip[:len(sbf_time),:])
        else:
            igj = None
            for sbf in self.subfaults:
                sbf_times = np.zeros(self.ntw+1)
                # Calculate subfault rupture time
                rup_dist  = mt.hypot(sbf.hypo_deltaz,sbf.hypo_distance)
                rup_init  = rup_dist/vrmax
                sbf_times[0] = rup_init
                sbf_times[1:] = np.linspace(rup_init+2*dt,rup_init+dt*(ntw+1),ntw)
                sbf.setSlip(sbf_times,np.zeros((ntw+1,2)))
        return igj

    def moment(self):
        moment = 0.
        for sbf in self.subfaults:
            if sbf.cumslip == None:
                continue
            moment += sbf.potency*mt.hypot(sbf.cumslip[0],sbf.cumslip[1])
        return moment
        
    def rectangular_tessellation(self,nx=1,ny=1,Fix_hypo = False,index_init=0):
        self.sbf_len = self.length/float(nx)
        self.sbf_wid = self.width/float(ny)
        self.sbf_nx = nx
        self.sbf_ny = ny
        self.Tessellation = 'Rectangular'
        self.Ordering = 'BTLR'
        #print nx,ny,self.sbf_len,self.sbf_wid,self.length,self.width
        dwdt = self.sbf_wid*mt.cos(mt.radians(self.dip))
        ddip = self.sbf_wid*mt.sin(mt.radians(self.dip))
        distmin = mt.hypot(self.length,self.width)
        index = index_init
        if False:
            for i in range(0,nx):
                lon,lat,baz = self.g.fwd(self.origin.x,self.origin.y,self.strike,\
                                        1000.*(i+0.5)*self.sbf_len)
                lon,lat,baz = self.g.fwd(lon,lat,self.strike-90.,1000.*0.5*dwdt)
                dep = self.h_bottom - 0.5*ddip
                sbf_cntr = Point(lon,lat,dep) 
                for j in range(0,ny):
                    self.subfaults.append(SubFault(strike=self.strike,dip=self.dip,\
                        length=self.sbf_len,width=self.sbf_wid,cntr=sbf_cntr,\
                        raypaths=self.raypaths,velmod=self.velmod,index=index))
                    index += 1
                    # Check for minimum subfault center vs. hypocener distance
                    az,baz,dist = self.g.inv(self.hypo.x,self.hypo.y,lon,lat)
                    fault_dist = mt.hypot(dep - self.hypo.z, dist*0.001)
                    if fault_dist < distmin:
                        distmin = fault_dist
                        sbf_mindist = self.subfaults[-1]               
                    # Set next subfault center
                    lon,lat,baz = self.g.fwd(lon,lat,self.strike-90.,1000.*dwdt)
                    dep -= ddip 
                    sbf_cntr = Point(lon,lat,dep)
        else:
            llons,lats,lbazs = self.g.fwd([self.origin.x]*nx,[self.origin.y]*nx,\
                                        [self.strike]*nx,\
                                        1000.*np.linspace(0.5*self.sbf_len,\
                                                self.length-0.5*self.sbf_len,nx))
            for llon,llat in zip(llons,lats):
                clons,clats,cbazs = self.g.fwd([llon]*ny,[llat]*ny,[self.strike-90.]*ny,\
                                        1000.*np.linspace(0.5*dwdt,(ny-0.5)*dwdt,ny))
                for j,(clon,clat) in enumerate(zip(clons,clats)):
                    dep = self.h_bottom - (j+0.5)*ddip
                    sbf_cntr = Point(clon,clat,dep) 
                    self.subfaults.append(SubFault(strike=self.strike,dip=self.dip,\
                        length=self.sbf_len,width=self.sbf_wid,cntr=sbf_cntr,\
                        raypaths=self.raypaths,velmod=self.velmod,index=index))
                    index += 1
                    # Check for minimum subfault center vs. hypocener distance
                    az,baz,dist = self.g.inv(self.hypo.x,self.hypo.y,clon,clat)
                    fault_dist = mt.hypot(dep - self.hypo.z, dist*0.001)
                    if fault_dist < distmin:
                        distmin = fault_dist
                        sbf_mindist = self.subfaults[-1]       
                        
        if not Fix_hypo:
            print 'Changing hypo from ',self.hypo,' to subfault center at ', \
                sbf_mindist.cntr
            self.hypo = sbf_mindist.cntr
        for sbf in self.subfaults:
            sbf.hypo_deltaz   = sbf.cntr.z - self.hypo.z
            az,baz,dist = self.g.inv(self.hypo.x,self.hypo.y,sbf.cntr.x,sbf.cntr.y) 
            sbf.hypo_azimuth  = az
            sbf.hypo_distance = dist*0.001 # in km
            #if sbf.cntr.z > 50.: AGREES WITH HKT to within 0.5 km  
            #    print '%7.2f'*10 % (az,baz,dist*0.001,sbf.cntr.x,sbf.cntr.y,\
            #    sbf.cntr.z,self.hypo.x,self.hypo.y,self.hypo.z,sbf.hypo_deltaz)
         
    def TimeSlipDistance_plot(self,ax,cmap,norm):
        time = []
        dist = []
        slip = []
        for sbf in self.subfaults:
            times = sbf.slip_rate[0].get_knots()[1:-3]
            for t in times:
              slipx = self.slip[0]([t])[0]
              slipy = self.slip[1]([t])[0]
              slip.append(mt.hypot(slipx,slipy))
              time.append(t)
              dist.append(mt.hypot(sbf.hypo_distance,sbf.hypo_deltaz))
        ax.scatter(time,dist,slip,cmap=cmap,norm=norm)
                
            
            
    
    def map_plot(self,ax,m=None,xpnd=0.5,norm=None, cmap=plt.cm.hot_r,
            dll=None,colorbar=True,labels='ws',field=None,time=None,
            Plot_subfaults=True,grid=True,resolution='h',plot_hypo=True,
            background='',cb_off=(0.1,0.1),cb_orient='vertical'): 
        '''
        '''
        # Draw map if none has been provided
        if m == None:
            lllon,urlon,lllat,urlat = self.geographic_limits(xpnd=xpnd)
            m = Basemap(lllon,lllat,urlon,urlat,projection='merc',\
            resolution=resolution)
            if background == 'shadedrelief':
                m.shadedrelief()
            else:
                m.drawmapboundary(fill_color='aqua') 
                # fill continents, set lake color same as ocean color. 
                m.fillcontinents(color='coral',lake_color='aqua')
            o = m.drawcoastlines()
            if grid ==True:
                if dll == None:
                    dll = []
                    if urlon - lllon > 5.:
                        dll.append(1.0)
                    else:
                        dll.append(0.5)
                    if urlat - lllat > 5.:
                        dll.append(1.0)
                    else:
                        dll.append(0.5)
                lbl = [0,0,0,0]
                if 'w' in labels: lbl[0] = 1
                if 'e' in labels: lbl[1] = 1
                if 'n' in labels: lbl[2] = 1
                if 's' in labels: lbl[3] = 1
                m.drawparallels(np.arange(-90.,90.,dll[1]),labels=lbl,labelstyle='+/-')
                m.drawmeridians(np.arange(-180.,180.,dll[0]),labels=lbl)#,labelstyle='+/-')
        if True: # Maybe later this will be an option
            if self.shape == 'Rectangular':
                fltply = rectangular_polygon(self.strike,self.dip,self.length,\
                                self.width,origin=self.origin,basemap=m)
                #poly = Polygon(fltply,facecolor='none',edgecolor='black',linewidth=.5)
                #ax.add_patch(poly)
                path = Poly_new(fltply)
                patch = patches.PathPatch(path,edgecolor='black',facecolor='none')
                #,lw=50,zorder=zorder+2001)
                ax.add_patch(patch)

        if Plot_subfaults:
           for sbf in self.subfaults:
              sbf.map_plot(ax,m,cmap=cmap,norm=norm,field=field,time=time,
                           zorder=o.zorder+1)
        if plot_hypo:
            m.plot(self.hypo.x,self.hypo.y,marker='*',color='cyan',latlon=True,
                    zorder=123)
        if norm != None and colorbar:   ### Now for the colorbar
            norm = mpl.colors.Normalize(vmin=0., vmax=norm)
            # get axes bounds.
            pos = ax.get_position()
            l, b, w, h = pos.bounds
            # create axes instance for colorbar on right.
            fig = ax.get_figure()
            if cb_orient == 'horizontal':
                cax = fig.add_axes([l+w+cb_off[0]*w, b+cb_off[1]*h, 0.7*w, 0.1*h])
            else:
                cax = fig.add_axes([l+w+cb_off[0]*w, b+cb_off[1]*h, 0.05*w, 0.9*h])
            cbar = mpl.colorbar.ColorbarBase(ax=cax,cmap=cmap,norm=norm,\
                                             orientation=cb_orient)
            cbar.set_label('metres slip')
            
        return m
         
    def xy_plot(self,ax,cmap=plt.cm.hot_r,field=None,norm=None,plot_hypo=True,
                labels=['lb'],time=None,colorbar=False,cb_off=(0.1,0.1),
                cb_orient='vertical',vec_scale=None):
        '''
        vec_scale:  scale factor for vecors in units of km/field
        '''
        #print field
        if self.shape == 'Rectangular':
            ax.set_xlim(0,self.length)
            ax.set_ylim(0,self.width)
            ax.set_aspect('equal')


            #if cmap == None and field != None:
            area_average = 0.   
            # Set scaling factor for slip vectors
            for sbf in self.subfaults:
                area_average += sbf.area
            area_average /= len(self.subfaults)
            if vec_scale == None:
                vec_scale = 2.*mt.sqrt(area_average)
            # Loop over subfaults to plot them
            x  = 0.5*self.sbf_len
            ix = 0
            while x < self.length:
                y  = 0.5*self.sbf_wid
                jy = 0
                while y < self.width:
                    if not hasattr(self,'Ordering'):
                        print 'fault.xy_plot: No ordering, assuming BTLR'
                        indx = ix*self.sbf_ny+jy
                    elif self.Ordering == 'BTLR':
                        indx = ix*self.sbf_ny+jy
                    elif self.Ordering == 'LRTB':
                        indx = (self.sbf_ny-jy-1)*self.sbf_nx+ix
                    sbf = self.subfaults[indx]
                    sbf.xy_plot(ax,(x,y),cmap=cmap,field=field,norm=norm,\
                            time=time,scale=vec_scale)
                    y += self.sbf_wid
                    jy += 1
                x += self.sbf_len
                ix += 1
        else:
            raise Exception, 'No non-rectangular faults can be plotted'
        if plot_hypo:
                plt.plot(self.hypo_x,self.hypo_y,'*',color='cyan')
        if 'b' in labels:
            ax.set_xlabel('Along-strike Distance (km)')
        if 'l' in labels:
            ax.set_ylabel('Along-dip Distance (km)')
        if not 'b' in labels:
            ax.axes.get_xaxis().set_visible(False)
        if not 'l' in labels:
            ax.axes.get_yaxis().set_visible(False)
        if norm != None and colorbar:   ### Now for the colorbar
            norm = mpl.colors.Normalize(vmin=0., vmax=norm)
            # get axes bounds.
            pos = ax.get_position()
            l, b, w, h = pos.bounds
            # create axes instance for colorbar on right.
            fig = ax.get_figure()
            if cb_orient == 'horizontal':
                cax = fig.add_axes([l+cb_off[0]*w, b+cb_off[1]*h, 0.7*w, 0.1*h])
            else:
                cax = fig.add_axes([l+w+cb_off[0]*w, b+cb_off[1]*h, 0.05*w, 0.9*h])
            cbar = mpl.colorbar.ColorbarBase(ax=cax,cmap=cmap,norm=norm,\
                                             orientation=cb_orient)
            cbar.set_label('metres slip')

    def rupture_GreenFuncs(self,phase,tdelay,vrmax,twin):
        '''
        Returns a set of subfault Green's functions for time tdelay seconds 
        after rupture. Currrently only works for a maximum rupture velocity and 
        time window following the rupture front.
        
        '''
        if vrmax == 0.:
            sbfs = self.subfaults
        else:
            sbfs = []
            for sbf in self.subfaults:
                rupture_distance = mt.hypot(sbf.hypo_distance,sbf.hypo_deltaz)
                rupture_initiation = rupture_distance/vrmax
                print rupture_distance,rupture_initiation,vrmax,tdelay
                if tdelay >= rupture_initiation and tdelay < rupture_initiation+twin:
                    sbfs.append(sbf)
        return sbfs

    def coseismic_matrix(self, gps, rakes, unorm=np.eye(3), alp=0.5, 
                 dpcor=False, order=True, utm = False, dstmx=0.):
       '''
       '''
       n_rake = len(rakes)
       n_comp = 3
       n_sflt = len(self.subfaults)
       elat   = np.zeros(n_sflt)
       elon   = np.zeros(n_sflt)
       edep   = np.zeros(n_sflt)
       strk   = np.zeros(n_sflt)
       dp     = np.zeros(n_sflt)
       lngt   = np.zeros(n_sflt)
       wdt    = np.zeros(n_sflt)
       rlon = gps[:,0]
       rlat = gps[:,1]
       rdep = gps[:,2]
       for i,sbf in enumerate(self.subfaults):
          elon[i] = sbf.cntr.x
          elat[i] = sbf.cntr.y
          edep[i] = sbf.cntr.z
          lngt[i] = sbf.length
          wdt[i]  = sbf.width
          strk[i] = sbf.strike
          dp[i]   = sbf.dip
       amat = _coseismic_matrix(alp,elon,elat,edep,strk,dp,lngt,wdt,
                rakes,rlon,rlat,rdep,utm,dstmx,dpcor,unorm,order) #,n_rake,n_comp,n_recv,n_sflt)
       return amat

    def coseismic_deformation(self,rlon,rlat,time=None,grid=True):
        elon=[]
        elat=[]
        edep=[]
        strk=[]
        dip=[]
        lng=[]
        wdt=[]
        disl1=[]
        disl2 = []
        for sbf in self.subfaults:
            elon.append(sbf.cntr.x)
            elat.append(sbf.cntr.y)
            edep.append(sbf.cntr.z)
            strk.append(sbf.strike)
            dip.append(sbf.dip)
            lng.append(sbf.length)
            wdt.append(sbf.width)
            if time == None:
                disl1.append(sbf.cumslip[0])
                disl2.append(sbf.cumslip[1])
            else:
                slipx = sbf.slip[0]([time])[0]
                slipy = sbf.slip[1]([time])[0]
                disl1.append(slipx)
                disl2.append(slipy)
        if grid:
            x,y = np.meshgrid(rlon,rlat)
        else:
            x = np.array(rlon)
            y = np.array(rlat)
        #print elon,elat,len(x),len(y)
        u,v,w = ffi.fault_disp(elon, elat, edep, strk, dip, lng, wdt, disl1,\
                       disl2, x.flatten(),y.flatten())
        if grid:
            u=u.reshape((len(rlat),len(rlon)))
            v=v.reshape((len(rlat),len(rlon)))
            w=w.reshape((len(rlat),len(rlon)))
        return u,v,w
         
# Should probably go into ffi.py
def velmod_findparam(velmod,parameter,z,iprofile=0):
    '''
    Find the value of elasticity parameter for a given depth z
    Input:
        velmod    = a velocity model structure as returned by read_velmod
        z         = depth in km
        parameter = 'rho', 'alpha', or 'beta'
        iprofile  = 0,1,2, to use source, bounce point or recevier profile
    '''
    ns = velmod['ns']
    nl = ns[iprofile]
    if   iprofile == 0:
        vmod = velmod['vmod'][0:4,       0:ns[0]      ]
    elif iprofile == 1:
        vmod = velmod['vmod'][0:4,    ns[0]:ns[0]+ns[1]    ] 
    elif iprofile == 2:
        vmod = velmod['vmod'][0:4,ns[0]+ns[1]:ns[0]+ns[1]+ns[2]] 
    h = 0.
    for j in range(0,nl):
        if z<h+vmod[3,j]:
            break
        else:
            h += vmod[3,j]
            
    if parameter == 'rho':
        value = vmod[2,j]
    elif parameter == 'alpha':
        value = vmod[0,j]
    elif parameter == 'beta':
        value = vmod[1,j]
    elif parameter == 'mu': # Return mu in Pa 
        # (scale rho to kg/m^3 (1.e3), beta^2 to (m/s)^2 (1.e6) 
        value = vmod[2,j]*vmod[1,j]*vmod[1,j]*1.e9
    else:
         raise Exception, 'Unknown parameter %s' % parameter
    return value


class SubFault():
   'An earthquake subfault class'

   def __init__(self, strike=None,dip=None,length=None,width=None,ulcrnr=None,\
    cntr=None,cumslip=None,rake=None,velmod=None,Verbose=False,raypaths=None,
    index=None):
      self.slip     = None
      self.ruptime  = None
      self.cumslip  = None
      self.t0       = None
      self.velmod   = velmod
      self.index    = index
      self.GreenFuncs = {}
      self.raypaths = raypaths
      g = Geod(ellps='WGS84')
      #Verbose = True
      if strike != None and dip != None and length != None and width != None:
         # Rectangular subfault - establish centre based on input (cntr,ulcrnr,etc)
         self.shape  = 'Rectangular'
         self.strike = strike
         self.dip    = dip
         self.length = length
         self.width  = width
         self.area   = length*width 
         dz = mt.sin(mt.radians(dip))*width
         dw = mt.cos(mt.radians(dip))*width
         if   type(cntr) == Point:
            self.cntr = cntr
         elif type(ulcrnr) == Point:
            #print '0 ',ulcrnr.x,ulcrnr.y
            (tmpx,tmpy,tmpbaz) = g.fwd(ulcrnr.x,ulcrnr.y,strike,500.*length)
            (tmpx,tmpy,tmpbaz) = g.fwd(tmpx,tmpy,strike+90.,500.*dw)
            self.cntr = Point(tmpx,tmpy,ulcrnr.z+0.5*dz)
         else:
            print 'ugh!'
            sys.exit(0)
         if Verbose:
            print 'subfault center at (%8.3f,%7.3f)' % (self.cntr.x,self.cntr.y)
         # Now that center is defined, calculate the boredering Polygon
         self.polygon = rectangular_polygon(self.strike,self.dip,self.length,\
                                self.width,center=self.cntr)
      else:
         raise Exception, 'No non-rectangular faults can be defined'
      # Set potency if velmod set
      if velmod != None:
         #mu  = velmod_findparam(self.velmod,'mu',self.cntr.z)
         # convert area from km^2 to m^2 (1.e6)
         self.set_potency() 
          
      #
      if type(cumslip) == tuple and len(cumslip) == 2:
         self.cumslip = cumslip
      elif type(cumslip) == float and type(rake) == float:
         self.cumslip = (cumslip*mt.cos(mt.radians(rake)),cumslip*mt.sin(mt.radians(rake)))
      elif cumslip != None or rake != None:
         raise Exception, 'Need either cumslip=(sx,sy) or cumslip and rake sets'
      if Verbose:
         if self.cumslip != None:
            print 'cumslip = (%g,%g)' % self.cumslip
         else:
            print 'cumslip not set'

   def __str__(self):
      pstr = 'Subfault'
      pstr += '\n\tcentre: (%8.3f, %7.3f, %5.2f)' % \
          (self.cntr.x,self.cntr.y,self.cntr.z)
      pstr += '\n\tvertices: '
      for xyz in self.polygon:
         pstr += '\n\t\t(%8.3f, %7.3f, %5.2f)' % (xyz.x,xyz.y,xyz.z)
      '''
      if self.slip != None:
         times = self.slipx.x
         x     = self.slipx.y
         y     = self.slipy.y
         pstr += '\n\tslip:\t'
         for i,time in enumerate(times):
            slip = self.slip(time)
            pstr += '%6.2f:(%6.2f,%6.2f)' % (time,slip.x,slip.y)
            if time != times[-1]: 
                pstr += ', '
                if (i-2) % 3 == 0: pstr += '\n\t\t' 
      else:
         pstr += '\n\tslip: None'
      '''
      for key in self.__dict__.keys():
          if key == 'slip' or key == 'cntr' or key == 'polygon'\
            or key == 'raypaths' or key == 'g' or key == 'slipx' or\
            key == 'slipy':
              continue
          if self.__dict__[key] == None:
              continue
          pstr += '\n\t%s:\t%s' % (key,self.__dict__[key].__str__())
      return pstr

   def copy(self):
       sbf = SubFault(strike=self.strike,dip=self.dip,\
                        length=self.length,width=self.width,cntr=self.cntr,\
                        raypaths=self.raypaths,velmod=self.velmod,index=self.index)
       sbf.area          = self.area
       if self.cumslip != None:
           if type(self.cumslip) == tuple:
               sbf.cumslip = self.cumslip
           else:
               sbf.cumslip = list(self.cumslip)
       sbf.hypo_azimuth  = self.hypo_azimuth
       sbf.hypo_deltaz   = self.hypo_deltaz
       sbf.hypo_distance = self.hypo_distance
       if self.polygon != None:
          sbf.polygon    = list(self.polygon)
       if self.ruptime != None:
           sbf.ruptime   = list(self.ruptime)
       sbf.shape         = self.shape
       if self.slip != None and self.index == 0:
           print "Can't copy subfault slip"
       if hasattr(self,'rake'):
           sbf.rake      = self.rake
       return sbf

   def containsPoint(self,p):
       '''
       Test if a subfault cotains a point. Point should have the
       attributes x nd y, corresponding to longitude and latitude respectively. 
       '''
       poly = ShapelyPolygon([(pt.x,pt.y) for pt in self.polygon])
       return poly.contains(ShapelyPoint(p.x,p.y))

   def ruptimes(self):
       if not hasattr(self,'slip_rate') or len(self.slip_rate)<1:
           raise Exception, "Subfault can't return ruptimes because sliprate not set"
       return self.slip_rate[0].get_knots()[1:-1]

   def set_potency(self,Verbose=False):
       '''
       Set the subfault potency. Velmod must be set, and if it is this
       routine will loop through the layers of the velocity model, adding the
       corresponding contribution of each layer to the subfault potency.
       '''
       if self.velmod is None:
           raise Exception, "Can't set potency for subfault %d with no velmod!" % self.index
       velmod = self.velmod
       iprofile = 0
       ns = velmod['ns']
       nl = ns[iprofile]
       if   iprofile == 0:
           vmod = velmod['vmod'][0:4,       0:ns[0]      ]
       elif iprofile == 1:
           vmod = velmod['vmod'][0:4,    ns[0]:ns[0]+ns[1]    ] 
       elif iprofile == 2:
           vmod = velmod['vmod'][0:4,ns[0]+ns[1]:ns[0]+ns[1]+ns[2]] 
       sbf_snd = mt.sin(mt.radians(self.dip))
       sbf_top = self.cntr.z - 0.5*self.width*sbf_snd
       sbf_btm = self.cntr.z + 0.5*self.width*sbf_snd
       ztop = 0.
       sbf_pot = 0.
       for j in range(0,nl):
            zbtm = ztop+vmod[3,j]
            mu = vmod[2,j]*vmod[1,j]*vmod[1,j]*1.e9 # mu in Pa 
            # Note that zbtm == ztop, or vmod[3,j], flags half-space
            if sbf_pot == 0. and (sbf_top < zbtm or zbtm == ztop):
                if sbf_btm < zbtm or zbtm == ztop: # subfault contained completely in layer j
                    sbf_pot  = mu*self.width
                    if Verbose:
                        print 'Subfault %d completely in layer %d, potency=%8.2e' % (self.index,j,sbf_pot)
                    break
                else:               # Only part of subfault is in layer j
                    sbf_pot  = mu*(zbtm-sbf_top)/sbf_snd
                    if Verbose:
                        print 'Subfault %d top %5.3f km in layer %d, potency=%8.2e' % (self.index,zbtm-sbf_top,j,sbf_pot)
            elif sbf_pot >  0.:
                if sbf_btm < zbtm or zbtm == ztop:
                    sbf_pot += mu*(sbf_btm-ztop)/sbf_snd
                    if Verbose:
                        print 'Subfault %d bottom %5.3f km in layer %d, potency=%8.2e' % (self.index,sbf_btm-ztop,j,sbf_pot)
                    break
                else:
                    sbf_pot += mu*vmod[3,j]/sbf_snd
                    if Verbose:
                        print 'Subfault %d includes all %5.3f km of layer %d, potency=%8.2e' % (self.index,vmod[3,j],j,sbf_pot)
            ztop += vmod[3,j]
       sbf_pot *= (self.length*1.e6) 
       self.potency = sbf_pot      
       return sbf_pot

   def getSlip(self):
       return self.ruptime,self.slip
   
   def setSlip(self,ruptime,slips):
       '''
       ruptime: an list of times at which slip values are set initial time is 
       slip   : a list of 2-component slip vectors, one for each ruptime[]
       
       setSlip sets initial slip=(0.,0.) at time t=ruptime[0], and appends to 
       this a slip vector for each subsequent time in ruptime. It returns a 
       functiont that will, as a funtion of time t, where 
       ruptime[0]<= t <=ruptime[-1], return a 2-D slip vector (Point) that 
       is linearly interpolated from the times in ruptime[]. 
       '''
       if len(slips) != len(ruptime):
           raise Exception, 'len(slip) != len(truptime)!'
       slipx = []
       slipy = []
       for slip in slips:
           slipx.append(slip.x)     
           slipy.append(slip.y)     
       self.slip = (UnivariateSpline(ruptime, slipx, s=0., k=1),
                    UnivariateSpline(ruptime, slipy, s=0., k=1))
       return 
       

   def map_plot(self,ax,m,cmap=None,field=None,time=None,norm=None,
                linewidth=1,alpha=0.,zorder=None):
      '''
      '''
      sbfply = rectangular_polygon(self.strike,self.dip,self.length,\
                                   self.width,center=self.cntr,basemap=m)
      path = Poly_new(sbfply)
      patch = patches.PathPatch(path,edgecolor='black',facecolor='none')
      ax.add_patch(patch)

      #sfply = []
      #for xyz in self.polygon:
      #   sfply.append(m(xyz.x,xyz.y))
      #sfply = np.array(sfply)
      #m.plot(sfply[:,0],sfply[:,1],zorder=zorder+200,latlon=False,color='black',
      #       lw=linewidth,alpha=alpha)
      self.fieldplot(sbfply,ax,cmap=cmap,field=field,norm=norm,time=time,
                zorder=zorder)
            
   def xy_plot(self,ax,cntr_xy,cmap=None,field=None,norm=None,scale=None,\
            linewidth=1,alpha=0.,zorder=None,time=None):
      if self.shape == 'Rectangular':
         sfply = []
         sfply.append((cntr_xy[0]-0.5*self.length,cntr_xy[1]-0.5*self.width))
         sfply.append((cntr_xy[0]+0.5*self.length,cntr_xy[1]-0.5*self.width))
         sfply.append((cntr_xy[0]+0.5*self.length,cntr_xy[1]+0.5*self.width))
         sfply.append((cntr_xy[0]-0.5*self.length,cntr_xy[1]+0.5*self.width))
         sfply.append((cntr_xy[0]-0.5*self.length,cntr_xy[1]-0.5*self.width))
         sfply = np.array(sfply)
      else:
         raise Exception, 'No non-rectangular faults can be plotted'
      path = Poly_new(sfply)
      if zorder == None:
          patch = patches.PathPatch(path,edgecolor='black',lw=1.5,facecolor='none')
          ax.add_patch(patch)
      else:
          ax.plot(sfply[:,0],sfply[:,1],zorder=zorder+200,color='black',
                        lw=linewidth,alpha=alpha)
      if type(time) == float:
          self.fieldplot(sfply,ax,cmap,field,norm,scale=scale,
                        cntr_xy=cntr_xy,time=time)
      elif type(time) == list or type(time) == tuple:
          verts = self.timeplot(sfply,ax,cmap,field,norm,scale=scale,
                        cntr_xy=cntr_xy,time=time)
      return 
         
   def fieldplot(self,sfply,ax,cmap=None,field=None,norm=None,scale=None,
                 time=None,zorder=None,cntr_xy=None):
      path = Poly_new(sfply)
      #patch = patches.PathPatch(path,facecolor='orange',edgecolor='black',lw=50,\
      #               zorder=zorder+2001)
      #ax.add_patch(patch)
      #return
      if field == None: return
      # Then plot the polygon faces
      if cmap == None or norm == None:
          raise Exception, 'Subfault needs cmap and xnorm to plot field %s' % field
      if not self.__dict__.has_key(field):
          raise Exception, 'Subfault does not have field' % field
      if self.__dict__[field] == None:
          return
      if field == 'slip':
          try:
              slipx = self.slip[0]([time])[0]
              slipy = self.slip[1]([time])[0]
              field_value = mt.hypot(slipx,slipy)
              #if field_value != 0.: print time,field_value,slipx,slipy
          except ValueError:
              print "Attempt to plot slip with invalid time or slip function not set"
              return    
      elif field == 'cumslip':
          field_value = mt.hypot(self.cumslip[0],self.cumslip[1])
      else:
          field_value = self.__dict__[field]
      
      if field_value == 0.: return
      color = cmap(field_value/norm)[:3]
      if zorder == None:
          patch = patches.PathPatch(path,facecolor=rgb2hex(color),
                                edgecolor='blue',lw=.5)
      else:
          patch = patches.PathPatch(path,facecolor=rgb2hex(color),
                                edgecolor='blue',lw=.5,zorder=zorder+1)
      ax.add_patch(patch)
      if field.endswith('slip') and scale != None:
          vecscale = scale/norm
          if field == 'cumslip':
              ax.arrow(cntr_xy[0],cntr_xy[1],self.cumslip[0]*vecscale,\
                self.cumslip[1]*vecscale,color='magenta',width=.1,clip_on=False)
          elif field == 'slip':
              ax.arrow(cntr_xy[0],cntr_xy[1],slipx*vecscale,\
                  slipy*vecscale,color='blue',width=.1,clip_on=False)

      '''
      Maybe I can use thi for something else .... ?
      # Make sure slip is a tuple and rake is set consitent with it   
      if type(self.slip) == tuple and len(self.slip) == 2:
         self.rake = mt.degrees(mt.atan2(self.slip[1],self.slip[0]))
      elif type(self.slip) == float and type(rake) == float:
         self.slip = (slip*mt.cos(mt.radians(rake)),slip*mt.sin(mt.radians(rake)))
         self.rake = rake
      elif slip != None or rake != None:
          raise Exception,'Need either slip=(sx,sy) or slip and rake sets'
      '''
      
   def timeplot(self,sfply,ax,cmap=None,field=None,norm=None,scale=None,
                 time=None,zorder=None,cntr_xy=None,colors=['green','red'],
                 alpha=0.35):
      path = Poly_new(sfply)
      #patch = patches.PathPatch(path,facecolor='orange',edgecolor='black',lw=50,\
      #               zorder=zorder+2001)
      #ax.add_patch(patch)
      #return
      if field == None: return
      if not self.__dict__.has_key(field):
          raise Exception, 'Subfault does not have field' % field
      if self.__dict__[field] == None:
          return
      times = np.linspace(time[0],time[1],100)
      ys    = np.linspace(sfply[1][1],sfply[2][1],100)
      xmin,xmax = sfply[0][0],sfply[1][0]
      xnorm = (xmax-xmin)/norm
      try:
          fieldx = getattr(self,field)[0](times)
          fieldy = getattr(self,field)[1](times)
          for irk,rake in enumerate([self.rake-45.,self.rake+45.]):
            field = fieldx*mt.cos(mt.radians(rake)) + \
                    fieldy*mt.cos(mt.radians(rake-90.))
            field = np.clip(xmax - xnorm*field,xmin,xmax)
            verts = [(field[0],ys[0]),(xmax,ys[0]),(xmax,ys[-1])]
            verts += zip(field[::-1],ys[::-1])
            path  = Poly_new(verts)
            patch = patches.PathPatch(path,facecolor=colors[irk],alpha=alpha,
                                 edgecolor='blue',lw=.5)            
            ax.add_patch(patch)
      except ValueError:
          print "Invalid field %s or tmin,tmax=(%gm,%g)" % (field,time[0],time[1])
          return
      rtims = self.slip_rate[0].get_knots()
      trup  = ys[0]+(ys[-1]-ys[0])*(rtims[1]-time[0])/(time[1]-time[0])
      ax.plot([xmin,xmax],[trup,trup],color='red')
      tstp  = ys[0]+(ys[-1]-ys[0])*(rtims[-2]-time[0])/(time[1]-time[0])
      ax.plot([xmin,xmax],[tstp,tstp],color='blue')
      return verts

   def GreensFuncCol(self,phase,ruptime,toff):
      '''
      Returns a Nx2 matrix (i.e., 2 column vectors, each containing subfault 
      Green's functions for rake+-45 deg, calculated by method bodywave for 
      time ruptime after the origin time.
      Input:
          phase:    currently only 'P' allowed
          ruptime:  time in seconds after origin, must equal a time in 
                    subfault's ruptime array, or 'None' is returned.
          toff:     arraival time - trace onset time, in seconds.
      Output:
          GF[n,2], Where n is the sum of the lenths of data in the subfaults 
                   Green's function for this phase, with each trace shifted by
                   ruptime seconds.
          None     If ruptime is not equal to an element of the subfault's 
                   ruptime array.
      '''

      if not ruptime in self.ruptime:
          return None
      nrow = 0
      for trc in self.GreenFuncs[phase][0]:
          nrow += trc.stats.npts
      GF = np.zeros((nrow,2))
      for jcol in range(0,2):
          idx = 0             
          for trc in self.GreenFuncs[phase][jcol]:
              if phase == 'P':
                  # Calculate time shift as a sum of threee terms:
                  #    (1) toff, the pre-arrival time window (normally 
                  #        chosen to be compatible with observed trace)
                  #    (2) delay, of subfault ray at receiver, measured w.r.t.
                  #        hypocentral ray (i.e., infinite rupture velocity)
                  #    (3) ruptime, subfault rupture time, w.r.t. origin time 
                  time_shift = ruptime + toff - trc.stats.delay
                  ishift = int(time_shift/trc.stats.delta + 0.4)
                  ## For compatibility with HKT
                  dly = trc.stats.delay
                  dt  = trc.stats.delta
                  if ruptime-dly+toff >= 0.:
                      ishift = int((ruptime-dly+toff)/dt+0.5) 
                  else:
                      ishift = int((ruptime-dly+toff)/dt-0.5) 
                  ishift -= 1
                  if ishift < 0.:
                      raise Exception, 'Ruptime+Toff(%g) < delay (%g)' \
                       % (ruptime+toff, trc.stats.delay)
                  #print trc.stats.station,tshift,ishift
                  if ishift == 0:
                       GF[idx:idx+trc.stats.npts,jcol] = trc.data[:]
                  else:
                       GF[idx+ishift:idx+trc.stats.npts,jcol] = trc.data[0:-ishift]
              else:
                  raise Exception, "No Green's functions for phase %s" % phase
              idx += trc.stats.npts
      return GF
      
   def bodywave(self,st,phase,t12,rake,velmod=None,filtargs=None):
      '''
       Uses the Kikuchi (Haskell) reflectivity code to calculate body wave 
       synthetic seismograms for a point source centered on a subfault. 
       Two sets of seismograms are calculated for rake-45. and rake+45. deg, 
       respectively. Slip along each of these directions is 1/sqrt(2), 
       corresponding to unit slip in the direction of rake.
       
       Input:
          st        = an obspy stream with traces for which the corresponding 
                      synthetics are to be calculated. 
              Note that the traces in st must have the following attributes in stats:
                  gcarc     = event-to-station great circle distance
                  azimuth   = event-to-station azimuth

          phase     = 'P', for vertical compoennt P wave 
                    = 'SH', for transverse component S wave
          velmod    = a velmod array, of the type returned by rouine read_velmod
                      - only needs to be set once for each subfault
          t12       = the half-duration of the symmetric, triangular source time 
                      function
          toff      = the offset of the start of trace w.r.t. phase arrival time. 
                      Must be large enough to accomodate arrivals earlier than the
                      corresponding phase from the hypocentre (usually a few sec).
          rake      = rake angle in degrees, synthetics calculated for rake+/-45.
          filtargs  = if set, traces will be filtered according to 
                      trace.filter(filtargs), e.g.: 
                filtargs = ("bandpass", freqmin=0.02,freqmax=0.1,
                            corners=4,zerophase=False)
                      
       Output:
          self.GreenFuncs[phase][0:1] is set to a list of 2 Stream objects, 
       one for each rake direction. The delay in arrival time at each receiver 
       of rays from the subfault. as measured w.r.t. the hypocentral ray (i.e., 
       assuming infinite rupture velocity) is stored in tr.stats.delay, where 
       tr is a trace element of each stream.  
      '''
      ### Check that velmod is supplied, and calculate moment parameters - only once when self.velmod = None
      if self.velmod == None and velmod == None:
         raise Exception, 'Subfault needs velmod for body wave calculation'
      elif self.velmod == None:
         self.velmod = velmod
      velmod = self.velmod
      
      z0 = self.cntr.z-self.hypo_deltaz
      scale_factor = 1.e-6*1.e7/1.e25 # Convert microns->metre, dyne-cm to nM, 
                                      # correct for HKT 1.e25)
      scale_factor *= mt.sqrt(0.5)    # slip=1/sqrt(2) along rake-45., rake+45

      if self.raypaths == None:
          raise Exception, 'Subfault raypaths must be set for bodywave calculation'
      elif not self.raypaths.has_key(phase):
          raise Exception, 'Subfault raypaths has no phase %s' % phase
      elif phase == 'P':  # Ray description parameters for P 
          ib = 1    # P wave incident on receiver
          ic = 1    # Vertical component at receiver
          v_source = velmod_findparam(self.velmod,'alpha',z0)
      elif phase == 'SH': # Ray description parameters for SH
          ib = 3    # SH wave incident on receiver
          ic = 1    # ignored for ib == 3
          v_source = velmod_findparam(self.velmod,'beta',z0)
      else:
          raise Exception, 'Unknown phase ',phase
      #
      if rake < -180. or rake > 180.:
          raise Exception, 'Expected -180.<rake<180., not %g' % rake
      else:
          self.rake = rake          
   
      ## Loop over traces in stream
      stbdy = [Stream(), Stream()]
      for tr in st:
        nt = len(tr.data)
        dt = tr.stats.delta
        gcarc = tr.stats.gcarc
        azimuth = tr.stats.azimuth
         # Calculate ray parameters, slowness p and geometerical spreading g
        gp,ps,v0 = self.raypaths[phase]
        idelt = int(gcarc)
        g=gp[idelt-1]+(gcarc-idelt)*(gp[idelt]-gp[idelt-1])
        p=ps[idelt-1]+(gcarc-idelt)*(ps[idelt]-ps[idelt-1])
        # Store delay w.r.t hypocenter in trc.stats
        tr.stats.delay = delay(azimuth,p,self.hypo_azimuth,\
                                self.hypo_distance,self.hypo_deltaz,v_source)
        # Calculate synthetics
        syn_m45 = kiksyn(nt,dt,ib,ic,self.strike,self.dip,self.rake-45.,
            self.cntr.z,azimuth,p,g,t12,velmod)      
        syn_p45 = kiksyn(nt,dt,ib,ic,self.strike,self.dip,self.rake+45.,
            self.cntr.z,azimuth,p,g,t12,velmod)
        syn_m45 *= self.potency*scale_factor
        syn_p45 *= self.potency*scale_factor
        if hasattr(tr,'norm'):
            syn_m45 /= tr.norm
            syn_p45 /= tr.norm
        stbdy[0].append(Trace(header=tr.stats,data=syn_m45))
        stbdy[1].append(Trace(header=tr.stats,data=syn_p45))
        if filtargs != None:
            stbdy[0][-1].filter (**filtargs)
            stbdy[1][-1].filter (**filtargs)
      self.GreenFuncs[phase] = stbdy        
      
def drawmap(lllon,lllat,urlon,urlat):
   m = Basemap(lllon,lllat,urlon,urlat,projection='merc',resolution='l')
   m.drawmapboundary(fill_color='aqua') 
   # fill continents, set lake color same as ocean color. 
   m.drawcoastlines()
   m.fillcontinents(color='coral',lake_color='aqua')
   if urlon - lllon < 2.5:
      dlon = 0.5
   elif urlon - lllon < 5.:
      dlon =1.0
   else:
      dlon = 2.
   m.drawparallels(np.arange(-90.,90.,1.0),labels=[1,0,0,0])
   m.drawmeridians(np.arange(-180.,180.,dlon),labels=[0,0,0,1])
   return m

def rectsmooth(fault):
    ordering = fault.Ordering
    nx       = fault.sbf_nx
    ny       = fault.sbf_ny
    ndup     = 2*fault.ntw
    nsbf = nx*ny
    M = np.zeros((ndup*nsbf,ndup*nsbf))
    if ordering == 'BTLR':
        ix = 0
        iy = 0
    elif ordering == 'LRTB':
        ix = 0
        iy = ny-1
    for irow in range(0,nx*ny):
        if ordering == 'BTLR':
            jx = 0
            jy = 0
        elif ordering == 'LRTB':
            jx = 0
            jy = ny-1
        for icol in range(0,nx*ny):
            xfact = 0.
            xfree = 0.
            if iy == ny-1:
               xfree = 1. 
            if ix == jx and iy == jy:
                #for k in range(0,ndup):
                #    M[ndup*irow+k,ndup*icol+k] = -6.
                xfact = -6.
            elif abs(ix-jx) == 1 and iy == jy:
                #for k in range(0,ndup):
                #    M[ndup*irow+k,ndup*icol+k] = 1.
                xfact = 1.
            elif abs(iy-jy) == 1 and ix == jx:
                #for k in range(0,ndup):
                #    M[ndup*irow+k,ndup*icol+k] = (1.+xfree)*1.
                xfact = 1.*(1.+xfree)
            elif abs(ix-jx) == 1 and abs(iy-jy) == 1:
                #for k in range(0,ndup):
                #    M[ndup*irow+k,ndup*icol+k] = (1.+xfree)*0.5
                xfact = 0.5*(1.+xfree)
            if xfact != 0.:
                M[ndup*irow:ndup*(irow+1),
                  ndup*icol:ndup*(icol+1)] = xfact*np.eye(ndup)
            if ordering == 'BTLR':
                jy += 1
                if jy == ny:
                    jy = 0
                    jx += 1
            elif ordering == 'LRTB':
                jx += 1
                if jx == nx:
                    jx = 0
                    jy -= 1
                
        if ordering == 'BTLR':
            iy += 1
            if iy == ny:
                iy = 0
                ix += 1
        elif ordering == 'LRTB':
            ix += 1
            if ix == nx:
                ix = 0
                iy -= 1

    return M


def smoothmat_check(fault,M,ix,iy):
    ndup = fault.ntw*2
    nx = fault.sbf_nx
    ny = fault.sbf_ny
    if fault.Ordering == 'BTLR':        
        irow =  (ix*ny+iy)*ndup
    elif fault.Ordering == 'LRTB':        
        irow =  (iy*ny+ix)*ndup
    mystring = " ix's "        
    for jx in range(max(ix-1,0),min(ix+2,fault.sbf_nx)):
        mystring += '  %03d     ' % jx
    print mystring
    for jy in range(max(iy-1,0),min(iy+2,fault.sbf_ny)):
        mystring = "iy=%02d" % jy          
        for jx in range(max(ix-1,0),min(ix+2,fault.sbf_nx)):
            jcol = (jx*ny+jy)*ndup
            mystring += '  %5.2f  ' % M[irow,jcol]
        print mystring     
    return
    

def point_in_poly(x,y,poly):
    '''
    faulDetermine if a point (x,y) is inside a polygon.
    poly is a list of (x,y,z) tuples, *but z is not used*.
    This seems simpler than the 'Casting' method included below. Seems 
    to work, too.
    '''
    n = len(poly)
    inside =False

    p1 = poly[0]
    for i in range(n+1):
        p2 = poly[i % n]
        if y > min(p1.y,p2.y):
            if y <= max(p1.y,p2.y):
                if x <= max(p1.x,p2.x):
                    if p1.y != p2.y:
                        xinters = (y-p1.y)*(p2.x-p1.x)/(p2.y-p1.y)+p1.x
                    if p1.x == p2.x or x <= xinters:
                        inside = not inside
        p1.x,p1.y = p2.x,p2.y

    return inside

## 
# 


##  
def InsidePolygon(polygon, p):
   '''
   Determine if a point is inside a given polygon or not
   Polygon is a list of (x,y) pairs. This fuction
   returns True or False.  The algorithm is called
   "Ray Casting Method".
   '''
   angle = 0.0
   n = len(polygon)
 
   for i, (h, v) in enumerate(polygon):
      p1 = Point(h - p.h, v - p.v)
      h, v = polygon[(i + 1) % n]
      p2 = Point(h - p.h, v - p.v)
      angle += Angle2D(p1.h, p1.v, p2.h, p2.v);
 
   if abs(angle) < mt.pi:
      return False
 
   return True
 
def Angle2D(x1, y1, x2, y2):
   theta1 = mt.atan2(y1, x1)
   theta2 = mt.atan2(y2, x2)
   dtheta = theta2 - theta1
   while dtheta > mt.pi:
      dtheta -= 2.0 * mt.pi
   while dtheta < -mt.pi:
      dtheta += 2.0 * mt.pi
 
   return dtheta
##

##
def neic2fault(filename,origin,xdr=0.,Verbose=False,velmod=None,
                NEIC_potency=False):
    '''
    Reads an NEIC finite fault model and converts it to a fault object.
        filename    Name of NEIC fintite fault file
        origin      An obspy origin object for earthquake hypocentre
        xdr         Directivity parameter: 
                            0. for bilateral rupture, 
                            1. for unilateral along strike
                           -1. for unilateral along anti-strike
        Verbose     True for lots of info printed out
    Sets cumslip to slip in meters, subfault potency as moment/slip, and 
    subfault.slip_rate to ruptime, half duration, moment).
    Slip is resolved along the fault coordinates, with x along stike and y updip
    '''
    proj = Geod(ellps='WGS84')
    hypo = Point(origin.longitude,origin.latitude,0.001*origin.depth)
    lines = open(filename,'r').readlines()
    lpns = []  #
    lpts = []
    dpps = []
    #print lines[1].split()
    nx = int(lines[1].split()[4])
    dx = float(lines[1].split()[6].rstrip('km'))
    ny = int(lines[1].split()[8])
    dy = float(lines[1].split()[10].rstrip('km'))
    if Verbose:
        print 'Reading NEIC file %s' % filename
        print '\tNx,Dx=%d,%4.1f,  Ny,Dy=%d,%4.1f' % (nx,dx,ny,dy)
    for line in lines[4:9]:
        lon,lat,dep = [float(x) for x in line.split()]
        lpns.append(lon)
        lpts.append(lat)
        dpps.append(dep)
    az,bz,dst = proj.inv(lpns[0],lpts[0],lpns[1],lpts[1])
    dst *= 0.001
    strike = az
    length = dst
    az,bz,dst = proj.inv(lpns[1],lpts[1],lpns[2],lpts[2])
    dst *= 0.001
    dip = mt.atan2(dpps[2]-dpps[1],dst)
    width = dst/mt.cos(dip)
    dip = mt.degrees(dip)
    if Verbose:
        print 'neic2ffi: Reading NEIC file %s' % filename
        print '\t Fault info (strike,dip,rake, length,width):'
        print 'Nx,Dx=%d,%4.1f,  Ny,Dy=%d,%4.1f' % (nx,dx,ny,dy)
    length = dx*nx
    width  = dy*ny
    fault = Fault(fault_spec=[hypo,strike,dip,0.,xdr,length,width],velmod=velmod)
    fault.sbf_len = dx
    fault.sbf_wid = dy
    fault.sbf_nx  = nx
    fault.sbf_ny  = ny
    fault.ntw = 1
    fault.Tessellation = 'Rectangular'
    fault.Ordering = 'LRTB'

    sbfs = []
    index = 0
    dwdt = fault.sbf_wid*mt.cos(mt.radians(dip)) # SF horizonatl width
    ddip = fault.sbf_wid*mt.sin(mt.radians(dip)) # SF vertical depth
    for line in lines[10:]:
        #Lat. Lon. depth slip rake strike dip t_rup h_dur mo
        if len(line.split()) == 11:
            ullat,ullon,uldep,sfslip,sfrake,sfstrike,sfdip,sftr,sft1,sft2,sfmo = \
                        [float(x) for x in line.split()]
            # Set the half-duration to average of forward and rear half-widths
            sfhd = (sft1,sft2)  
        else:
            ullat,ullon,uldep,sfslip,sfrake,sfstrike,sfdip,sftr,sfhd,sfmo = \
                        [float(x) for x in line.split()]
            # Set triangle to be symmetric
            sft1 = sfhd
            sft2 = sfhd
        # NEIC uses upper left corner, so convert to subfault center
        lon,lat,baz     = proj.fwd(ullon,ullat,strike,1.e3*0.5*fault.sbf_len)
        sflon,sflat,baz = proj.fwd(lon,lat,strike+90.,1.e3*0.5*dwdt)
        sfdep = uldep + 0.5*ddip
        if velmod is None:
            sbfs.append(SubFault(strike=sfstrike,dip=sfdip,length=dx,width=dy,
                    cntr=Point(sflon,sflat,sfdep),index=index))
        else:
            sbfs.append(SubFault(strike=sfstrike,dip=sfdip,length=dx,width=dy,
                    cntr=Point(sflon,sflat,sfdep),index=index,
                    velmod=velmod,raypaths=fault.raypaths))
        # Convert slip from cm to meters
        sbfs[-1].cumslip = (0.01*sfslip*mt.cos(mt.radians(sfrake)),\
                            0.01*sfslip*mt.sin(mt.radians(sfrake)))
        if sfslip == 0.:
            print "WARNING! Can't set potency for subfault %d where slip=0" % index
        else:
            neic_potency = sfmo*1.e-5/sfslip # Convert slip->m, moment->Nm
            if not velmod is None:
                if not np.isclose(neic_potency,sbfs[-1].potency,rtol=3.e-2):
                    print 'Subdfault %03d: Potency from velmod(depth=%3d): %8.2e' % (index,
                                                        sbfs[-1].cntr.z,sbfs[-1].potency)
                    print '               Inferred from NIEC Moment/slip: %8.2e' % neic_potency
                    if NEIC_potency:
                        print 'Using NEIC'
            if velmod is None or NEIC_potency:
                sbfs[-1].potency = neic_potency 
        ####
        
        sbfs[-1].hypo_deltaz   = sbfs[-1].cntr.z - fault.hypo.z
        az,baz,dist = proj.inv(hypo.x,hypo.y,sbfs[-1].cntr.x,sbfs[-1].cntr.y) 
        sbfs[-1].hypo_azimuth  = az
        sbfs[-1].hypo_distance = dist*0.001 # in km

        # Set subfault slip
        trmx = 1000.
        times = [0.,sftr,sftr+sft1,sftr+sft1+sft2,trmx]
        #
        slipx_rates = [0.,0.,2.*sbfs[-1].cumslip[0]/(sft1+sft2),0.,0.]
        slipy_rates = [0.,0.,2.*sbfs[-1].cumslip[1]/(sft1+sft2),0.,0.]
        sbfs[-1].slip_rate = (UnivariateSpline(times, slipx_rates, s=0., k=1),
                              UnivariateSpline(times, slipy_rates, s=0., k=1))
        sbfs[-1].slip = (sbfs[-1].slip_rate[0].antiderivative(),
                         sbfs[-1].slip_rate[1].antiderivative())           
        #
        if Verbose:
            print sbfs[-1]
        #print sbfs[-1].index,sbfs[-1].cumslip,(sbfs[-1].slip[0]([trmx])[0],sbfs[-1].slip[1]([trmx])[0])
        index += 1
    fault.subfaults = sbfs
    return fault
##
 
def main(argv=None):
    from obspy.core.event import Event, Origin, Magnitude, NodalPlane
    from obspy.core import read, UTCDateTime
        
    hypo = Point(142.861,38.1035,18.)

       
    nc = 'satake2013.nc'   
    dt = 30.

    # Don't know if these are right
    xpnd=0.65
    strike = 195.5
    dip = 0.
    length = 450. #30
    width  = 210  #14
    h_top  = 0.145700-0.02
    xdr    = 0.053668+0.005
    if False:
        nx = 9
        ny = 4
        ny_fine = 2
        nx_fine = 4
    else:
        ny = 14
        nx = 1
        nx_fine = 10
        ny_fine = 7
    fault1 = Fault(fault_spec=[hypo,strike,dip,h_top,xdr,length,width])
    fault2 = Fault(fault_spec=[hypo,strike,dip,h_top,xdr,length,width])
    fault  = Fault(fault_spec=[hypo,strike,dip,h_top,xdr,length,width])
    fault1.rectangular_tessellation(nx=nx,ny=ny,Fix_hypo=True)
    fault2.rectangular_tessellation(nx=2*nx,ny=2*ny,Fix_hypo=True)
    sbfs = []
    index = 0
    for idx in range(nx):
        for idy in range(ny):
            index1 = ny*idx+idy
            if idx < nx_fine and idy >= ny_fine:
                for idx2 in range(2*idx,2*idx+2):
                    for idy2 in range(2*idy,2*idy+2):
                        index2 = 2*ny*idx2+idy2        
                        sbfs.append(fault2.subfaults[index2])
            else:
                sbfs.append(fault1.subfaults[index1])
            sbfs[-1].index = index
            index +=1
    fault.subfaults = sbfs
   
    fig = plt.figure(figsize=(7.5, 5.))  
    fig.subplots_adjust(left=0.01,right=0.99,bottom=0.1,top=0.95,hspace=0.05,wspace=0.05)
    dll = 0.00833333333333
    rlon = np.arange(140.0,146.01,dll)
    rlat = np.arange( 35.5, 42.01,dll)
    #rlon = np.arange(141.0,143.51,0.00833333333333)
    #rlat = np.arange( 36.5, 40.01,0.00833333333333)
    x,y = np.meshgrid(rlon,rlat)

    
    ax = fig.add_subplot(1, 1, 1)
    m = Basemap(rlon.min(),rlat.min(),rlon.max(),rlat.max(),
                        projection='merc',resolution='h')
    m.drawmapboundary() 
    m.drawcoastlines()
    m.fillcontinents(color='coral',lake_color='aqua')                            
    m.drawparallels(np.arange(-90.,90.,1.),labelstyle='+/-')
    m.drawmeridians(np.arange(-180.,180.,1.))#,labelstyle='+/-')
    for sbf in fault1.subfaults:
        #x,y = m(sbf.cntr.x,sbf.cntr.y)
        #plt.plot(x,y,marker='*',color='red',zorder=200)
        sbf.map_plot(ax,m,zorder=200)
    for sbf in fault2.subfaults:
        #x,y = m(sbf.cntr.x,sbf.cntr.y)
        #plt.plot(x,y,marker='*',color='green',zorder=200)
        sbf.map_plot(ax,m,zorder=200)

    plt.savefig('cosbfs.pdf')
    plt.show()

    #
    return fault
 

#####
if __name__ == "__main__":
    fault = main()
    sys.exit(0)

