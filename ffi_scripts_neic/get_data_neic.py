#!/usr/bin/env python
import getopt
import sys
import math as mt
import numpy as np
import pylab as plt
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib import colors,colorbar
from matplotlib.patches import Polygon
import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
from obspy.core import read,UTCDateTime,Trace,Stream
from obspy.core.event import Pick
import os,warnings
import pickle
source_dir = os.environ['HOME'] + '/RSES-u5484353/Academics/PhD_research/02_HistoricalTsunamis/1992_Flores/20041111_Alor_earthquake/seismic_inversion/philscode'
sys.path.append(source_dir)
from ffi import kiksyn,read_velmod,delay,thetasr
from surfw_lib import surfw_excite,surfw_calc,surfw_phv,taper,filter,readhrv,surfw_matrix
from surfw_lib import read_gdm52,read_de06
from obspy.core.inventory import Inventory,Station,Network,Channel
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException
from obspy.signal.rotate import rotate_ne_rt,rotate2zne
from obspy.geodetics import locations2degrees,kilometer2degrees
from obspy.geodetics.base import gps2dist_azimuth
import matplotlib.gridspec as gridspec
from scipy.optimize import nnls
from fault8 import Point,Fault,rectsmooth,SubFault,velmod_findparam,neic2fault
from scipy.interpolate import UnivariateSpline
from pyproj import Geod

def get_isctimes(filename,otime):
    picks = {}
    for line in open(filename,'r'):
        if line[18:21] == ' P ' or line[18:21] == ' S ':
            flds = line.split()
            sta   = flds[0]
            phase = line[19]
            hour,minute = [int(x) for x in line[28:37].split(':')[:2]]
            second = float(line[28:37].split(':')[2])
            time = UTCDateTime(otime.year,otime.month,otime.day,
                                hour,minute,second)
            if not sta in picks.keys():
                picks[sta] = {phase:time}
            else:
                if phase in picks[sta].keys(): continue
                picks[sta][phase] = time
    return picks
            
            
def surfsynth_fault(fault,th,otime,st,frq4,gvel_win):
    # Calculate phase velocities, etc from epicenter to stations
    ns = len(st)
    stas = []
    stlo = np.zeros(len(st))
    stla = np.zeros(ns)
    dt   = np.zeros(ns)
    nt   = np.zeros((ns,), dtype=np.int) 
    ib   = np.zeros((ns,), dtype=np.int)
    ntm = 0 
    for i,tr in enumerate(st):
        stas.append(tr.stats.station)
        stlo[i] = tr.stats.coordinates['longitude']
        stla[i] = tr.stats.coordinates['latitude']
        nt[i]   = tr.stats.npts
        if nt[i] > ntm: ntm = nt[i]
        dt[i]   = tr.stats.delta
        if tr.stats.channel[-1] == 'Z':
            ib[i] = 7
        elif tr.stats.channel[-1] == 'T':
            ib[i] = 6
    npw,lcent,td,c,u,q,c1,u1,q1 = surfw_phv(fault.hypo.x,fault.hypo.y,stas,
                                            stlo,stla,ib,nt,dt,2*ntm)
    #
    sto = Stream()
    d = []
    for itr,tr in enumerate(st):
        to = Trace(header=tr.stats)
        '''
        # Calculate indices for group velocty window
        dist,azm,bazm = gps2DistAzimuth(fault.hypo.y,fault.hypo.x,
                                        tr.stats.coordinates['latitude'],
                                        tr.stats.coordinates['longitude'])
        tbeg,tend = (dist*.001/gvel_win[1],dist*.001/gvel_win[0])
        # TSUM 8851.43284391 1609.35142617 2723.51779813 8851.43284391
        # ANMO 8213.81084094 1493.4201529 2527.3264126 8213.81084094
        ibeg,iend = (int(tbeg/tr.stats.delta+.5),int(tend/tr.stats.delta+.5))
        print dist*.001,tr.stats.station,tbeg,tend,ibeg,iend
        #if ib[itr] == 6:
        #    ibeg=0
        #    iend=tr.stats.npts
        to.stats.starttime = otime+float(ibeg)*tr.stats.delta
        to.stats.npts      = iend-ibeg+1
        #to.data            = np.zeros(to.stats.npts)
        '''
        # Set wave type, Rayleigh =2, Love = 1
        if tr.stats.channel[-1] == 'Z':
            ityp = 2
        elif tr.stats.channel[-1] == 'T':
            ityp = 1
        df   = 1.0/(2.*(lcent[itr]-1)*tr.stats.delta)
        stlo = tr.stats.coordinates['longitude']
        stla = tr.stats.coordinates['latitude']
        syn = np.zeros(tr.stats.npts)
        for sbf in fault.subfaults:
            elo = sbf.cntr.x
            ela = sbf.cntr.y
            edp = sbf.cntr.z
            stk = sbf.strike
            dip = sbf.dip
            ruptimes = sbf.ruptimes()[0:3]
            sftr = ruptimes[0]
            sfhd = 0.5*(np.diff(ruptimes).sum())
            rak = mt.degrees(mt.atan2(sbf.cumslip[1],sbf.cumslip[0]))
            slp = mt.hypot(sbf.cumslip[1],sbf.cumslip[0])
            # The 1.e-18 is from Kikuchi (1.e-25) and dyne-cm to Nm
            bmom = sbf.potency*slp*1.e-18 
            # I don't see we one would subtrat the rupture time from start time
            #xt  = sftr  # Subfault rupture time 
            #tstart = (tr.stats.starttime - otime) - xt
            tstart = (tr.stats.starttime - otime) - sftr
            gs = surfw_calc(ela,elo,stla,stlo,edp,sfhd,stk,dip,rak,\
                           npw[itr],lcent[itr],td[:,itr],df,frq4,tstart,\
                           c[:,itr],u[:,itr],q[:,itr],\
                           c1[:,itr],u1[:,itr],q1[:,itr],ityp,len(c[:,itr])) 
            syn += bmom*gs[0:tr.stats.npts]

        to.data = syn
        sto.append(to)
        d += list(to.data)
    return sto,np.array(d) 
        

def bodysynth_fault(fault,stobs,toff):
      '''
      Returns a Nx2 matrix (i.e., 2 column vectors, each containing subfault 
      Green's functions for rake+-45 deg, calculated by method bodywave for 
      time ruptime after the origin time.
      Input:
          phase:    currently only 'P' allowed
          ruptime:  time in seconds after origin, must equal a time in 
                    subfault's ruptime array, or 'None' is returned.
          toff:     arrival time - trace onset time, in seconds.
      Output:
          GF[n,2], Where n is the sum of the lenths of data in the subfaults 
                   Green's function for this phase, with each trace shifted by
                   ruptime seconds.
          None     If ruptime is not equal to an element of the subfault's 
                   ruptime array.
      '''
      stgn = {}
      chan = {'P':'BHZ', 'SH':'BHT'}
      for sbf in fault.subfaults:
        if not hasattr(sbf,'rake'):
            raise Exception, 'Subfault must have rake set in order to call bodysynth_fault'
        times   = sbf.ruptimes()
        for jwn in range(0,fault.ntw):
            ruptime = times[jwn] 
            risetm1 = times[jwn+1]-times[ jwn ] 
            risetm2 = times[jwn+2]-times[jwn+1] 
            sliprate = ((sbf.slip_rate[0]([ruptime+risetm1]))[0],
                        (sbf.slip_rate[1]([ruptime+risetm2]))[0])
            sliprate = sbf.cumslip
            for jrk in range(0,2):
                for phase in stobs.keys():
                    if phase != 'P' and phase != 'SH':
                        continue
                    if not phase in stgn.keys():
                        stgn[phase] = Stream()
                    if phase not in sbf.GreenFuncs.keys():
                        raise Exception, "No Green's functions for phase %s" % phase
                    for tro,trc in zip(stobs[phase],sbf.GreenFuncs[phase][jrk]):
                        # Either initialize or select stgn trace
                        if jwn == 0 and jrk == 0 and sbf.index == 0:
                            stgn[phase].append(Trace(header=tro.stats,
                                                     data=np.zeros(tro.stats.npts)))
                            trgn = stgn[phase][-1]
                        else:
                            trgn = stgn[phase].select(station=tro.stats.station,
                                                        channel=chan[phase])[0]
                        # Calculate time shift as a sum of four terms:
                        #    (1) toff, the pre-arrival time window (normally 
                        #        chosen to be compatible with observed trace)
                        #    (2) delay, of subfault ray at receiver, measured w.r.t.
                        #        hypocentral ray (i.e., infinite rupture velocity)
                        #    (3) ruptime, subfault rupture time, w.r.t. origin time 
                        #    (4) the half-duration, due align triangle start
                        time_shift = ruptime + toff - trc.stats.delay + risetm1
                        ishift = int(time_shift/trc.stats.delta +  0.4)
                        if (tro.stats.station == 'RSSD' or tro.stats.station == 'PMSA') and \
                            jrk == 0 and (sbf.index == 0 or sbf.index == len(fault.subfaults)-1) and \
                            jwn == 0:
                            print tro.stats.station,phase,sbf.index,trc.stats.delay
                        '''
                        ## For compatibility with HKT
                        dly = trc.stats.delay
                        dt  = trc.stats.delta
                        if ruptime-dly+toff >= 0.:
                            ishift = int((ruptime-dly+toff)/dt+0.5) 
                        else:
                            ishift = int((ruptime-dly+toff)/dt-0.5) 
                        ishift -= 1
                        '''
                        if ishift < 0.:
                            raise Exception,('Ruptime+Toff(%g) too small ' + \
                            'to accommodate subfault time delay (%g)') %\
                            (ruptime+toff,trc.stats.delay)
                        if jrk == 0:  # Slip along direction rake=-45
                            slip =  sliprate[0]*mt.cos(mt.radians(sbf.rake-45.)) \
                                   +sliprate[1]*mt.sin(mt.radians(sbf.rake-45))
                        else:         # Slip along direction rake+45.
                            slip =  sliprate[0]*mt.cos(mt.radians(sbf.rake+45.)) \
                                   +sliprate[1]*mt.sin(mt.radians(sbf.rake+45.))
                        #slip *= risetim
                        
                        if ishift == 0:
                            trgn.data += slip*trc.data[:]
                        else:
                            trgn.data[ishift:trc.stats.npts] += \
                                         slip*trc.data[0:-ishift]
      return stgn  

      
def bodywave(sbf,st,phase,t1,t2,rake=90.,velmod=None,frq4=None):
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
          t1,2      = the half-durations of an asymmetric, triangular source time 
                      function
          toff      = the offset of the start of trace w.r.t. phase arrival time. 
                      Must be large enough to accomodate arrivals earlier than the
                      corresponding pahse from the hypocentre (usually a few sec).
          rake      = rake angle in degrees, synthetics calculated for rake+/-45.
          frq4      = if set to a numpy array of lngth 4, traces will be filtered 
                      using frequency domain cosine tapers in the frequency ranges
                      frq4[0]-frq4[1] (high-pass) and frq4[2]-frq4[3] (low pass).
                      
       Output:
          sbf.GreenFuncs[phase][0:1] is set to a list of 2 Stream objects, 
       one for each rake direction. The delay in arrival time at each receiver 
       of rays from the subfault. as measured w.r.t. the hypocentral ray (i.e., 
       assuming infinite rupture velocity) is stored in tr.stats.delay, where 
       tr is a trace element of each stream.  
      '''
      ### Check that velmod is supplied, and calculate moment parameters - only once when sbf.velmod = None
      if sbf.velmod == None and velmod == None:
         raise Exception, 'Subfault needs velmod for body wave calculation'
      elif sbf.velmod == None:
         sbf.velmod = velmod
      velmod = sbf.velmod
      z0 = sbf.cntr.z-sbf.hypo_deltaz
      scale_factor = 1.e-6*1.e7/1.e25 # Convert microns->metre, dyne-cm to nM, 
                                      # correct for HKT 1.e25)
      scale_factor *= 1.e6            # back to microns

      if sbf.raypaths == None:
          raise Exception, 'Subfault raypaths must be set for bodywave calculation'
      elif not sbf.raypaths.has_key(phase):
          raise Exception, 'Subfault raypaths has no phase %s' % phase
      elif phase == 'P':  # Ray description parameters for P 
          ib = 1    # P wave incident on receiver
          ic = 1    # Vertical component at receiver
          v_source = velmod_findparam(sbf.velmod,'alpha',z0)
      elif phase == 'SH': # Ray description parameters for SH
          ib = 3    # SH wave incident on receiver
          ic = 1    # ignored for ib == 3
          v_source = velmod_findparam(sbf.velmod,'beta',z0)
      else:
          raise Exception, 'Unknown phase ',phase
      #
      if rake < -180. or rake > 180.:
          raise Exception, 'Expected -180.<rake<180., not %g' % rake
      else:
          sbf.rake = rake          
      ## Loop over traces in stream
      stbdy = [Stream(), Stream()]
      for tr in st:
        #nt = len(tr.data)
        print tr
        print 2**mt.ceil(mt.log(tr.stats.npts,2)), tr.stats.npts
        nt = 2**mt.ceil(mt.log(tr.stats.npts,2))
        dt = tr.stats.delta
        gcarc = tr.stats.gcarc
        azimuth = tr.stats.azimuth
         # Calculate ray parameters, slowness p and geometerical spreading g
        gp,ps,v0 = sbf.raypaths[phase]
        idelt = int(gcarc)
        g=gp[idelt-1]+(gcarc-idelt)*(gp[idelt]-gp[idelt-1])
        p=ps[idelt-1]+(gcarc-idelt)*(ps[idelt]-ps[idelt-1])
        # Store delay w.r.t hypocenter in trc.stats
        tr.stats.delay = delay(azimuth,p,sbf.hypo_azimuth,\
                                sbf.hypo_distance,sbf.hypo_deltaz,v_source)
        # Calculate synthetics
        #print tr.stats.station,nt,len(tr.data),gcarc,azimuth
        syn_m45 = kiksyn(nt,dt,ib,ic,sbf.strike,sbf.dip,rake-45.,sbf.cntr.z,
                        azimuth,p,g,t1,t1+t2,velmod=velmod,frq4=np.array(frq4))      
        syn_p45 = kiksyn(nt,dt,ib,ic,sbf.strike,sbf.dip,rake+45.,sbf.cntr.z,
                        azimuth,p,g,t1,t1+t2,velmod=velmod,frq4=np.array(frq4))
        syn_m45 *= sbf.potency*scale_factor
        syn_p45 *= sbf.potency*scale_factor
        stbdy[0].append(Trace(header=tr.stats,data=syn_m45[0:tr.stats.npts]))
        stbdy[1].append(Trace(header=tr.stats,data=syn_p45[0:tr.stats.npts]))
      sbf.GreenFuncs[phase] = stbdy        


def plot_datacomp(ax,stobs,phases,stgrn,origin,rowcol=None,
                    yscale=0.5,ttick=100.,timing='',fontsize=6,tbuf=.15):
    #phases = stobs.keys()
    torg = origin.time
    ntr = 0
    trng = 0.
    print phases,stobs.keys()
    for phase in phases:
        if not phase in stobs.keys(): continue
        ntr += len(stobs[phase])
        for tr in stobs[phase]:
            trng = max(trng,tr.stats.npts*tr.stats.delta)
    if rowcol == None: # Try to make a sensible choice for nrow,col
        nrow = int(ntr/2.5+0.5)
        ncol = ntr/nrow
        if nrow*ncol < ntr:
            nrow += 1
    else:
        nrow = rowcol[0]
        ncol = rowcol[1]
    irow = 0
    icol = 0
    tof1 = tbuf*trng
    tmin = None
    for phase in phases:
        if not phase in stobs.keys(): continue
        for idx,tr in enumerate(stobs[phase]):
            dist,azm,bazm = gps2dist_azimuth(origin['latitude'],origin['longitude'],
                                            tr.stats.coordinates['latitude'],
                                            tr.stats.coordinates['longitude'])
            dist *= 0.001
            toff = 0.
            if timing == 'origin_time':
                toff = tr.stats.starttime-torg
            ttoff = tof1+toff+0.5*(trng-tr.times()[-1])
            ax.hlines([-irow],tr.times()[0]+ttoff,tr.times()[-1]+ttoff,
                       colors=['gray'],linestyles=['dotted'])
            if stgrn != None:
                tsyn = stgrn[phase][idx]
                tsyoff = 0.
                if timing == 'origin_time':
                    tsyoff = tsyn.stats.starttime-torg
                #if phase == 'P':
                #    print tr.stats.station,tsyn.data.max(),tr.data.max()
                if phase == 'P':
                    ax.plot(ttoff+tsyn.times()+tsyoff-toff,-irow + 1.*yscale*tsyn.data/tr.data.max(),color='red')           
                else:
                    ax.plot(ttoff+tsyn.times()+tsyoff-toff,-irow + yscale*tsyn.data/tr.data.max(),color='red')           
            ax.plot(ttoff+tr.times(),-irow + yscale*tr.data/tr.data.max(),color='black')
            if icol == 0: # Set tmin
                if tmin is None:
                    tmin = ttoff+tr.times()[0]
                else:
                    tmin = min(tmin,ttoff+tr.times()[0])
            if hasattr(tr.stats,'pick'):
                pick_time = (tr.stats.pick.time - tr.stats.starttime)+ttoff
                ax.vlines([pick_time],-irow-0.5,-irow+0.5,colors='red')
            #ax.vlines([dist/gvel_win[phase][0],dist/gvel_win[phase][1]],
            #          -irow-0.5,-irow+0.5,colors=['green','green'])
            delta = kilometer2degrees(dist)
            txtoff = tof1+toff #-tbuf*trng
            ax.text(txtoff,          -irow-.25,'$\Delta=$%2.0f' % (delta),\
                    fontsize=fontsize)
            ax.text(txtoff+trng,-irow-.25,'$\phi=$%3.0f' % (azm),\
                    fontsize=fontsize,horizontalalignment='right')
            ax.text(txtoff,          -irow+.25 ,\
                    '%s %s' % (tr.stats.station,phase),fontsize=fontsize)
            ax.text(txtoff+trng,-irow+.25 ,'%4.0f$\mu$' % (tr.data.max()),\
                    fontsize=fontsize,horizontalalignment='right')
            if irow == nrow-1:
               icol += 1
               irow = 0
               tof1 += (2.*tbuf+1.)*trng
            else:
                irow += 1
    plt.setp( ax.get_yticklabels(), visible=False)
    ax.set_ylim((-(nrow-0.5),0.5))  
    ax.set_xlim((tmin,tmin+ncol*((2.*tbuf+1.)*trng))) 
    tmin,tmax =  ax.get_xlim()
    ax.set_xticks(np.arange(tmin,tmax,ttick))
    x  = tmin+(ncol-0.5)*((2.*tbuf+1.)*trng)
    xs = [x-0.5*ttick,x-0.5*ttick,x-0.5*ttick,x+0.5*ttick,x+0.5*ttick,x+0.5*ttick]
    y  = -(nrow-1.)
    ys = [y-0.15,y+0.15,y,y,y+0.15,y-0.15]
    ax.plot(xs,ys)
    ax.text(x,y-0.15,'%g sec' % ttick,ha='center',va='center')
    return
    




def plot_sar(sarqt,fault,sar_mat,x,rakes):
    # Plot data and rsult
    fig = plt.figure(figsize=(20.,20. / 1.618))
    ZMIN,ZMAX = (-1.5,1.5)
    norm = colors.Normalize(vmin=ZMIN, vmax=ZMAX)

    # Plot data    
    ax_sar = plt.subplot(221)
    ax_sar.scatter(sarqt[:,0],sarqt[:,1],c=sarqt[:,2],norm=norm)

    #xsol[0][0:12] = 0.
    # Plot prediction
    ax_pred = plt.subplot(223)
    pred = np.dot(sar_mat,x)
    ax_pred.scatter(sarqt[:,0],sarqt[:,1],c=pred,norm=norm)
    
    # Plot fault slip
    ax_flt = plt.subplot(222)
    for i,sbf in enumerate(fault.subfaults):
        slipx = x[ 2*i ]*mt.cos(mt.radians(rakes[0])) + \
                x[2*i+1]*mt.cos(mt.radians(rakes[1]))
        slipy = x[ 2*i ]*mt.sin(mt.radians(rakes[0])) + \
                x[2*i+1]*mt.sin(mt.radians(rakes[1]))
        sbf.cumslip = (slipx,slipy)
    m = fault.map_plot(ax_flt,xpnd=.75,field='cumslip',cmap=cm.jet,norm=10.)#,field='cumslip',cmap=cmap,norm=1.)
    
    cax = fig.add_axes([0.5, 0.1, 0.025, 0.35])
    colorbar.ColorbarBase(ax=cax,label='Range/-Vertical Displacement (m)',norm=norm)
    
    dax = fig.add_axes([0.7, 0.1, 0.25, 0.4])
    rsd = sarqt[:,2] - pred
    dax.hist(rsd)
    plt.show()

      

         
def getobs(mseed_filename,client,event,phases,frq4,windows,stas,stalocs,picks=None,
            delta_T={'P':1.,'SH':1.,'R':10.,'L':10.},taper=None,adjtime=None):
    # Connect to arclink server
    #st = read('../mseed/mini.seed')
    org = event.preferred_origin()
    if org is None:
        org = event.origins[0]
    st = read(mseed_filename)
    stobs  = {'params':{'filter':frq4, 'windows':windows}}
    syn    = {}
    torg   = org.time
    trngmx = 3600.
    invout = None
    # First do any requried time adjustments 
    if not adjtime is None:
        for tr in st:
            if not tr.stats.station in adjtime.keys(): 
                continue
            print 'Adjusting time for station %s by %g secs' % \
                (tr.stats.station,adjtime[tr.stats.station])
            tr.stats.starttime -= adjtime[tr.stats.station]
            
    for phase in phases:
        if not phase in stas.keys(): continue
        stobs[phase] = Stream()
        for sta in stas[phase]:
            # If this is a body wave phase find the pick - skip if none found
            if phase == 'P' or phase == 'SH':
                sta_pick = None
                # If no picks supplied then get them from events
                if picks is None:
                    for pick in event.picks:
                        if pick.phase_hint == phase[0:1] and \
                            pick.waveform_id.station_code == sta:
                                sta_pick = pick
                                break
                else: # Get them from picks - e.g. returned by get_isctimes
                    if sta in picks.keys() and phase[0:1] in picks[sta]:
                       sta_pick = Pick() 
                       sta_pick.time = picks[sta][phase[0:1]]
                if sta_pick is None: 
                    print 'No %s pick found for station %s - skipping' % (phase,sta)
                    continue
                
            # Set location code if prescribed, otherwise use '00' (preferred)
            if sta in stalocs.keys():
                loc = stalocs[sta]
            else:
                loc = '00'
            # Select the channels for this station - skip if none found
            chans = st.select(station=sta,location=loc)
            if len(chans) == 0: # if nothing for loc='00', try also with ''
                loc = ''
                chans = st.select(station=sta,location=loc)
            if len(chans) == 0:                
                print 'No channels found for %s' % sta
                continue
            try:
                inv = client.get_stations(network=chans[0].stats.network,
                                          station=sta,location=loc,
                                          starttime=torg,endtime=torg+100.,
                                          level='response')
            except Exception as e:
                warnings.warn(str(e))
                print 'FDSNWS request failed for trace id %s - skipping' % sta
                continue
            try: 
                coordinates = inv[0].get_coordinates(chans[0].id)
            except:
                print 'No coordinates found for station %s, channel %s' % \
                            (sta,chans[0].id)
                continue
            dist,azm,bazm = gps2dist_azimuth(org['latitude'],org['longitude'],
                                            coordinates['latitude'],
                                            coordinates['longitude'])
            gcarc = locations2degrees(org['latitude'],org['longitude'],
                                            coordinates['latitude'],
                                            coordinates['longitude'])
            if phase == 'R' or phase == 'P': # Rayleigh or P wave
                try:
                    tr = st.select(station=sta,component='Z',location=loc)[0]
                except IndexError:
                    print 'No vertical for %s:%s' % (sta,loc)
                    continue
                try:
                    inv = client.get_stations(network=tr.stats.network,
                                              station=sta,
                                              channel=tr.stats.channel,
                                              location=loc,
                                              starttime=torg,
                                              endtime=torg+100.,
                                              level='response')
                except Exception as e:
                    warnings.warn(str(e))
                    print 'FDSNWS request failed for trace id %s - skipping' % tr.id
                    continue
                tr = tr.copy()
                tr.stats.response    = inv[0].get_response(tr.id,torg)
                tr.stats.coordinates = inv[0].get_coordinates(tr.id)
                tr.remove_response(pre_filt=frq4[phase],output='DISP')
                tr.stats.gcarc = gcarc 
                tr.stats.azimuth = azm
                #t1 = minv[0].get_responeax(tr.stats.starttime,t+dist/rvmax)
                #t2 = min(tr.stats.endtime  ,t+dist/rvmin)
                t1 = max(torg,tr.stats.starttime)
                t2 = min(torg+trngmx,tr.stats.endtime)
                tr.trim(starttime=t1,endtime=t2)
                decim = int(0.01+delta_T[phase]/tr.stats.delta)
                ch = inv.select(station=sta,channel=tr.stats.channel)[0][0][0]
                print tr.id,' ',tr.stats.sampling_rate,' decimated by ',decim,\
                        'sensitivity=',ch.response.instrument_sensitivity.value
                if tr.stats.starttime-torg < 0.:
                    tr.trim(starttime=torg)
                tr.decimate(factor=decim,no_filter=True)
                tr.data *= 1.e6 # Convert to microns
            elif phase == 'L' or phase == 'SH': # Love or SH wave
                if len(chans.select(component='E')) != 0:
                    try:
                        tr1a = st.select(station=sta,component='E',location=loc)[0]
                        tr2a = st.select(station=sta,component='N',location=loc)[0]
                    except:
                        print 'Station %s does not have 2 horizontal componets -skipping' % sta
                        continue
                elif len(chans.select(component='1')) != 0:
                    try:
                        tr1a = st.select(station=sta,component='1',location=loc)[0]
                        tr2a = st.select(station=sta,component='2',location=loc)[0]
                    except:
                        print 'Station %s does not have 2 horizontal componets -skipping' % sta
                        continue
                tr1 = tr1a.copy()
                tr1.data = tr1a.data.copy()
                tr2 = tr2a.copy()
                tr2.data = tr2a.data.copy()
                ch1 = inv.select(station=sta,channel=tr1.stats.channel)[0][0][0]
                ch2 = inv.select(station=sta,channel=tr2.stats.channel)[0][0][0]
                tr1.stats.response = ch1.response
                tr1.remove_response(pre_filt=frq4[phase],output='DISP')
                tr2.stats.response = ch2.response
                tr2.remove_response(pre_filt=frq4[phase],output='DISP')
                strt = max(tr1.stats.starttime,tr2.stats.starttime)    
                endt = min(tr1.stats.endtime,  tr2.stats.endtime)   
                tr1.trim(starttime=strt,endtime=endt) 
                tr2.trim(starttime=strt,endtime=endt) 
                # Rotate components first to ZNE
                vert,north,east = rotate2zne(tr1.data,ch1.azimuth,0.,
                                                tr2.data,ch2.azimuth,0.,
                                                np.zeros(tr1.stats.npts),0.,0.)
                radial,transverse = rotate_ne_rt(north,east,bazm)
                tr = Trace(header=tr1.stats,data=transverse)
                tr2 = Trace(header=tr2.stats,data=radial)
                tr.stats.channel = tr.stats.channel[:-1]+'T'
                # Change one of the invout channels to end in 'T'
                net = inv[-1]
                stn = net[0]
                chn = stn[0]
                chn.code = chn.code[:-1]+'T'
                #
                tr.stats.gcarc = gcarc 
                tr.stats.azimuth = azm
                decim = int(0.01+delta_T[phase]/tr.stats.delta)
                print tr.id,' ',tr.stats.sampling_rate,' decimated by ',decim
                print '%s: sensitivity=%g, azimuth=%g, dip=%g' % (ch1.code,
                        ch1.response.instrument_sensitivity.value,ch1.azimuth,ch1.dip)
                print '%s: sensitivity=%g, azimuth=%g, dip=%g' % (ch2.code,
                        ch2.response.instrument_sensitivity.value,ch2.azimuth,ch2.dip)
                if tr.stats.starttime-torg < 0.:
                    tr.trim(starttime=torg)
                    tr2.trim(starttime=torg)
                tr.decimate(factor=decim,no_filter=True)
                tr2.decimate(factor=decim,no_filter=True)
                tr.radial = 1.e6*tr2.data
                tr.stats.coordinates = coordinates
                tr.data *= 1.e6 # Convert to microns
            if phase == 'R' or phase == 'L': # Window according to group velocity window
                gwin = windows[phase]
                tbeg,tend = (dist*.001/gwin[1],dist*.001/gwin[0])
                tr.trim(torg+tbeg,torg+tend)
            elif phase == 'P' or phase == 'SH': # Window by times before and after pick
                tbef,taft = windows[phase]
                tr.trim(sta_pick.time-tbef,sta_pick.time+taft)
                idx = int(0.5+tbef/tr.stats.delta)
                avg = tr.data[:idx].mean()
                tr.data -= avg
                if not taper is None:
                    itp = int(taper*tr.stats.npts)
                    for i in range(tr.stats.npts-itp,tr.stats.npts):
                        tr.data[i] *= 0.5*(1.+mt.cos(mt.pi*(i-(tr.stats.npts-itp))/float(itp)))
                tr.stats.pick = sta_pick
            stobs[phase].append(tr)
            # Appen station inventory to invout
            if invout is None:
                invout = inv
            else:
                invout += inv
        # Pickle to file
    return stobs,invout         
                                    
def main(phases):
    from obspy.io.xseed import Parser
    from obspy.core.event import Event, Origin, Magnitude, NodalPlane
    from obspy.core import read, UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy import read_events

    stas_templates = {
            'ALL': ['POHA', 'PET', 'MAJO', 'BILL', 'YSS', 'MA2', 'MDJ', 'YAK',
                    'INCN', 'TIXI', 'AIS', 'CASY', 'QSPA', 'DRV', 'SBA', 'DZM',
                    'RAR', 'MSVF', 'AFI', 'HNR', 'WMQ', 'BRVK', 'KMI',
                    'AAK', 'CHTO', 'ABKT', 'FURI', 'KMBO', 'MSEY']
            }
    
    sta_adjtime = {}
    stas = {}
    for phase in phases:
        stas[phase] = stas_templates['ALL']
    
    #stalocs = {'KDAK':'10'}
    stalocs = {}
    mseed_filename = '../data/2004-11-11-mw75-timor-region-2.miniseed'
    client = Client('IRIS') 
    #client = Client('http://localhost:8080')
    #client = Client('http://compute2.rses.anu.edu.au:8080')
    DO_SYNTH = True
    if DO_SYNTH and ('R' in phases or 'L' in phases):
        print 'Initialize surface wave calculations'
        surfw_excite('./java ')
        readhrv(source_dir+'/surfw/ETL-JGR97 ')
        read_gdm52(source_dir+'/surfw/GDM52/')                                                           
        read_de06(source_dir+'/surfw/DE06a/')                                                            

    
    event_name = 'Alor_20041111'

    FAULT_FROM_PKL = False
    if os.path.isfile('event.pkl'):
        event = pickle.load(open('event.pkl','r'))
        print "SAYA DI SANA!"
    else:
        event  = read_events('./quakeml.xml')[0]
        pickle.dump(event,open('event.pkl','w'))
        print "SAYA DI SINI!"

    org = event.preferred_origin()
    if org is None:
        org = event.origins[0]
    
    if FAULT_FROM_PKL:
        fault = pickle.load(open('fault_neic.pkl','r'))
    else: 
        fault = neic2fault('p000d85g.param',org,xdr=-0.375,
                            velmod=read_velmod('./velmod.txt'),NEIC_potency=True)
    
    # Set parameters for deconvolution and time windowing
    windows = {'R':(2.75,4.5), 'L':(3.25,5.5), # Group velocity windows for surface waves
               'P':(5,75.), 'SH':(5,85.)}   # Times before and after body wave phases
    frq4 = {}
    for key in ['R','L']:  frq4[key] = [0.002,0.004,0.008,0.0125] # Surface wave filter
    for key in ['P','SH']: frq4[key] = [0.004,0.01,.25,0.5]       # Body wave filter
    
    origin = Origin()
    origin.time = UTCDateTime(org.time)
    origin.latitude  = fault.hypo.y
    origin.depth     = fault.hypo.z
    origin.longitude = fault.hypo.x
    fault.ntw = 1
    #stobs = getobs_surf(client,org,phases,frq4,gvel_win,stas,stalocs)
    print 'WAKTU=',origin.time
    
    stobs,inv = getobs(mseed_filename,client,event,phases,frq4,windows,stas,stalocs,
                   picks=get_isctimes('isc.txt',org.time),taper=0.2,adjtime=sta_adjtime)
    #return stobs,inv
    # write out the station inventory
    inv.write('%s.inv' % event_name,format="STATIONXML")

    # Calculate the subfault bodywave Green's functions
    for phase in ['P','SH']:
        if not phase in phases:
            continue
        if phase in phases:
            for sbf in fault.subfaults:
                sft1,sft2 = np.diff(sbf.ruptimes()[0:3])
                bodywave(sbf,stobs[phase],phase,sft1,sft2,frq4=frq4[phase])

    if not FAULT_FROM_PKL:
        pickle.dump(fault,open('fault_neic.pkl','w'))

    figs = {}
    ntrc = {'P':0, 'SH':0, 'R':0, 'L':0}
    if 'R' in stobs.keys() or 'L' in stobs.keys():
        stobs_path ='./stobs_surf_%s.pkl' % event_name 
        print 'Pickling stream to %s\n' % stobs_path
        stobs_surf = {'params':stobs['params']}
        for key in stobs.keys():
            if key == 'R' or key == 'L': 
                ntrc[key] = len(stobs[key])
                stobs_surf[key] = stobs[key]
        pickle.dump(stobs_surf,open(stobs_path,"wb"))
        figs['Surf'] = plt.figure(figsize=(12.,12.))
    if 'P' in stobs.keys() or 'SH' in stobs.keys():
        stobs_path ='./stobs_body_%s.pkl' % event_name 
        print 'Pickling stream to %s\n' % stobs_path
        stobs_body = {'params':stobs['params']}
        for key in stobs.keys():
            if key == 'P' or key == 'SH': 
                ntrc[key] = len(stobs[key])
                stobs_body[key] = stobs[key]
        pickle.dump(stobs_body,open(stobs_path,"wb"))
        figs['Body'] = plt.figure(figsize=(12.,12.))

#    """
    ### Plot the data
    npl_col = 2 # One for body and one for surface waves
    ax_wvf = {}
    th = 10.
    print windows['P']
    if 'P' in stobs.keys() or 'SH' in stobs.keys():
        print "WINDOWS[P][0]",windows['P'][0]
        stref = bodysynth_fault(fault,stobs,windows['P'][0])
    else:
        stref = {}
    ##stref,d5 = surfsynth_fault(fault,th,org.time,stobs['R'],frq4,gvel_win['R'])
    mxtrc = max(ntrc['R'],ntrc['L'])
    ncol = mxtrc/11+1
    nrow = min(11,mxtrc)
    print mxtrc,nrow,ncol
    if 'R' in stobs.keys():
        plt.figure(figs['Surf'].number)
        phs = 'R'
        #.return stobs
        stref[phs],d5 = surfsynth_fault(fault,th,org.time,stobs[phs],frq4[phs],windows[phs])
        return stobs,stref
        ax_wvf['R'] = plt.subplot2grid((2,npl_col), (0,0), rowspan=2, colspan=1)
        plot_datacomp(ax_wvf['R'],stobs,['R'],stref,origin,
                            rowcol=(nrow,ncol),yscale=0.25,ttick=500.)
        plt.show()
    if 'L' in stobs.keys():
        plt.figure(figs['Surf'].number)
        phs = 'L'
        stref[phs],d5 = surfsynth_fault(fault,th,org.time,stobs[phs],frq4[phs],windows[phs])
        ax_wvf['L'] = plt.subplot2grid((2,npl_col), (0,1), rowspan=2, colspan=1)
        plot_datacomp(ax_wvf['L'],stobs,['L'],stref,origin,
                            rowcol=(nrow,ncol,5),yscale=0.25,ttick=500.)
        
    mxtrc = max(ntrc['P'],ntrc['SH'])
    ncol = mxtrc/11+1
    nrow = min(11,mxtrc)
    print mxtrc,nrow,ncol
    sys.exit(0)
    if 'P' in stobs.keys():
        plt.figure(figs['Body'].number)     
        #stref = None
        ax_wvf['P'] = plt.subplot2grid((2,npl_col), (0,0), rowspan=2, colspan=1)
        plot_datacomp(ax_wvf['P'],stobs,['P'],stref,origin,
                            rowcol=(nrow,ncol),yscale=0.25,ttick=100.)
    if 'SH' in stobs.keys():
        plt.figure(figs['Body'].number)
        ax_wvf['SH'] = plt.subplot2grid((2,npl_col), (0,1), rowspan=2, colspan=1)
        plot_datacomp(ax_wvf['SH'],stobs,['SH'],stref,origin,
                            rowcol=(nrow,ncol),yscale=0.25,ttick=100.)
                            
    if 'P' in stobs.keys() or 'SH' in stobs.keys():
        plt.figure(figs['Body'].number)
        plt.tight_layout(pad=5., w_pad=0.5, h_pad=.0)
        plt.savefig('getdata_PSH.pdf')
    if 'R' in stobs.keys() or 'L' in stobs.keys():
        plt.figure(figs['Surf'].number)
        plt.tight_layout(pad=5., w_pad=0.5, h_pad=.0)
        plt.savefig('getdata_RL.pdf')
                    
    plt.show()
#    """
    return (fault,stref,stobs) #,d0,M
    #return (fault,stobs)


###
if __name__ == "__main__":
     output = main(['R','L','P','SH'])
    
