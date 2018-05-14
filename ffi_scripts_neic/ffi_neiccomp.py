#!/usr/bin/env python
import sys

import math as mt
import numpy as np
from matplotlib import colors,colorbar,cm
from matplotlib import pyplot as plt
from obspy.core import read,Stream,Trace,UTCDateTime
#from obspy.core.util import kilometer2degrees,gps2DistAzimuth
from obspy.geodetics import kilometer2degrees,locations2degrees,gps2dist_azimuth
from obspy import read_events

from scipy.optimize import nnls
import os
from obspy.imaging.beachball import Beach
import pickle
from fault8 import Point,velmod_findparam,Fault,read_velmod,rectsmooth
import matplotlib as mpl
from obspy.core.event import Event, Origin, Magnitude, NodalPlane
from surfw_lib import surfw_excite,surfw_calc,surfw_phv,taper,filter,readhrv,surfw_matrix
from surfw_lib import read_gdm52,read_de06
from scipy.interpolate import UnivariateSpline
ffipy_dir = os.environ['HOME'] + '/RSES-u5484353/Academics/PhD_research/02_HistoricalTsunamis/1992_Flores/20041111_Alor_earthquake/seismic_inversion/philscode'
if not ffipy_dir in sys.path:
    sys.path.append(ffipy_dir)
from ffi import fault_disp, kiksyn, thetasr, delay


def main():
    ffi_dir = '.'
    ntw = 5
    vrmax = 2.4
    th = 3.
    slipmax = 10.
    
    ## Data to reject
    #reject_stas = {'P':['NVS'],
    #               'SH':[],#['RAR', 'SNZO', 'SPA', 'TATO', 'YSS'],#['YSS','ERM','KIP'],
    #               'R':['NVS','NOUC','PPT','WMQ'], #NVS
    #               'L':['NOUC']} #TLY
                   
    reject_stas = {'P':['NVS','OGS'],
                   'SH':['NOUC','SNZO','OGS'],#['RAR', 'SNZO', 'SPA', 'TATO', 'YSS'],#['YSS','ERM','KIP'],
                   'R':['MDJ', 'LSA','KMI','LZH','ENH','NVS','SSE','NOUC','PPT','WMQ','OGS','SNZO','OBN','ERM'], #NVS
                   'L':['MDJ', 'LSA','KMI','LZH','ENH','NVS','SSE','NOUC','TLY','WMQ','OGS','SNZO','KIP','SUR']} 

    
    ## Becker & Lay stations
    #reject_stas = {'P':['ENH','OGS','NOUC','INU','TSK','HYB','ERM','DRV'
    #                    'WMQ','AAK','NVS','ARU','OBN','SUR','SSE','TATO',
    #                    'DRV','WMQ'],
    #               'SH':['ENH','OGS','NOUC','INU','MAJO','TSK','HYB','ERM','DRV'
    #                    'WMQ','AAK','NVS','ARU','OBN','YSS','KIP','CAN','KMI',
    #                    'TLY','DRV','WMQ'],
    #               'R':['ENH','OGS','NOUC','INU','MAJO','TSK','HYB','ERM','DRV'
    #                    'WMQ','AAK','NVS','ARU','OBN'],
    #               'L':['ENH','OGS','NOUC','INU','MAJO','TSK','HYB','ERM','DRV'
    #                    'WMQ','AAK','NVS','ARU','OBN']}


                   
    velmod = read_velmod(ffi_dir+'/velmod.txt')
    event_name = 'Alor_20041111'

    if os.path.isfile('event.pkl'):
        event = pickle.load(open('event.pkl','r'))
    else:
        event  = read_events('./quakeml.xml')[0]
        pickle.dump(event,open('event.pkl','w'))
    # Load the neic fault
    fault_ref = pickle.load(open('fault_neic.pkl'))
    trmx = 1000.
    

    org = event.preferred_origin()
    if org is None:
        org = event.origins[0]
    origin_time = org.time
    nx = fault_ref.sbf_nx
    ny = fault_ref.sbf_ny
    # Set up the fault from scratch
    if True:
        # gcmt is hypocntre followed by strike,dip,rake
        #hypo = Point(org.longitude,org.latitude,0.001*org.depth)
        hypo = Point(124.868,-8.152,0.001*org.depth)
        hypo.z = 10.
        #hypo = Point(org.longitude,org.latitude,25.)
        (strike,dip,rake) = (80.,19.,90.)
        origin_time = org.time
        length = 240.
        width  = 80. 
        h_top  = 0.
        xdrct  = -0.6    #directivity: -1.0 - +1.0, -1=right 0=centre 1=left
        trmx   = length/vrmax + (ntw-1)*th
        fault  = Fault(fault_spec=[hypo,strike,dip,h_top,xdrct,length,width],velmod=velmod)
        print 'Hypo, strike, dip, length, width: ',hypo,strike,dip,length,width
        nx = 24                            #24
        ny = 8                              #6
        fault.rectangular_tessellation(nx=nx,ny=ny)
        fault.nwin = ntw
        fault.vrmax = vrmax

    else:
        fault = fault_ref.copy()

    fault.ntw = ntw
    fault.vrmax = vrmax
    fault.th = th
    #return fault
        
                
    Data_type = 'Body and Surface Waves'    
    phases_to_use    = ['R','L','P','SH']    
    phases_to_invert = ['R','L','P','SH']
#    Data_type = 'Surface Waves Only'
#    phases_to_use    = ['R','L']    
#    phases_to_invert = ['R','L']
#    Data_type = 'Body Waves Only'    
#    phases_to_invert = ['P','SH']
#    phases_to_use    = ['P','SH']

    #Mo = 10.**(1.5*(Mw+10.7)-7)
    origin = Origin()
    origin.time = UTCDateTime(origin_time)
    origin.latitude  = fault.hypo.y
    origin.depth     = fault.hypo.z
    origin.longitude = fault.hypo.x

    print 'hypo = %5.2f,%5.2f,%5.2f' % (origin.longitude,origin.latitude,origin.depth)
    #print 'strike,dip,length,width: ',strike,dip,length,width
    #print 'at subfault ',fault.subfaults[ih].cntr

    # Read the data
    stobs = {}
    if 'R' in phases_to_use or 'L' in phases_to_use:
        surfobs  = pickle.load(open('%s/stobs_surf_%s.pkl' % (ffi_dir,event_name),'rb'))
        frq4 = surfobs['params']['filter']
        wndw = surfobs['params']['windows']
        for phase in ['R','L']: 
            print phase,reject_stas[phase]
            if not phase in phases_to_use: continue
            stobs[phase] = Stream()
            if not phase in surfobs.keys(): continue
            for tr in surfobs[phase]:
                if not tr.stats.station in reject_stas[phase]:
                    stobs[phase].append(tr)
                else:
                    print 'Rejecting %s for phase %s' % (tr.stats.station,phase)
    else:
        frq4 = {}
        wndw = {}
    # Set up for surface wave calculation
    if 'R' in phases_to_use or 'L' in phases_to_use:
        surfw_excite(ffi_dir+'/java ')
        readhrv(ffipy_dir+'/surfw/ETL-JGR97 ')
        readhrv(ffipy_dir+'/surfw/ETL-JGR97 ')
        read_gdm52(ffipy_dir+'/surfw/GDM52/')
        read_de06(ffipy_dir+'/surfw/DE06a/')
     
    x = np.zeros(2*fault.ntw*len(fault.subfaults))
    nparam = len(x)
    Amat = {}
    Bvec = {}
    #  Calculate surface wave matrix
    if 'R' in phases_to_invert or 'L' in phases_to_invert:
        norm = True
        Amat['surf'],surf_stgn = surfsynth_matrix(fault,origin.time,stobs,frq4['R'],normalize=norm)
        Bvec['surf'] = data_vector(stobs,surf_stgn,phases=['R','L'],normalize=norm)
    else:
        Amat['surf'] = np.empty((0,nparam))
        Bvec['surf'] = np.array([])
    
    # Set slip vector from fault_ref - this was used for testing only
    slip = np.zeros(2*len(fault.subfaults))
    k = 0
    for sbf1,sbf2 in zip(fault_ref.subfaults,fault.subfaults):
        sbf2.rake = 90.
        slip[ k ] = sbf1.cumslip[0]*mt.cos(mt.radians(sbf2.rake-45.)) \
                  + sbf1.cumslip[1]*mt.cos(mt.radians(90.-sbf2.rake+45.))
        slip[k+1] = sbf1.cumslip[0]*mt.cos(mt.radians(sbf2.rake+45)) \
                  + sbf1.cumslip[1]*mt.cos(mt.radians(90.-sbf2.rake-45.))                  
        sbf1.rake = 45.
        k += 2
    # Leave the interpolants the way they are for now, i.e. =fault_ref
    # Set Fault SlipInterpolants to match MTW rupture model
    SetFaultSlipInterpolants(fault,stf=(vrmax,th))
    #return fault
    # Now the body waves
    if 'P' in phases_to_use or 'SH' in phases_to_use:        
        bodyobs  = pickle.load(open('%s/stobs_body_%s.pkl' % (ffi_dir,event_name),'rb'))
        for key in bodyobs['params']['filter']:
            frq4[key] = bodyobs['params']['filter'][key]
            wndw[key] = bodyobs['params']['windows'][key]
            print 'Data Window ',key,': ',wndw[key]
        for phase in ['P','SH']: 
            if not phase in phases_to_use or not phase in bodyobs.keys(): 
                continue
            stobs[phase] = Stream()
            for tr in bodyobs[phase]:
                if not tr.stats.station in reject_stas[phase]:
                    stobs[phase].append(tr)
                else:
                    print 'Rejecting %s for phase %s' % (tr.stats.station,phase)

            for sbf in fault.subfaults:
                if False: # Set from fault_ref - used ofr testing only
                    sbf.slip_rate = fault_ref.subfaults[sbf.index].slip_rate
                    sft1,sft2 = np.diff(sbf.ruptimes()[0:3])
                else: # Set using fault.ntw, fault.thd, and fault.vrmax
                    sbf.rake = 90.
                    sft1 = sft2 = th
                bodywave(sbf,stobs[phase],phase,sft1,sft2,frq4=frq4['P'])
            for sbf in fault_ref.subfaults:
                sbf.rake = 90.
                sft1,sft2 = np.diff(sbf.ruptimes()[0:3])
                bodywave(sbf,stobs[phase],phase,sft1,sft2,frq4=frq4['P'])
            

    if 'P' in phases_to_invert or 'SH' in phases_to_invert:
        norm = True
        Amat['body'],bdyw_stgn = bodyw_matrix(fault,stobs,wndw['P'][0],normalize=norm)
        Bvec['body'] = data_vector(stobs,bdyw_stgn,phases=['P','SH'],normalize=norm)
    else:
        Amat['body'] = np.empty((0,nparam))
        Bvec['body'] = np.array([])
    #return stobs,Amat,Bvec
    
    # Read in the SAR model
    if 'sar' in phases_to_invert:
        sarqt = read_sarqt('sar/SARtree.dat')
        # Now the coseismic matrix
        tmpmat = fault.coseismic_matrix(sarqt, rakes, unorm=np.eye(3))#np.transpose([[0.,0.,1.]]))#u
        sarmat = np.repeat(tmpmat[2::3,:],fault.ntw,axis=1)
        Amat['sar'] = sarmat
        Bvec['sar'] = sarqt[:,2]
    else:
        Amat['sar'] = np.empty((0,nparam))
        Bvec['sar'] = np.array([])
    
    
    M = rectsmooth(fault)
    gamma = 2.#0.5
    #alpha = {'body':1.-gamma, 'sar':100., 'surf':gamma}
    alpha = {'body':2., 'sar':100., 'surf':2.}
    ####alpha = {'body':1.-gamma, 'surf':gamma, 'tsunami':0.5, 'disp':5.}
    beta = 1.e-1
    
    #A = np.vstack((Amat['surf'],alpha*Amat['body'],beta*M))
    #b = np.hstack((Bvec['surf'],alpha*Bvec['body'],np.zeros(M.shape[0])))
    A = np.array([[] for _ in xrange(M.shape[0])]).T
    b = np.array([])
    for data_type in ['surf','body']: #'sar',
    ####for data_type in ['tsunami','disp','surf','body']:
        print A.shape,Amat[data_type].shape
        A = np.vstack((A,alpha[data_type]*Amat[data_type]))
        b = np.hstack((b,alpha[data_type]*Bvec[data_type]))
    #return A,b
    A = np.vstack((A,beta*M))
    b = np.hstack((b,np.zeros(M.shape[0])))
    slip,r = nnls(A,b)
    SetFaultSlipInterpolants(fault,slip=slip)#,vrmax,t12)
    #return fault,fault_ref,slip,x,Amat['body'],Bvec['body']
    

    # Now calculate the synthetics through multiplication by the GF matrix
    stref = {}
    rowcol = {}
    if 'R' in phases_to_invert or 'L' in phases_to_invert:
        d0 = matrix_to_synth(fault,surf_stgn,Amat['surf'],normalize=norm)
        d5 = surfsynth_fault(fault_ref,th,org.time,stobs,stref,frq4)
        # For plotting
        mxtrc = 0
        for phase in surfobs.keys(): mxtrc = max(mxtrc,len(surfobs[phase]))
        rowcol['surf'] = (min(10,mxtrc) , mxtrc/10+1)
        print 'surf: ',mxtrc,rowcol['surf']
    if 'P' in phases_to_invert or 'SH' in phases_to_invert:
        #d0 = bodymatrix_to_synth(Amat['body'],body_stgn,slip) 
        bdyw_stgn = bodysynth_fault1(fault,stobs,wndw['P'][0])#,MatSlip=(Amat['body'],slip))
        bodysynth_fault(fault_ref,stobs,stref,wndw['P'][0])
        # For plotting
        mxtrc = 0
        for phase in bodyobs.keys(): mxtrc = max(mxtrc,len(bodyobs[phase]))
        rowcol['body'] = (min(10,mxtrc) , mxtrc/10+1)

    figs = {
        #'R'  :{'number':1, 'title_bot':'Love Waves (%d - %d s)'     % (tshrt,tlong)},
        #'L'  :{'number':2, 'title_bot':'Rayleigh Waves (%d - %d s)' % (tshrt,tlong)},
        'R'  :{'number':1, 'title_bot':'Love Waves'     , 'rwcl':rowcol['surf'], 'tt':1000. },
        'L'  :{'number':2, 'title_bot':'Rayleigh Waves ', 'rwcl':rowcol['surf'], 'tt':1000. },
        'P'  :{'number':3, 'title_bot':'P Waves'        , 'rwcl':rowcol['body'], 'tt':100. },
        'SH' :{'number':4, 'title_bot':'S Waves '       , 'rwcl':rowcol['body'], 'tt':100. },
        'FFI':{'number':5, 'title_bot':'Finite Fault NEIC vs. LSQ'},
        }
    title_top = ['NIEC', 'Ntw=%d,th=%3.1f, Vrmax=%3.1f %s' % (fault.ntw,th,vrmax,Data_type)]  
    #title_top = ['Ntw=%d,th=%3.1f, Vrmax=%3.1f %s' % (fault.ntw,th,vrmax,Data_type)]
    Main_title = 'FFI Comparison with NEIC'           
    axs = {}
    stgn = {}
    for phase in phases_to_use:
        if phase == 'P' or phase =='SH':
            stgn[phase] = bdyw_stgn[phase]
        if phase == 'R' or phase =='L':
            stgn[phase] = surf_stgn[phase]
    figwv = {}
    for key in figs.keys():
        # Waveform plots
        if key in phases_to_use and key in stobs.keys() and len(stobs[key])>0:
            figwv[key] = plt.figure(num=figs[key]['number'],figsize=(12.,12.))
            npl_row,npl_col = (1,2) 
            if key in phases_to_use:
                if key in ['R','L']:
                    axky = key+'_'+title_top[0]
                    #stref[key],d5 = surfsynth_fault0(fault_ref,th,org.time,stobs[key],frq4[key],wndw[key])
                    axs[axky]  = plt.subplot2grid((npl_row,npl_col), (0,0), rowspan=2, colspan=1)
                    plot_datacomp(axs[axky],stobs,[key],stref,origin,
                                    rowcol=figs[key]['rwcl'],yscale=0.25,ttick=figs[key]['tt'])
                else:
                    axky = key+'_'+title_top[0]
                    axs[axky]  = plt.subplot2grid((npl_row,npl_col), (0,0), rowspan=2, colspan=1)
                    plot_datacomp(axs[axky],stobs,[key],stref,origin,title=title_top[0],
                                    rowcol=figs[key]['rwcl'],yscale=0.25,ttick=figs[key]['tt'])                    
                axky = key+'_'+title_top[1]
                axs[axky]  = plt.subplot2grid((npl_row,npl_col), (0,1), rowspan=2, colspan=1)
                plot_datacomp(axs[axky],stobs,[key],stgn,origin,title=title_top[1],
                                rowcol=figs[key]['rwcl'],yscale=0.25,ttick=figs[key]['tt'])
                plt.tight_layout(pad=5., w_pad=0.5, h_pad=.0)
                plt.text(0.5,0.965, Main_title, fontsize=20,horizontalalignment='center',
                         transform=figwv[key].transFigure)      
                plt.savefig('%s_%s.pdf' % (event_name,key))
        elif key in ['sar']:
            print "Here's where you'd plot SAR"
            continue
            fig3 = plt.figure(num=ifg,figsize=(20.,20. / 1.618))
            plot_sar(fig3,sarqt,fault,sarmat,x,rakes,slipmax=slipmax,
                        cmap = plt.cm.hot_r)
            fig3.savefig('SARfits.pdf')
        elif key in ['FFI']: # Plot both fault and fault_ref models
            fig_rptmdl = plt.figure(num=figs[key]['number'],figsize=(15.,15.))
            times = [20.,40.,80.,160.]
            npl_row = len(times)
            npl_col = 2
            norm = colors.Normalize(vmin=0., vmax=slipmax)
            #trmx = 1000.
            #for sbf in fault.subfaults:
            #        print sbf.index,sbf.cumslip ,(sbf.slip[0]([trmx])[0],sbf.slip[1]([trmx])[0])
            #sys.exit(0)
            for ipl,time in enumerate(times):             
                lbl ='lb'
                ax_ref = plt.subplot2grid((npl_row,npl_col), (ipl,0))
                fault_ref.xy_plot(ax_ref,field='slip',time=time,norm=slipmax,labels=lbl)
                ax_inv = plt.subplot2grid((npl_row,npl_col), (ipl,1))
                lbl = 'b'
                fault.xy_plot(ax=ax_inv,field='slip',time=time,norm=slipmax,labels=lbl)
                if ipl == 0:
                    ax_ref.text(0.5,1.2,'NEIC Rupture Model Mw=%3.1f' % \
                                (2.*mt.log10(fault_ref.moment())/3.- 6.033), \
                                fontsize=19,horizontalalignment='center', transform=ax_ref.transAxes)
                    ax_inv.text(0.5,1.2,'Least Squares - %s Mw=%3.1f' % \
                                (Data_type,2.*mt.log10(fault.moment())/3.- 6.033), \
                                fontsize=19,horizontalalignment='center', transform=ax_inv.transAxes)
                ax_inv.text(0.,1.05,'time=%3.0f sec'% time, fontsize=18,horizontalalignment='center',
                            transform=ax_inv.transAxes)
            plt.tight_layout(pad=5., w_pad=0.5, h_pad=.0)
            cax = fig_rptmdl.add_axes([0.25, 0.05, 0.5, 0.025])
            colorbar.ColorbarBase(ax=cax,label='Slip (m)',cmap=plt.cm.hot_r,norm=norm,orientation='horizontal')
            plt.text(0.5,0.925, Main_title, fontsize=20,horizontalalignment='center',transform=fig_rptmdl.transFigure) 
            fig_rptmdl.savefig('%s_%s.pdf' % (event_name,'RptMdls'))     
    plt.show()
    #return (fault,slip,r,A,b,fault_ref,stobs,stgn,stref)
    return (fault,slip,r,A,b,fault_ref,stobs,stgn,stref,Amat,Bvec,M)

def read_sarqt(filename):
    sarqt = []
    for line in open(filename,'r'):
        sarqt.append([float(x) for x in line.split()])
        #sarqt[-1].append(1.)
    sarqt = np.array(sarqt)
    sarqt[:,2] = -sarqt[:,2]
    return sarqt

def TimeSlipDistance_plot(fault,ax,cmap,norm):
        time = []
        dist = []
        slip = []
        for sbf in fault.subfaults:
            times = sbf.slip_rate[0].get_knots()[1:-3]
            for t in times:
              slipx = sbf.slip[0]([t])[0]
              slipy = sbf.slip[1]([t])[0]
              slip.append(mt.hypot(slipx,slipy))
              time.append(t)
              dist.append(mt.hypot(sbf.hypo_distance,sbf.hypo_deltaz))
        ax.scatter(time,dist,slip,cmap=cmap,norm=norm)

def data_vector(stobs,stgrn,phases=None,normalize=False):
    d = []
    for phase in stgrn.keys():
        if phases != None and not phase in phases:
            continue
        sto = stobs[phase]
        stg = stgrn[phase]
        norm = 1.
        for to,tg in zip(sto,stg):
            to.trim(starttime=tg.stats.starttime,endtime=tg.stats.endtime)
            if normalize:
                norm = to.stats.calib
            d += list(to.data*norm)
    return np.array(d)
    
def plot_sar(fig,sarqt,fault,sar_mat,x,rakes,slipmax=10.,cmap=plt.cm.hot_r):
    # Plot data and rsult
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
    m = fault.map_plot(ax_flt,xpnd=.75,field='cumslip',cmap=cmap,
            norm=slipmax,background='shadedrelief')
    
    cax = fig.add_axes([0.5, 0.1, 0.025, 0.35])
    colorbar.ColorbarBase(ax=cax,label='Range/-Vertical Displacement (m)',
            cmap=cmap,norm=norm)
    
    dax = fig.add_axes([0.7, 0.1, 0.25, 0.4])
    rsd = sarqt[:,2] - pred
    dax.hist(rsd)
    
def plot_datacomp(ax,stobs,phases,stgrn,origin,rowcol=None,title=None,
                    yscale=0.5,ttick=100.,timing='',fontsize=6,tbuf=.15):
    #phases = stobs.keys()
    torg = origin.time
    ntr = 0
    trng = 0.
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
            xnrml = 2./(tr.data.max()-tr.data.min())
            if stgrn != None:
                tsyn = stgrn[phase][idx]
                tsyoff = 0.
                if timing == 'origin_time':
                    tsyoff = tsyn.stats.starttime-torg
                #if phase == 'P':
                #    print tr.stats.station,tsyn.data.max(),tr.data.max()
                if phase == 'P':
                    ax.plot(ttoff+tsyn.times()+tsyoff-toff,-irow + 1.*yscale*tsyn.data*xnrml,color='red')           
                else:
                    ax.plot(ttoff+tsyn.times()+tsyoff-toff,-irow + yscale*tsyn.data*xnrml,color='red')           
            ax.plot(ttoff+tr.times(),-irow + yscale*tr.data*xnrml,color='black')
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
            ax.text(txtoff+trng,-irow+.25 ,'%4.0f$\mu$' % (1./xnrml),\
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
    #ax.set_xticks(np.arange(tmin,tmax,ttick))
    # Set a bar at lower right to show the time scale
    if not ttick is None:   # Plot a horiontal bar to show time scale
        x  = tmin+(ncol-0.5)*((2.*tbuf+1.)*trng)
        xs = [x-0.5*ttick,x-0.5*ttick,x-0.5*ttick,x+0.5*ttick,x+0.5*ttick,x+0.5*ttick]
        y  = -(nrow-1.)
        ys = [y-0.15,y+0.15,y,y,y+0.15,y-0.15]
        ax.plot(xs,ys)
        ax.text(x,y-0.15,'%g sec' % ttick,ha='center',va='center')
    if not title is None:   # Set title is desired
        ax.set_title(title)
    return
        
def bodyw_matrix(fault,stobs,toff,normalize=False):
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
      nrow = 0
      for phase in stobs.keys():
          if phase != 'P' and phase != 'SH': continue
          for trc in stobs[phase]:
              nrow += trc.stats.npts
              if normalize:
                  trc.stats.calib = 1./max(trc.data.max(),abs(trc.data.min()))
      ncol = len(fault.subfaults)*2*fault.ntw
      amat = np.zeros((nrow,ncol))
      icol = 0
      for sbf in fault.subfaults:
        times   = sbf.slip_rate[0].get_knots()[1:]
        for jwn in range(0,fault.ntw):
            ruptime = times[jwn] 
            risetim = times[jwn+1]-times[jwn]
            for jrk in range(0,2):
                jcol = icol + jwn*2 + jrk
                irow = 0             
                for phase in stobs.keys():
                    if phase != 'P' and phase != 'SH':
                        continue
                    if phase not in sbf.GreenFuncs.keys():
                        raise Exception, "No Green's functions for phase %s" % phase
                    for tro,trc in zip(stobs[phase],sbf.GreenFuncs[phase][jrk]):
                        # Calculate time shift as a sum of three terms:
                        #    (1) toff, the pre-arrival time window (normally 
                        #        chosen to be compatible with observed trace)
                        #    (2) delay, of subfault ray at receiver, measured w.r.t.
                        #        hypocentral ray (i.e., infinite rupture velocity)
                        #    (3) ruptime, subfault rupture time, w.r.t. origin time 
                        #    (4) the half-duration, due align triangle start 
                        #time_shift = ruptime + toff - trc.stats.delay + sbf.stf[1]
                        time_shift = ruptime + toff - trc.stats.delay + risetim
                        ishift = int(time_shift/trc.stats.delta + 0.4)

                        if ishift < 0.:
                            raise Exception,('Ruptime+Toff(%g) too small ' + \
                            'to accommodate subfault time delay (%g)') %\
                            (ruptime+toff,trc.stats.delay)
                        if ishift == 0:
                            amat[irow:irow+trc.stats.npts,jcol] = trc.data[:]
                        elif ishift<trc.stats.npts:
                            amat[irow+ishift:irow+trc.stats.npts,jcol] = trc.data[0:-ishift]
                            if normalize:
                                amat[irow+ishift:irow+trc.stats.npts,jcol] *= tro.stats.calib
                        irow += trc.stats.npts
        icol += 2*fault.ntw
      #return amat
      stgn = {}
      irow = 0
      for phase in stobs.keys():
            if phase != 'P' and phase != 'SH': continue
            stgn[phase] = Stream()
            st = stobs[phase]
            for tr in st:
                trgn = Trace(header=tr.stats)
                trgn.data = np.zeros(trgn.stats.npts) # To satisfy obspy
                trgn.greens_funcs = amat[irow:irow+trgn.stats.npts,:]
                irow += trgn.stats.npts
                stgn[phase].append(trgn)
      return amat,stgn

def bodysynth_fault(fault,stobs,stgn,toff,Verbose=False):
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
      chan = {'P':'BHZ', 'SH':'BHT'}
      icol = 0
      for sbf in fault.subfaults:
        if not hasattr(sbf,'rake'):
            raise Exception, 'Subfault must have rake set in order to call bodysynth_fault'
        times   = sbf.slip_rate[0].get_knots()[1:]
        for jwn in range(0,fault.ntw):
            ruptime = times[jwn] 
            risetim = times[jwn+1]-times[jwn] 
            falltim = times[jwn+2]-times[jwn+1] 
            sliprate = ((sbf.slip_rate[0]([ruptime+risetim]))[0],
                        (sbf.slip_rate[1]([ruptime+risetim]))[0])
            for jrk in range(0,2):
                if jrk == 0:  # Rotate sliprate to direction sbf.rake-45
                    slip =  sliprate[0]*mt.cos(mt.radians(sbf.rake-45.)) \
                                   +sliprate[1]*mt.cos(mt.radians(90.-sbf.rake+45.))
                else:         # Rotate sliprate to direction abf.rake+45.
                    slip =  sliprate[0]*mt.cos(mt.radians(sbf.rake+45.)) \
                                   +sliprate[1]*mt.cos(mt.radians(90.-sbf.rake-45.))
                # Convert sliprate to slip by multiplying by 0.5 X triangle base
                slip *= 0.5*(risetim+falltim)
                jcol = icol + jwn*2 + jrk
                colmax = 0.
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
                        #time_shift = ruptime + toff - trc.stats.delay +sbf.stf[1]
                        time_shift = ruptime + toff - trc.stats.delay + risetim
                        ishift = int(time_shift/trc.stats.delta + 0.4)
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
                        #if jcol == 0 or jcol == len(fault.subfaults)*fault.ntw*2-1 or \
                        #    ishift>=trc.stats.npts:
                        #    print 'jcol=%d, %s,tshift=%g,ishift=%d,npts=%d,ruptime=%g,tdelay=%g' % \
                        #    (jcol,tro.stats.station,time_shift,ishift,trc.stats.npts,ruptime,trc.stats.delay)
                        if ishift == 0:
                            trgn.data += slip*trc.data[:]
                            colmax = max(colmax,trc.data.max())
                        elif ishift<trc.stats.npts:
                            trgn.data[ishift:trc.stats.npts] += \
                                         slip*trc.data[0:-ishift]
                            colmax = max(colmax,trc.data[0:-ishift].max())
                if Verbose and (jcol == 0 or jcol == len(fault.subfaults)*fault.ntw*2-1):
                    print 'bodysynth_fault: jcol=%d, trmax=%g, slip=%g' % (jcol,colmax,slip)
        icol += 2*fault.ntw
      return stgn  

def bodysynth_fault1(fault,stobs,toff,MatSlip=None,Verbose=False):
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
      if not MatSlip is None:
          amat,slipvec = MatSlip
          d = np.dot(amat,slipvec)
          Verbose = True
      stgn = {}
      chan = {'P':'BHZ', 'SH':'BHT'}
      icol = 0
      for sbf in fault.subfaults:
        if not hasattr(sbf,'rake'):
            raise Exception, 'Subfault must have rake set in order to call bodysynth_fault'
        times   = sbf.slip_rate[0].get_knots()[1:]
        for jwn in range(0,fault.ntw):
            ruptime = times[jwn] 
            risetim = times[jwn+1]-times[jwn] 
            falltim = times[jwn+2]-times[jwn+1] 
            sliprate = ((sbf.slip_rate[0]([ruptime+risetim]))[0],
                        (sbf.slip_rate[1]([ruptime+risetim]))[0])
            for jrk in range(0,2):
                if jrk == 0:  # Slip along direction rake=-45
                    slip =  sliprate[0]*mt.cos(mt.radians(sbf.rake-45.)) \
                           +sliprate[1]*mt.cos(mt.radians(135.-sbf.rake))
                else:         # Slip along direction rake+45.
                    slip =  sliprate[0]*mt.cos(mt.radians(sbf.rake+45.)) \
                           +sliprate[1]*mt.cos(mt.radians(90.-sbf.rake-45.))
                slip *= 0.5*(risetim+falltim)
                # If Verbose=True, check whether calculatec calculated slip 
                # and slip vector are different 
                jcol = icol + jwn*2 + jrk
                if Verbose and not MatSlip is None and \
                                (not np.isclose(slip,slipvec[jcol])):
                    print 'Slip vector not close for jcol,jrk %d,%d: %g,%g' % (jcol,jrk,slip,slipvec[jcol])
                colmax = 0.
                irow = 0
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
                        #time_shift = ruptime + toff - trc.stats.delay +sbf.stf[1]
                        time_shift = ruptime + toff - trc.stats.delay + risetim
                        ishift = int(time_shift/trc.stats.delta + 0.4)
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
                        if ishift == 0:
                            trgn.data += slip*trc.data[:]
                            colmax = max(colmax,trc.data.max())
                            # Check whether un-shifted GFs and matrix column are the same
                            if not MatSlip is None: 
                                if False in np.isclose(trc.data[:],\
                                                        amat[irow:irow+trc.stats.npts,jcol]):
                                    print phase,trc.stat.station,sbf.index,\
                                          'matrix column does not match un-shifted GF'
                                d01 = slip*trc.data[:]
                        elif ishift<trc.stats.npts:
                            trgn.data[ishift:trc.stats.npts] += \
                                         slip*trc.data[0:-ishift]
                            colmax = max(colmax,trc.data[0:-ishift].max())
                            if (not MatSlip is None):
                                if False in np.isclose(np.zeros(ishift),\
                                                       amat[irow:irow+ishift,jcol]):
                                    print phase,trc.stat.station,sbf.index,' not zero for index<ishift',
                                # Check whether shifted GFs and matrix column are the same
                                if False in np.isclose(trc.data[0:-ishift],\
                                                       amat[irow+ishift:irow+trc.stats.npts,jcol]):
                                    print phase,trc.stats.station,sbf.index,\
                                          'matrix column does not match shifted GF'
                                d01 = np.zeros(trc.stats.npts)
                                d01[ishift:trc.stats.npts] += slip*trc.data[0:-ishift]
                        # Check whether each column of matrix multiplication matches
                        if (not MatSlip is None):
                            d02 = slipvec[jcol]*amat[irow:irow+trc.stats.npts,jcol]
                            if ishift >=trc.stats.npts:
                                if False in np.isclose(d02,np.zeros(trc.stats.npts)):
                                    print phase,trc.stats.station,sbf.index,' != 0 why?'
                            elif False in np.isclose(d01,d02):
                                print phase,trc.stats.station,sbf.index,' why?'
                            # Check whether full matrix multplication (over all columns) matches full GF
                            if sbf == fault.subfaults[-1] and jwn == fault.ntw-1 and jrk == 1:
                                if False in np.isclose(d[irow:irow+trc.stats.npts],trgn.data):
                                    print '%2s:%4s: Fault synthetic does not match matmul %g,%g' % \
                                           (phase,trc.stats.station,d[irow:irow+trc.stats.npts].max(),\
                                           trgn.data.max())
                                    #for krow in range(irow,irow+trc.stats.npts):
                                    #    print krow,d[krow],trgn.data[krow-irow]
                        irow += trc.stats.npts
                if Verbose and (jcol == 0 or jcol == len(fault.subfaults)*fault.ntw*2-1):
                    print 'bodysynth_fault1: jcol=%d, trmax=%g, slip=%g' % (jcol,colmax,slip)
        icol += 2*fault.ntw
      return stgn  


def bodywave(sbf,st,phase,t1,t2,rake=None,velmod=None,frq4=None):
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
      if not rake is None:
        if rake < -180. or rake > 180.:
            raise Exception, 'Expected -180.<rake<180., not %g' % rake
        else:
            sbf.rake = rake 
      else:
        if not hasattr(sbf,'rake'):         
            raise Exception, 'Bodywave called with rake=None, and subfault %d has no rake' % sbf.index
        rake = sbf.rake
      ## Loop over traces in stream
      stbdy = [Stream(), Stream()]
      for tr in st:
        #nt = len(tr.data)
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


   

def surfsynth_matrix(fault,otime,stobs,frq4,gvel_win=None,normalize=False):
    '''
    If gvel_win is set to a dictionary containing Rayleigh and Love group
    velocities:
        gvel_win = {'R':(3.,4.5), 'L':(3.5,7.)}
    then the synthetics are windowed accordingly. Ig gvel_win is not set,
    the synthetics are calculated to have the same timing as the data stobs.
    # Calculate phase velocities, etc from epicenter to stations
    Returns amat, stgrn, where:
        amat:  is a matrix of tsunami Green's funcitons, with each row 
               corresponding to a time sample of the observed data, arranged
               in the order given by:
                   for phase in stobs_keys():   
                      for tr in stobs[phase]
               Each column of amat represents a component of slip on a subfault, 
               first component along rake-45., the second along rake+45. These 
               component pairs are arranged in the order:
                   for subfault in fault.subfaults 
    
    '''
    ntw  = fault.ntw
    th   = fault.th
    vmax = fault.vrmax
    ns = 0
    for phase in stobs.keys():
        if phase =='R' or phase == 'L':
            ns +=len(stobs[phase])
    stas  = []
    stlo  = np.zeros(ns)
    stla  = np.zeros(ns)
    dt    = np.zeros(ns)
    nt    = np.zeros((ns,), dtype=np.int) 
    ib    = np.zeros((ns,), dtype=np.int)
    tstrt = np.zeros(ns)
    ibeg  = np.zeros((ns,), dtype=np.int)
    iend  = np.zeros((ns,), dtype=np.int)
    ntm = 0 
    nrow = 0
    js = 0
    for phase in stobs.keys():
        if phase != 'R' and phase != 'L':
            continue
        st = stobs[phase]
        if gvel_win != None:
            gv = gvel_win[phase]
        for j,tr in enumerate(st):
            i = j+js
            stas.append(tr.stats.station)
            stlo[i] = tr.stats.coordinates['longitude']
            stla[i] = tr.stats.coordinates['latitude']
            nt[i]   = tr.stats.npts
            dt[i]   = tr.stats.delta
            if normalize:
                tr.stats.calib = 1./max(tr.data.max(),abs(tr.data.min()))
            if tr.stats.channel[-1] == 'Z':
                ib[i] = 7
            elif tr.stats.channel[-1] == 'T':
                ib[i] = 6
            tstrt[i] = 0. # Just start synths at origin time #tr.stats.starttime - otime
            # Calculate indices for group velocty window
            dist,azm,bazm = gps2dist_azimuth(fault.hypo.y,fault.hypo.x,
                                            tr.stats.coordinates['latitude'],
                                            tr.stats.coordinates['longitude'])
            if gvel_win != None:
                tstrt[i] = tr.stats.starttime - otime # Does this make sense?               
                tbeg,tend = (dist*.001/gv[1],dist*.001/gv[0])
                ibeg[i] = int(tbeg/tr.stats.delta+.5)
                iend[i] = int(tend/tr.stats.delta+.5)
            else:
                tstrt[i]  = tr.stats.starttime - otime
                tbeg    = tr.stats.starttime - otime
                ibeg[i] = 1
                iend[i] = tr.stats.npts
                '''
                ibeg[i] = int(tbeg/tr.stats.delta+.5)
                iend[i] = ibeg[i]+tr.stats.npts-1                
                if tr.stats.station == 'WMQ':
                    print tr.stats.station,tbeg,iend[i]-ibeg[i]+1
                '''
            if iend[i] > ntm: ntm = iend[i]
            nrow += iend[i]-ibeg[i]+1
        js += len(st)
    ntm  = 2048        
    nsf  = len(fault.subfaults)
    sflon = np.zeros(nsf)
    sflat = np.zeros(nsf)
    sfdep = np.zeros(nsf)
    sfstk = np.zeros(nsf)
    sfdip = np.zeros(nsf)
    sfrak = np.zeros(nsf)
    sfrds = np.zeros(nsf)
    sfrtm = np.zeros(nsf)
    sfhdr = np.zeros(nsf)
    sfpot = np.zeros(nsf)
    ncol = 0
    for i,sbf in enumerate(fault.subfaults):
        sflon[i] = sbf.cntr.x
        sflat[i] = sbf.cntr.y
        sfdep[i] = sbf.cntr.z
        sfstk[i] = sbf.strike
        sfdip[i] = sbf.dip
        sfrak[i] = sbf.rake = 90.
        #sfrak[i] = mt.degrees(mt.atan2(sbf.cumslip[1],sbf.cumslip[0]))
        sfrds[i] = mt.hypot(sbf.hypo_deltaz,sbf.hypo_distance)
        sfrtm[i] = 0. #sftr
        sfhdr[i] = th #sfhd
        sfpot[i] = sbf.potency
        ncol += 2
    ncol *= ntw
    amat = surfw_matrix(fault.hypo.x,fault.hypo.y,fault.hypo.z,stas,stlo,stla,\
                        dt,nt,ib,tstrt,ibeg,iend,sflon,sflat,sfdep,sfstk,\
                        sfdip,sfrak,sfrds,sfhdr,sfrtm,sfpot,frq4,ntw,vmax,\
                        nrow,ncol,ntm,ns,nsf) 
    stgn = {}
    js = 0
    irow = 0
    for phase in stobs.keys():
        if phase != 'R' and phase != 'L':
            continue
        stgn[phase] = Stream()
        st = stobs[phase]
        for j,tr in enumerate(st):
            i = j+js
            trgn = Trace(header=tr.stats)
            #trgn.stats.starttime = otime+float(ibeg[i])*tr.stats.delta
            #if phase == 'L':
            #    print trgn.stats.station
            #    print '\t',trgn.stats.starttime,ibeg[i]
            #    print '\t',tr.stats.starttime
            trgn.stats.npts      = iend[i]-ibeg[i]+1
            #print trgn.stats.station ,trgn.stats.npts 
            trgn.data = np.zeros(trgn.stats.npts) # To satisfy obspy
            trgn.greens_funcs = amat[irow:irow+trgn.stats.npts,:]
            # Normalize the matrix rows but not the Green's functions
            amat[irow:irow+trgn.stats.npts,:] *= tr.stats.calib
            irow += trgn.stats.npts
            stgn[phase].append(trgn)
        js += len(st)
    return amat,stgn



def matrix_to_synth(fault,stgn,amat,th=None,normalize=False):
    # Assemble slip vector
    ntw = fault.ntw
    d = np.zeros(2*ntw*len(fault.subfaults))
    for i,sbf in enumerate(fault.subfaults):
        if th == None:
            times       = sbf.slip_rate[0].get_knots()[1:]
            th          = times[2]-times[1]
        slips_x = th*sbf.slip_rate[0].get_coeffs()[2:2+ntw]
        slips_y = th*sbf.slip_rate[1].get_coeffs()[2:2+ntw]
        for jtw in range(ntw):
            k = 2*(ntw*i+jtw)
            d[ k ] =  slips_x[jtw]*mt.cos(mt.radians(sbf.rake-45.)) +\
                      slips_y[jtw]*mt.sin(mt.radians(sbf.rake-45.)) 
            d[k+1] =  slips_x[jtw]*mt.cos(mt.radians(sbf.rake+45.)) +\
                      slips_y[jtw]*mt.sin(mt.radians(sbf.rake+45.)) 
                    
    for phase in stgn.keys():
        st = stgn[phase]
        for tr in st:
            if tr.stats.station == 'PMSA':
                print '3: ',tr.stats.station,phase,d[200],d[201]
            tr.data = np.dot(tr.greens_funcs,d)
            if normalize:
                tr.data /= tr.stats.calib
    return d

def bodymatrix_to_synth(amat,stgn,slip):
    d = np.dot(amat,slip)
    irow = 0
    for phase in stgn.keys():
        if phase != 'P' and phase != 'SH': continue
        st = stgn[phase]
        for tr in st:
            tr.data = d[irow:irow+tr.stats.npts]
            irow += tr.stats.npts
    return d


def SetFaultSlipbySlipVector(fault,x,th):
     for i,sbf in enumerate(fault.subfaults):
        times   = sbf.slip_rate[0].get_knots()
        trmx = times[-1]
        if len(times) != fault.ntw+4:
            print 'Subfault %d has times (%d) != fault.ntw (%d)' % \
                    (sbf.index,len(times),fault.ntw)
        slipx_rates = np.zeros(fault.ntw+4)
        slipy_rates = np.zeros(fault.ntw+4)
        for jtw in range(fault.ntw):
            k = 2*(fault.ntw*i+jtw)
            slipr_rake1 = x[ k ]/th
            slipr_rake2 = x[k+1]/th
            slipx_rates[2+jtw] = slipr_rake1*mt.cos(mt.radians(sbf.rake-45.)) +\
                                 slipr_rake2*mt.cos(mt.radians(sbf.rake+45.))
            slipy_rates[2+jtw] = slipr_rake1*mt.sin(mt.radians(sbf.rake-45.)) +\
                                 slipr_rake2*mt.sin(mt.radians(sbf.rake+45.))
        sbf.slip_rate = (UnivariateSpline(times, slipx_rates, s=0., k=1),
                         UnivariateSpline(times, slipy_rates, s=0., k=1))
        sbf.slip = (sbf.slip_rate[0].antiderivative(),
                    sbf.slip_rate[1].antiderivative())  
        sbf.cumslip = (sbf.slip[0]([trmx])[0],sbf.slip[1]([trmx])[0])                     
        
def surfsynth_fault(fault,th,otime,stobs,stgn,frq4):
    d = []
    for phase in stobs.keys():
        if phase != 'R'and phase != 'L':
            continue
        st = stobs[phase]
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
        stgn[phase] = Stream()
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
            #print tr.stats.station,':'
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
                            npw[itr],lcent[itr],td[:,itr],df,frq4[phase],tstart,\
                            c[:,itr],u[:,itr],q[:,itr],\
                            c1[:,itr],u1[:,itr],q1[:,itr],ityp,len(c[:,itr])) 
                syn += bmom*gs[0:tr.stats.npts]
    
            to.data = syn
            stgn[phase].append(to)
            d += list(to.data)
    return stgn,np.array(d) 

def surfsynth_fault0(fault,th,otime,st,frq4,gvel_win):
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
        #print tr.stats.station,':'
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


    
def SetFaultSlipInterpolants(fault,slip=None,stf=None,trmax=None):
    '''
    
    '''
    xdrct = 0.
    if not stf is None:
        vrmax,th = stf
        fault.vrmax = vrmax
        fault.th = th
        if trmax == None:
            trmax = 1.5*fault.length/vrmax+(fault.ntw+2)*th
    else: # Find trmax by loopoing over subfaults
        trmax = 0.
        for sbf in fault.subfaults:
            trmax = max(trmax,sbf.slip_rate[0].get_knots()[-1])
    ntw = fault.ntw
    for i,sbf in enumerate(fault.subfaults):
        if stf is None and not hasattr(sbf,'slip_rate'):
            raise Exception, 'SetFaultSlipInterpolants must have either stf or subfault.slip_rate'
        if not stf is None:       
            rup_dist = mt.hypot(sbf.hypo_deltaz,sbf.hypo_distance)
            phi = mt.radians(fault.strike-sbf.hypo_azimuth)
            if xdrct != 0. and mt.sin(phi) > 0.:
                drctvty = 0.5*(1.-mt.cos(2.*phi))
                #print 'azm=%6.2f, drctvty=%5.2f' % (sbf.hypo_azimuth,drctvty),mt.degrees(phi)
                rup_dist *= (1.+xdrct*drctvty)
            if rup_dist == 0.: # Special case if right at hypocenter, shift by th/10.
                times = [0.,0.1*th]
            else:
                times = [0.,rup_dist/vrmax]
        else:
            times   = sbf.slip_rate[0].get_knots()
        slipx_rates = [0.,0.]
        slipy_rates = [0.,0.]
        for jtw in range(ntw):
            if not stf is None:       
                times.append(times[-1]+th)
            else:
                td = times[jtw+3]-times[jtw+1] 
            k = 2*(ntw*i+jtw)
            if not slip is None:
                slipr_rake1 = 2.*slip[ k ]/td
                slipr_rake2 = 2.*slip[k+1]/td
                slipx_rates.append(slipr_rake1*mt.cos(mt.radians(sbf.rake-45.)) +\
                                   slipr_rake2*mt.cos(mt.radians(sbf.rake+45.)))
                slipy_rates.append(slipr_rake1*mt.sin(mt.radians(sbf.rake-45.)) +\
                                slipr_rake2*mt.sin(mt.radians(sbf.rake+45.)))
            else:
                slipx_rates.append(0.)
                slipy_rates.append(0.)
        # 
        if not stf is None:       
            times       += [times[-1]+th,trmax]
        slipx_rates += [0.,0.]        
        slipy_rates += [0.,0.]  
        sbf.slip_rate = (UnivariateSpline(times, slipx_rates, s=0., k=1),
                         UnivariateSpline(times, slipy_rates, s=0., k=1))
        sbf.slip = (sbf.slip_rate[0].antiderivative(),
                    sbf.slip_rate[1].antiderivative())  
        sbf.cumslip = (sbf.slip[0]([trmax])[0],sbf.slip[1]([trmax])[0])                     

def SetFaultSlipbyRate(fault,x,vrmax,th):
    ntw = fault.ntw
    trmx = fault.length/vrmax + (ntw-1)*th
    for i,sbf in enumerate(fault.subfaults):
        rup_dist = mt.hypot(sbf.hypo_deltaz,sbf.hypo_distance)
        if rup_dist == 0.: # Special case if right at hypocenter, shift by th/10.
            times = [0.,0.1*th]
        else:
            times = [0.,rup_dist/vrmax]
        slipx_rates = [0.,0.]
        slipy_rates = [0.,0.]
        for jtw in range(ntw):
            times.append(times[-1]+th) 
            k = 2*(ntw*i+jtw)
            slipr_rake1 = x[ k ]/th
            slipr_rake2 = x[k+1]/th
            slipx_rates.append(slipr_rake1*mt.cos(mt.radians(sbf.rake-45.)) +\
                               slipr_rake2*mt.cos(mt.radians(sbf.rake+45.)))
            slipy_rates.append(slipr_rake1*mt.sin(mt.radians(sbf.rake-45.)) +\
                               slipr_rake2*mt.sin(mt.radians(sbf.rake+45.)))
        times       += [times[-1]+th,trmx]
        slipx_rates += [0.,0.]        
        slipy_rates += [0.,0.]  
        sbf.slip_rate = (UnivariateSpline(times, slipx_rates, s=0., k=1),
                         UnivariateSpline(times, slipy_rates, s=0., k=1))
        sbf.slip = (sbf.slip_rate[0].antiderivative(),
                    sbf.slip_rate[1].antiderivative())  
        sbf.cumslip = (sbf.slip[0]([trmx])[0],sbf.slip[1]([trmx])[0])                     

def read_i_inv(ffi_dir='.',filename='i_inv'):
    lines = open(ffi_dir+'/'+filename,'r').readlines()
    tshft = {}
    # Time window for body wave data                                                                  
    lt  = int(lines[2].split()[0])
    # Length of computed Green's functions                                                            
    nt  = int(lines[2].split()[1])
    # Pad to match inv_lh_mom                                                                         
    if mt.log(nt)/mt.log(2) % 1 != 0.:
        nt = 2**int(mt.log(nt)/mt.log(2)+1)
    elon,elat,edep = [float(x) for x in lines[1].split()[0:3]]
    tshft['P'] = float(lines[2].split()[2])
    tshft['SH'] = float(lines[2].split()[3])
    ifil =   int(lines[5].split()[0])
    f1   = float(lines[5].split()[1])
    f2   = float(lines[5].split()[2])
    dtstep,vrmax,rise_time = [float(x) for x in lines[6].split()[1:4]]
    ntw = int(rise_time/dtstep+0.5)
    t12 = float(lines[7].split()[1])
    ia = int(lines[15].split()[0])
    x0,y0,z0 = [float(x) for x  in lines[ia+15].split()[0:3]]
    return lt,nt,tshft,dtstep,ifil,f1,f2,t12,vrmax,ntw,ia

   ###
if __name__ == "__main__":
    output = main()
    sys.exit(0) 