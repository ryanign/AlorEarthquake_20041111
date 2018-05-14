c      parameter(ms=500,mt=500,msts=90,mxyzt=2000,mxyz=500)
c      integer im(ms),ib(ms),ic(ms),isurf(ms),nt(ms),
c     &          nd(ms),igr(mxyzt),lcent(ms),np(ms),is(ms)
c      real az(ms),az2(ms),del(ms),p(ms),g(ms),ts0(ms),stla(ms),stlo(ms),
c     &     lp(ms),hp(ms),dt(ms),ss(mt),wt(ms),fc(ms),s(mt,ms),
c     &     max_obs(ms),tgreen(3),txa(10),ww(10),si(mt,ms),
c     +	   fx(2*mt),xt(mxyzt),td(2*mt,ms),
c     &     gs(mt,mxyzt,ms),
c     +     c(2*mt,ms),u(2*mt,ms),q(2*mt,ms),
c     &     c1(2*mt,ms),u1(2*mt,ms),q1(2*mt,ms),
c     &     x(mxyz),y(mxyz),z(mxyz),str(mxyz),dip(mxyz),rak(mxyz),
c     &     gr1(mt),area(mxyz),bmo(mxyz),xa(mxyzt),xr(mxyzt),
c     &     xz(mxyzt),d(mxyzt),sx(2*mt),px(2*mt),qx(2*mt),
c     &     rx(2*mt)
c
c      complex zp0(10),zz(10)
c      integer ist,isg,isl,iss,js,jfc,intvl,ifil5,maxnd
c      real ff1,ff2,evlo,evla,evdp,vrup
c      character*80 evname,grfl
c      character*80 nam(ms),excfnm,phvfnm,exf
c      character*1 comp(10),atyp(2)
c      logical itshere
c      data ww/0,0,0,0,0,1,1,0,0,0/
c      data fc /ms*1.0/
c      data atyp /'L','R'/
c      
cc     -- read inversion parameters --------------------------------------
c      read(5,'(a80)') evname
c      read(*,*) evlo,evla,evdp
c      read(5,*) tn,tgreen(1),txa(1),txa(2),txa(5),txa(6)
c      read(5,'(a80)') excfnm
c      read(5,*) ifil5,ff1,ff2, maxnd,vrup
c      read(5,*) ifil,f1,f2
c      read(5,*) ntstep,dtstep,v0,v1
c      read(5,*) mxs,t1,t2
c
cc Read the rupture model
c      open(43,file='o_inv-nta',action='read')
c      do 1234 k=1,mxyzt
c        read(43,'(7(f8.3,x),3(e12.5,x))',end=1235) 
c     1              xt(k),x(k),y(k),z(k),
c     2              str(k),dip(k),rak(k),d(k),area(k),bmo(k)
c        xt(k) = xt(k)-dtstep
c 1234 continue
c 1235 close(43)
c      nta = k-1
c      write(*,*) nta,' point sources read'
c  
c      call read_data(ns,nd,nam,isurf,ib,ts0,stla,stlo,
c     &       dt,s,nt,ifil,f1,f2,txa,msts,ms,mt)
c
c      call surfw_excite(excfnm)
c
cc     -- compute the green's functions ----------------------------------
cc     -- done in a separate loop so that we save and retrieve the -------
cc     -- the Green's functions ------------------------------------------
c      write(*,*) "Start computing Green's functions"
c      ist=0
c      isg=0
c      isl=0
c      iss=0
c      call surfw_phv(evlo,evla,nam,stlo,stla,ib,nt,dt,np,lcent,td,
c     &                   c,u,q,c1,u1,q1,ns,2*mt)
c
c      do 70 i=1,ns
c        write(*,'("Station: ",a7)') nam(i)
cc        f=g(i)*ds(i)
c        if (isurf(i).ne.1) then
c          write(*,*) 'Cannot come here!'
c          stop
c        else if (isurf(i).eq.1) then
c
cc         -- compute surface wave Green's functions ---------------------
c	  iss=iss+1
c	  is(i)=iss
cc         -- ityp:  1=Love, 2=Rayleigh ----------------------------------
c	  ityp=ib(i)-5
c          df=1.0/(2.*(lcent(i)-1)*dt(i))
c
c	  do kk=1,nta
c	    depth=z(kk)
cc           -- we need lat/lon for every gridpoint-----------------------
c            ela=y(kk)
c	    elo=x(kk)
c            call surfw_calc(ela,elo,stla(i),stlo(i),depth,t1,str(kk),
c     &                      dip(kk),rak(kk),np(i),lcent(i),td(1,i),df,
c     &                      f1,f2,ts0(i)-xt(kk),c(1,i),u(1,i),q(1,i),
c     &                      c1(1,i),u1(1,i),q1(1,i),ityp,
c     &                      gs(1,kk,is(i)),2*mt)
c	  enddo
c        endif
c70    continue
c      write(*,*) "Finished computing Green's functions"
cc     -------------------------------------------------------------------
c      
c      call surfw_calcsyn(nam,ib,nd,d,gs,is,ts0,dt,bmo,nta,ns,mt,mxyzt)
c      end
c#
      subroutine read_data(ns,nd,nam,isurf,ib,ts0,stla,stlo,
     &       dt,s,nt,ifil,f1,f2,txa,msts,ms,mt)
      implicit real (a-h,o-z)
      implicit integer (i-n)
      character*80 evname,nam(ms),grfl
      real az(ms),az2(ms),del(ms),p(ms),g(ms),ts0(ms),stla(ms),
     &     stlo(ms),hp(ms),ds(ms),dt(ms),ss(mt),ww(10),wt(ms),fc(ms),
     &     max_obs(ms),s(mt,ms),txa(10)
      integer ix0(ms),im(ms),ib(ms),ic(ms),isurf(ms),lp(ms),nt(ms),
     &     nd(ms)
      data ww/0,0,0,0,0,1,1,0,0,0/
Cf2py intent(in) ifil,f1,f2,txa,ms,mt
Cf2py intent(out) ns,nd,nam,isurf,ib,ts0,stla,stlo,dt,s,nt
Cf2py depend(ms) nd,nam,isurf,ib,ts0,stla,stlo,dt,s,nt
Cf2py depend(mt) s

      ist=0
      isg=0
      isl=0
      iss=0
      js=1
      jfc=1
      read(1,'(a80)') evname
      do 100, ii=1,ms
        fc(ii) = 1
        read(1,'(a7,x,a8)',end=999) nam(js),grfl
c	print*, 'Processing ',nam(js),js,ms
        read(1,*) az(js),az2(js),del(js),p(js),g(js),ix0(js)
        read(1,*) im(js),ib(js),ic(js)
        isurf(js)=0
        ts0(js)=0.0
        if(ib(js).eq.6 .or. ib(js).eq. 7) then
c         -- ib = 6 or 7 means that we are looking at surface waves -----
          iss=iss+1
	  if(iss.gt.msts) then
             print*,'Too many surface wave data'
             stop
          endif
          isurf(js)=1
          stla(js)=az(js)
          stlo(js)=az2(js)
          lp(js)=p(js)
          hp(js)=del(js)
          ts0(js)=g(js)
        endif
        if(im(js).eq.1) then
c         -- read poles and zeros ---------------------------------------
          read(1,*) izp,izz,ip
          read(1,*) ds(js),a0
          do 11 k=1,izp
 11          read(1,*) temp1,temp2
          do 12 k=1,izz
 12          read(1,*) temp1,temp2
        else
c         -- read electro/mechanical constants --------------------------
          read(1,*) ds(js),ts,tg,hs,hg,cpl
        endif
c       -- read the data ------------------------------------------------
        read(1,*) xm,intvl,dt(js)
	if(intvl.gt.mt) then
	  write(*,*) 'Too many datapoints'
	  stop
        endif
        read(1,*) (ss(i),i=1,intvl)
c       -- filter the data in case of surface waves ---------------------
	if( (ib(js).eq.6 .or. ib(js).eq. 7 ) .and. ifil.eq.1) then
	  call taper(ss,intvl,.05,.05,ss)
	  call filter(ss,ss,intvl,dt(js),f1,f2,3,"b")
	elseif(ib(js) .eq. 5 .and. ifil5 .eq. 1) then
	  call taper(ss,intvl,.05,.05,ss)
	  call filter(ss,ss,intvl,dt(js),ff1,ff2,3,"b")
        endif
c       -- number of green function points to next power of 2 -----------
        ndum=intvl
	nt(js)=log(ndum*1.)/log(2.)
	nt(js)=2**nt(js)
	if(nt(js).lt.ndum) nt(js)=nt(js)*2
	if(nt(js).gt.mt .and. ib(js).ne.5) then
	  write(*,'("Greens function has too many points ")') nt(js)
	  stop
        endif
c       -----------------------------------------------------------------

c Left from instrument response
        ds(js)=ds(js)*1.e4
c       -----------------------------------------------------------------
c       -- remove stations outside the time window or zero weight -------
        ttn=intvl*dt(js)
        nn=ttn/dt(js)
        if(ix0(js).le.intvl) then
          fc(js)=fc(jfc)
	  jfc=jfc+1
          if(fc(js).ne.0. .and. ww(ib(js)) .ne. 0.) then
            if(ib(js).lt.5) then
	      ix0(js)=ix0(js)+nint(txa(ib(js))/dt(js))
	    else
	      ts0(js)=ts0(js)-txa(ib(js))-ix0(js)*dt(js)
 	    endif
	    i=1
	    if(ib(js).lt.5) then
	      i1=ix0(js)+i
	    else
	      i1=i
            endif
	    max_obs(js)=0.0
	    if(ib(js).ne.10) then
               do 13 while (i.le.nn .and. i1.lt.intvl)
                 if(ib(js).lt.5) then
	           i1=ix0(js)+i
	         else
	           i1=i
                 endif
                 if(i1.gt.0) then
	           s(i,js)=ss(i1)*xm
		   max_obs(js)=amax1(max_obs(js),abs(s(i,js)))
                 else
	           s(i,js) = 0.0
                 endif
	         i=i+1
13             continue
	    else
	       s(1,js)=ss(1)
	       max_obs(js)=abs(ss(1))
	       i=i+1
	    endif
	    if(iwt.eq.0) then
              wt(js)=1.e6/ds(js)*fc(js)
	    elseif(iwt.eq.1) then
	      al=log(max_obs(js))/max_obs(js)
	      wt(js)=al*1.e6
	    else
	      wt(js)=1.e6/max_obs(js)
	    endif
	    nd(js)=i-1
	    wt(js)=ww(ib(js))*wt(js)/nd(js)
	    if( (ib(js).eq.6.or.ib(js).eq.7) .and. nd(js).gt.0) js=js+1
          endif
        endif
c       -----------------------------------------------------------------
100   continue
999   ns=js-1
      write(*,'(i4," stations read")') ns
      return
      end

c#
      subroutine surfw_calcsyn(nam,ib,nd,d,gs,is,ts0,dt,bmo,nta,ns,
     &                         mt,mxyzt)
      implicit real (a-h,o-z)
      implicit integer (i-n)
      integer ib(ns),nd(ns),is(ns)
      real d(nta),gs(mt,mxyzt,ns),bmo(nta),sy(mt),ts0(ns),dt(ns)
      character*80 nam(ns),sac_syn
      character*1 comp(10)
Cf2py intent(in) nam,ib,nd,d,gs,is,ts0,dt,bmo,nta,ns,mt,mxyzt

      data comp /'P','S','S','V','H','L','R','R','G','G'/
c    Calculate synthetics
      do 700 i=1,ns
	idx=index(nam(i),' ')-1
	if(idx.lt.0) idx=7
	sac_syn=nam(i)(1:idx)//'.'//comp(ib(i))//'.nnewsyn'
 	if(ib(i).ne.5 .and. ib(i).ne.10) then
	  do 710 j=1,nd(i)
	    sy(j)=0.0
	    do 720 kk=1,nta
	      sy(j)=sy(j)+d(kk)*gs(j,kk,is(i))*bmo(kk)
c              write(*,*) j,kk,is(i),gs(j,kk,is(i)),d(kk),bmo(kk)
720         continue
710       continue
          endif
          call wsac1(sac_syn,sy,nd(i),ts0(i),dt(i),nerr)
700   continue
      return
      end
c#
      subroutine surfw_matrix(elon,elat,edep,stas,stlo,stla,dt,nt,ib,
     &                     tstrt,ibeg,iend,sflon,sflat,sfdep,sfstk,
     &                     sfdip,sfrak,sfrds,sfhdr,sfrtm,sfpot,
     &                     frq4,ntw,vmax,nrow,ncol,mt,amat,ns,nsf)     
c------------------------------------------------------------------------------
c This routine calculates a surface wave Green's functions matrix. 
c The output time series is bandpass filtered between frequencies f1,f2 using 
c a 3-pole butterworth filter
c
c Input
c     elat,elon:    latitude,longitude of point source
c     stla,stlo:    latitude,longitude of stations
c     depth:        depth of point source
c     t1:           point source slip rise time 
c     sfstk:        strike of point source fault mechanism
c     sfdip:        dip     ''        ''
c     sfrak:        rake    ''        ''
c     np:           power of 2 of time series length 
c     lcent:        time series length (lcent = 2**np)
c     td:           array of periods for dispersion curve
c     df:           frequency interval for fft
c     f1,f2:        bandpass filter corner frequencies    
c     starttime:    start time of output time series w.r.t. origin
c     c,u,q:        phase velocity parameters for fundamental
c     c1,u1,q1:     phase velocity parameters for first overtone (applied
c                   only for Love wave (ityp=1)
c     sx,px,rx:     I don't know what these are
c     ityp:         surface wave type (1:Love,2:Rayleigh)
c     mt:           dimensions of arrays:c,u,q,c1,u1,q1,gs,td,sx,px,qx,rx
c Output
c     gs:           array containing surface wave Green's function
c The units returned are microns/1e25dyne.cm (as in kikuchi's code -
c see surfw_simple). 
c------------------------------------------------------------------------------
      implicit real (a-h,o-z)
      implicit integer (i-n)
      real gr1(2*mt),tstrt(ns),
     &     sx(2*mt),px(2*mt),qx(2*mt),rx(2*mt),gs(2*mt),frq4(4),
     &     stla(ns),dt(ns),tsrt(ns),sflon(nsf),stlo(ns),
     &     sflat(nsf),sfdep(nsf),sfstk(nsf),sfdip(nsf),sfrak(nsf),
     &     sfrds(nsf),sfhdr(nsf),amat(nrow,ncol),sfrtm(nsf),sfpot(nsf)
      integer ibeg(ns),iend(ns),ib(ns),nt(ns),lcent(ns),np(ns),idx
      real td(2*mt,ns),c(2*mt,ns),u(2*mt,ns),q(2*mt,ns),
     &     c1(2*mt,ns),u1(2*mt,ns),q1(2*mt,ns)
      character*(*) stas(ns)
Cf2py intent(in) elat,elon,edep,stas,stlo,stla,dt,nt,ib,ibeg,iend,sflon,sflat
Cf2py intent(in) sfdep,sfstk,sfdip,sfrak,sfrds,sfhdr,frq4stla,stlo,depth,t1
Cf2py intent(in) str,dip,rak,np,lcent,td,df,frq4,nrow,tstrt
Cf2py intent(out) amat
Cf2py depend(nrow) amat
Cf2py depend(ncol) amat
Cf2py depend(ns)   stas,stlo,stla,dt,nt,ib,tsrt,ibeg,iend,tstrt
Cf2py depend(nsf)  sflon,sflat,sfdep,sfstk,sfdip,sfrak,sfrds,sfhdr,sfrtm,sfpot
       amat(:,:) = 0.0
c* First caculate phase velocites between epicenter and stations
c      write(*,*) elat,elon,edep,stas(1),stlo(1),stla(1),ib(1),nt(1),
c     &       mt,nrow,ncol,nsf
c      write(*,*) 'call surfw_phv'
      if (vmax.eq.0..and.ntw.gt.1) then
        write(*,*) 'surfw_matrix: vmax=0., ntw>1 - cannot',
     &              ' do multiple rupture times yet!'
        stop
      endif  
      call surfw_phv(elon,elat,stas,stlo,stla,ib,nt,
     &                  dt,np,lcent,td,c,u,q,c1,u1,q1,ns,2*mt)
c      write(*,*) 'returned from surfw_phv'
      irow = 1
      do is=1,ns
c* Set up excitation 
        if (ib(is).eq.7) then
            ityp = 2
        else if (ib(is).eq.6) then
            ityp = 1   
        endif
        df = 1.0/(2.*(lcent(is)-1)*dt(is))
        icol = 1 
        mtw = max(1,ntw)
ccc  Loop over subfaults
        do jsf=1,nsf
ccc  Loop over time windows
          do itw=1,mtw
            if (vmax.eq.0.) then
              tshft = sfrtm(jsf)
            else
              tshft = sfrds(jsf)/vmax+(itw-1)*sfhdr(jsf)
            endif
c            write(*,*) '#1 ',sfrtm(jsf),tshft,sfhdr(jsf)
            tstart = tstrt(is)-tshft
ccc  Loop over rakes
            do jrk=1,2
              do idx =1,2*mt
                gs(idx) = 0.
              enddo
              rake = sfrak(jsf)+(jrk-1.5)*90.
c              write(*,*) 'call getexcite'
              call getexcite(sfdep(jsf),td(1,is),sx,px,qx,rx,
     &                            lcent(is),ityp)
c              write(*,*) 'returned from getexcite'
c              return
c              write(*,*) 'call surfw_simple ',df,is,jsf
c              if (jsf.eq.1) then
c                write(*,*) '#2 ',sflat(jsf),sflon(jsf),stla(is),
c     &                 stlo(is),
c     &                    sfstk(jsf),sfdip(jsf),rake,np(is),df,
c     &                    tstrt(is),
c     &                  q(1,is),sx(1),px(1),qx(1),rx(1)
c               endif
c              if (jsf.eq.nsf) then
c                write(*,*) stas(is),' call surfw_simple ',ityp,ib(is),tstart
c              endif
              call surfw_simple(sflat(jsf),sflon(jsf),stla(is),
     &                  stlo(is),sfstk(jsf),sfdip(jsf),rake,np(is),df,
     &                  frq4,tstart,sfhdr(jsf),ityp,c(1,is),u(1,is),
     &                  q(1,is),sx,px,qx,rx,gs)
c        if (ityp.eq.1.and.jsf.eq.1) then
c          write(*,*) ' call surfw_simple #1 ',sflat(jsf),sflon(jsf),
c     &        stla(is),stlo(is),sfstk(jsf),sfdip(jsf),rake,np(is),
c     &        df,frq4(1),tstart,sfhdr(jsf),
c     &                          c1(1,is),u1(1,is),q1(1,is),sx(1),px(1),
c     &               qx(1),rx(1)
c          if (jsf.eq.1) then
c            write(*,*) 'gs: ',is,jsf,itw,jrk,(gs(ibeg(is)+j),j=35,39)
c          endif
c            write(*,*) 'returned from surfw_simple'
c            return
              if (ityp.eq.11) then
c                write(*,*) 'call getexcite'
      	        call getexcite(sfdep(jsf),td(1,is),sx,px,qx,rx,
     &	                          lcent(is),ityp+2)
c                write(*,*) 'call surfw_simple (overtone)',is,jsf
    	        call surfw_simple(sflat(jsf),sflon(jsf),stla(is),
     &           	stlo(is),sfstk(jsf),sfdip(jsf),rake,np(is),
     &                  df,frq4,tstart,sfhdr(jsf),ityp+2,c1(1,is),
     &                  u1(1,is),q1(1,is),sx,px,qx,rx,gr1)
c                write(*,*) 'returned from surfw_simple (overtone)'
                do j=1,lcent(is)
     	          gs(j)=gs(j)+gr1(j)
                enddo
              endif
c            write(*,*) 'fill amat ',irow,irow+iend(is)-ibeg(is),', gs ',
c     &                          ibeg(is),iend(is),icol
c ibeg(is) has been caclulated using zero-based indexing, so need to +1
              do jrow=irow,irow+iend(is)-ibeg(is)
ccc The 1.e-18 is from Kikuchi (1.e-25) a dyne-cm to Nm conversion (1.e7)
                  amat(jrow,icol) = sfpot(jsf)*1.e-18*
     &                            gs(ibeg(is)+jrow-irow+1)
              enddo
c              write(*,*) 'amat ',is,icol,irow,irow+iend(is)-ibeg(is)
c              write(*,*) (amat(jrow,icol),jrow=irow,
c     &                          irow+iend(is)-ibeg(is))
              icol = icol+1
            enddo
          enddo
        enddo
        irow = irow+iend(is)-ibeg(is)+1
      enddo
c      write(*,*) 'done'
      return
      end

c#
c#
      subroutine surfw_calc(ela,elo,stla,stlo,depth,t1,str,dip,rak,
     &                      np,lcent,td,df,frq4,starttime,c,u,q,
     &                      c1,u1,q1,ityp,gs,mt)
c------------------------------------------------------------------------------
c This routine calculates surface wave Green's functions. 
c The output time series is bandpass filtered between frequencies f1,f2 using 
c a 3-pole butterworth filter
c
c Input
c     ela,elo:      latitude,longitude of point source
c     stla,atlo:    latitude,longitude of station
c     depth:        depth of point source
c     t1:           point source slip rise time 
c     str:          strike of point source fault mechanism
c     dip:          dip     ''        ''
c     rak:          rake    ''        ''
c     np:           power of 2 of time series length 
c     lcent:        time series length (lcent = 2**np)
c     td:           array of periods for dispersion curve
c     df:           frequency interval for fft
c     f1,f2:        bandpass filter corner frequencies    
c     starttime:    start time of output time series w.r.t. origin
c     c,u,q:        phase velocity parameters for fundamental
c     c1,u1,q1:     phase velocity parameters for first overtone (applied
c                   only for Love wave (ityp=1)
c     sx,px,rx:     I don'r know what these are
c     ityp:         surface wave type (1:Love,2:Rayleigh)
c     mt:           dimensions of arrays:c,u,q,c1,u1,q1,gs,td,sx,px,qx,rx
c Output
c     gs:           array containing surface wave Green's function
c The units returned are microns/1e25dyne.cm (as in kikuchi's code -
c see surfw_simple). 
c------------------------------------------------------------------------------
      implicit real (a-h,o-z)
      implicit integer (i-n)
      real c(mt),u(mt),q(mt),c1(mt),u1(mt),q1(mt),gr1(mt),td(mt),
     &     sx(mt),px(mt),qx(mt),rx(mt),gs(mt),frq4(4)
Cf2py intent(in) ela,elo,stla,stlo,depth,t1,str,dip,rak,np,lcent,td,df,frq4
cCf2py intent(in) ela,elo,stla,stlo,depth,t1,str,dip,rak,np,lcent,td,df,f1,f2
Cf2py intent(in) starttime,c,u,q,c1,u1,q1,ityp,mt
Cf2py intent(out) gs
Cf2py depend(mt) gs,c,u,q,c1,u1,q1
c      write(*,*) 'call getexcite'
c      call flush()
      call getexcite(depth,td,sx,px,qx,rx,lcent,ityp)
c      write(*,*) '1# ',td(1),sx(1),px(1),qx(1),rx(1)
c      call flush()
      call surfw_simple(ela,elo,stla,stlo,str,dip,
     +			      rak,np,df,frq4,starttime,t1,
     +			      ityp,c,u,q,sx,px,qx,rx,gs)
c      if (ityp.eq.1.and.ela.lt.9.23) then
c      if (ityp.eq.1) then
c        write(*,*) ' call surfw_simple #2 ',ela,elo,stla,stlo,
c     &    str,dip,rak,np,df,frq4(1),starttime,t1,
c     &                          c1(1),u1(1),q1(1),sx(1),px(1),
c     &             qx(1),rx(1)
c        write(*,*) 'gf: ',(gs(j),j=15,18)
c      endif
c      write(*,*) '2# ',c(1),u(1),q(1),px(1),qx(1),rx(1),gs(1)
c      call flush()
      if(ityp.eq.11) then
  	call getexcite(depth,td,sx,px,qx,rx,lcent,ityp+2)
  	call surfw_simple(ela,elo,stla,stlo,str,dip,
     +				rak,np,df,frq4,starttime,t1,ityp+2,
     &                          c1,u1,q1,sx,px,qx,rx,gr1)
        do j=1,lcent
 	   gs(j)=gs(j)+gr1(j)
        enddo
      endif
      return
      end

c#
      subroutine surfw_phv(evlo,evla,stas,stlo,stla,ib,nt,dt,
     &                         np,lcent,td,c,u,q,c1,u1,q1,ns,mt)
c------------------------------------------------------------------------------
c This routine calculates surface wave phase velocities for the Harvard model
c Input
c     evla,evlo:    hypocenter latitude,longitude
c     stas:         a character array containing station names
c     stlo,stla:    latitudes,longitudes of stations
c     ib:           surface wave type (6:Love, 7:Rayliegh)
c     --     6     *  Love wave     HT                                 --
c     --     7     *  Rayleigh wave UD                                 --
c     nt,dt:        input time series length and sample interval
c     ns:           number of stations
c Output
c     np:           power of 2 of time series length 
c     lcent:        spectrum length (lcent = 1+0.5*2**np)
c     td:           array of periods for dispersion curve
c     c,u,q:        phase velocity parameters for fundamental
c     c1,u1,q1:     phase velocity parameters for first overtone 
c
c Given a time series of length nt and sample interval dt for which waveforms
c are required, this subroutine will return the next greatest power of 2 (np)
c that contains nt, as well as the number of postive frequencies in the 
c spectrum for synthesizing the output waveform.
c------------------------------------------------------------------------------
      implicit real (a-h,o-z)
      implicit integer (i-n)
      integer ib(ns),nt(ns),lcent(ns),np(ns)
      real stla(ns),stlo(ns),td(mt,ns),dt(ns),
     +     c(mt,ns),u(mt,ns),q(mt,ns),
     &     c1(mt,ns),u1(mt,ns),q1(mt,ns)
      character*(*) stas(ns)
Cf2py intent(in) evlo,evla,stas,stlo,stla,ib,nt,dt,ns,mt
Cf2py intent(out) np,lcent,td,c,u,q,c1,u1,q1
cf2py depend(mt) c,u,q,c1,u1,q1,td
cf2py depend(ns) c,u,q,c1,u1,q1,td
      do 60 i=1,ns
        call binfd(nt(i),lx1,np(i))
        np(i)=np(i)+1
        lcent(i)=lx1+1
        df=1.0/(2.*lx1*dt(i))
        if(lcent(i).gt.mt) then
           print*,'lcent is larger than mtg: ',lcent(i),mt
           stop
        endif
        do j=1,lcent(i)
           fx=(j-1)*df
           if(j.gt.1) then
              td(j,i)=1/fx
           else
              td(j,i)=99999.
           endif
        enddo
        ityp=ib(i)-5
        call phvpath0(evla,evlo,stla(i),stlo(i),td(1,i),lcent(i),ityp,
     &                c(1,i),u(1,i),q(1,i))
c        call phvpath0(evla,evlo,stla(i),stlo(i),td(1,i),lcent(i),ityp+2,
c     &                c1(1,i),u1(1,i),q1(1,i))
c        print *, 'done with phv for ',stas(i)
 60   continue
      return
      end
c#
      subroutine surfw_excite(excfnm)
c------------------------------------------------------------------------------
c This subroutine calls rdexcite for the Rayliegh wave fundamental and
c the Love wavefundamental plus 1st overtone.
c The precalculated excitation functions need to be tabulated in files 
c with names following the convention:
c           -- excfnm-[L/R]-imode-idep.exf                               --
c------------------------------------------------------------------------------
      implicit real (a-h,o-z)
      implicit integer (i-n)
      character*(*) excfnm
      character*256 exf
Cf2py inent(in) excfnm
      idx=index(excfnm,' ')-1
      exf=excfnm(1:idx)//'-L-0.exf'
      call rdexcite(exf,1)
      exf=excfnm(1:idx)//'-R-0.exf'
      call rdexcite(exf,2)
      exf=excfnm(1:idx)//'-L-1.exf'
      call rdexcite(exf,3)
      print*,'Surfw: Read excitation functions'
      return
      end
c#
c------------------------------------------------------------------------------
      subroutine filter(x,y,nx,dt,f1,f2,n,lhorb)
c     butterworth type filter, if lhorb="l","h",or"b", it performs
c     low-pas, high-pass, and band-pass filter, respectively
c     if lhorb="l", then f2 is the cut-off freq., and f1 is dummy
c     if lhorb="h", then f1 is the cut-off freq., and f2 is dummy
c     if lhorb="b", then f1 is low-, and f2 is high-frequency cut off.
c     n is the order, usually 3 or 4
c     x(i) is input and y(i) is output, nx is #of points and dt is dt.
c     the filter response is in w(i) in common, df is freq. increment
c     July, 1990, hiroo kanamori
c     adapted to use function call to butter, HKT 00/06/05

      character*1 lhorb
      parameter (ndim=24000)
c      parameter (ndim=240000)
      complex c(ndim),butter
      real x(*),y(nx)
Cf2py intent(in) x,nx,dt,f1,f2,n,lhorb
Cf2py intent(out) y
Cf2py depend(nx) y

      call binfd(nx,lx1,np)
      lx1=lx1*2
      if(lx1.gt.ndim)  then
         write(*,*)  ' data too large, increase ndim in sub.filter.f '
         stop
      endif
      np=np+1
      nxp=nx+1
      lhalf=lx1/2
      lcent=lhalf+1

      do i=nxp,lx1
         x(i)=x(nx)
      enddo

      do  i=1, lx1
         c(i)=cmplx(x(i),0.0)
      enddo
      call  cool(np,c,-1.)
      df=1.0/(float(lx1)*dt)

      do  i=1, lcent
        f=float(i-1)*df
        c(i)=c(i)*butter(lhorb,n,f,f1,f2)
      enddo

      call conj(c,lcent)
      call cool(np,c,1.)
      do i=1, nx
         y(i)=real(c(i))/float(lx1)
      enddo
      return
      end
c# 
      SUBROUTINE TAPER (A, N, START, END, B)

      DIMENSION A(N), B(N)
Cf2py intent(in) A,N,START,END
Cf2py intent(out) B
Cf2py depend(n) B,A

      AN = N

      M1 = AN*START+0.5

      M2 = M1 + 1

      IF(M1) 10, 10, 11

   11 ANG = 3.1415926 / FLOAT(M1)

      DO 12 I=1, M1

      XI = I

      CS = (1.0-COS(XI*ANG))/2.0

   12 B(I) = A(I)*CS

   10 M3 = AN*END+0.5

      M5 = N-M3

      M4 = M5 + 1

      IF(M3) 13, 13, 14

   14 ANG = 3.1415926 / FLOAT (M3)

      DO 15 I=M4,N

      XI = I-N-1

      CS = (1.0-COS(XI*ANG))/2.0

   15 B(I) = A(I)*CS

   13 DO 16 I=M2, M5

   16 B(I) = A(I)

      RETURN

      END
c#
      SUBROUTINE CONJ(C,LX1)

C     LX1 = 2**N/2 + 1

      COMPLEX C(*)

      NM1 = LX1 - 1

      DO 30 I=1, NM1

      NT = LX1 + I

      NU = LX1 - I

      C(NT) = CONJG (C(NU))

   30 CONTINUE

      C(LX1) = CMPLX (REAL(C(LX1)),0.0)

      RETURN

      END
c#
      SUBROUTINE BINFD(LX,LX1,N)
c      lx should be eq. to less than 2**20 (1048576)
C     2**N.LE.LX.LT.2**(N+1)

      DO 3 I=1,20

      IND = 2**I

      IF(LX-IND) 4, 5, 3

    3 CONTINUE

      N = 100
      write(*,*)  'Warning  lx too large!'

      RETURN

    4 CONTINUE

      N = I-1

      GO TO 6

    5 CONTINUE

      N = I

    6 CONTINUE

      LX1 = 2**N

      RETURN

      END

c#
      SUBROUTINE COOL(NN,DATAI,SIGNI)
      DIMENSION DATAI(*)
      N=2**(NN+1)
      J=1
      DO 5 I=1,N,2
      IF(I-J)1,2,2
    1 TEMPR=DATAI(J)
      TEMPI=DATAI(J+1)
      DATAI(J)=DATAI(I)
      DATAI(J+1)=DATAI(I+1)
      DATAI(I)=TEMPR
      DATAI(I+1)=TEMPI
    2 M=N/2
    3 IF(J-M)5,5,4
    4 J=J-M
      M=M/2
      IF(M-2)5,3,3
    5 J=J+M
      MMAX=2
    6 IF(MMAX-N)7,10,10
    7 ISTEP=2*MMAX
      THETA=SIGNI*6.28318531/FLOAT(MMAX)
      SINTH=SIN(THETA/2.)
      WSTPR=-2.0  *SINTH*SINTH
      WSTPI= SIN(THETA)
      WR=1.
      WI=0.
      DO 9 M=1,MMAX,2
      DO 8 I=M,N,ISTEP
      J=I+MMAX
      TEMPR=WR*DATAI(J)-WI*DATAI(J+1)
      TEMPI=WR*DATAI(J+1)+WI*DATAI(J)
      DATAI(J)=DATAI(I)-TEMPR
      DATAI(J+1)=DATAI(I+1)-TEMPI
      DATAI(I)=DATAI(I)+TEMPR
    8 DATAI(I+1)=DATAI(I+1)+TEMPI
      TEMPR=WR
      WR=WR*WSTPR-WI*WSTPI+WR
    9 WI=WI*WSTPR+TEMPR*WSTPI+WI
      MMAX=ISTEP
      GO TO 6
   10 RETURN
      END
 

c#
      complex function butter(lhorb,n,f,f1,f2)
c     butterworth type filter, if lhorb="l","h",or"b", it performs
c     low-pas, high-pass, and band-pass filter, respectively
c     if lhorb="l", then f2 is the cut-off freq., and f1 is dummy
c     if lhorb="h", then f1 is the cut-off freq., and f2 is dummy
c     if lhorb="b", then f1 is low-, and f2 is high-frequency cut off.
c     n is the order, usually 3 or 4
c     July, 1990, hiroo kanamori
      character*1 lhorb
      real*8  a1, a2

      nfp=4*n
      if(lhorb.eq.'b') then 
         a2=1./(1.+(f/f2)**nfp)
c        a1=(f**nfp)/(f**nfp+f1**nfp)
         a1=(f/f1)**nfp/((f/f1)**nfp+1.)
      else if(lhorb.eq.'l') then
         a2=1./(1.+(f/f2)**nfp)
	 a1=1.
      else if(lhorb.eq.'h') then
	 a2=1.
         a1=(f/f1)**nfp/((f/f1)**nfp+1.)
      else 
         write(*,*) 'lhorb should be either l, h, or b'
         stop
      endif
      butter=cmplx(a1*a2,0.)
      return
      end

c#



c------------------------------------------------------------------------
      subroutine surfw_simple(evla,evlo,stla,stlo,str,dip,rak,np,df,
     +			      frq4,tst,ts0,ityp,c,u,q,sr1,pr1,qr1,ra,x)
c     ----------------------------------------------------------------
c     -- This subroutine computes waveforms using mode-summation     -
c     -- using the phasevelocities and excitation functions that are -
c     -- passed on in the subroutine call. The surface waves are     -
c     -- high-pass filtered at 333.sec                               -
c     -- The units returned are microns/1e25dyne.cm (as in kikuchi's -
c     -- code). the excitation functions are originally in cm/1e27   -
c     -- dyne.cm (Hiroo's convention)                                -
c     --                                                             -
c     -- 2005/01/09: added more overtones                            _
c     --                                                             -
c-----------------------------------------------------------------------

      parameter(ntuq=12,mex=200,mxt=2048)
      real x(*),tmax(7),frq4(4)
      real tt(mex),sr1(*),pr1(*),qr1(*),ra(*),ft(mex)
      real u(*),c(*),q(*)
      real f(mxt),t(mxt)
      real*8 baz,az,dist

      integer ntt(mex)

      character*80 exf

      complex cc1,cc2,cc3,cc4,ccc(mxt),ccc2(mxt),ccc2r(mxt),cc6,butter

c     -- setup parameters for fft ------------------------------------
      lx1=2**np
      lcent=1+lx1/2
      data tmax /400., 400.,  400., 400, 400., 400., 310./

      pi=acos(-1.0)
      conv=pi/180.
c lor: Love or Rayleigh, 1 for Love, 0 for Rayleigh
      lor=mod(ityp,2)
c cc1 takes into avcount the pi/4 source phase shift - negative for Love?
      if(lor.eq.0) then
        cc1=cmplx(cos(pi/4.0),sin(pi/4.0))
      else
        cc1=cmplx(cos(pi/4.0),-sin(pi/4.0))
      endif

c     -- subroutine to compute surface waves based on the Harvard ----
c     -- phasevelocity maps ------------------------------------------
c     -- need to do a call readhrv in the main routine ---------------

c     -- common format statements ------------------------------------
25    format(a80)


c     -- read q and u ------------------------------------------------
c     -- interpolate excitation functions, q and u -------------------

      do i=1,lcent
	f(i)=(i-1)*df
	if(i.gt.1) then
	  t(i)=1/f(i)
        else
	  t(i)=9999.
        endif
c       write(*,'(f8.3,8(x,e10.3))') t(i),c(i),sr1(i),qr1(i),pr1(i)
      enddo

      sr1(1)=sr1(2)
      qr1(1)=qr1(2)
      pr1(1)=pr1(2)
      c(1)=c(2)
      q(1)=q(2)
      u(1)=u(2)
      q(1)=q(2)
c     -- done interpolating excitation functions ---------------------

c     -- source parameters -------------------------------------------

      call thetasr(1.d0*evla,1.d0*evlo,1.d0*stla,1.d0*stlo,baz,az,dist)
      c5=sqrt(abs(sin(dist*conv)))
      distkm=6371.*2*pi*dist/360.
      fai=str-az
      snl = sin(conv*rak)
      cnl = cos(conv*rak)
      snd = sin(conv*dip)
      cnd = cos(conv*dip)
      snf = sin(conv*fai)
      cnf = cos(conv*fai)
      cn2d= cos(2.0*conv*dip)
      sn2f= sin(2.0*conv*fai)
      cn2f= cos(2.0*conv*fai)

      ssr=snl*snd*cnd
      psr=cnl*snd*sn2f-snl*snd*cnd*cn2f
      qsr=snl*cn2d*snf+cnl*cnd*cnf
      qsl=-cnl*cnd*snf+snl*cn2d*cnf
      psl=snl*snd*cnd*sn2f+cnl*snd*cn2f
      
c     -- prepare for fft ---------------------------------------------
      do i=1,lcent
	c7=2.0*pi*f(i)*(distkm/c(i)-tst)
c cc2 is the phase propagation term
	cc2=cmplx(cos(c7),-sin(c7))
c cc3 is the source term
	if(lor.eq.0) then
	  cc3=cmplx((sr1(i)*ssr+pr1(i)*psr)/c5,qr1(i)*qsr/c5)
	else
	  cc3=-cmplx(pr1(i)*psl/c5,qr1(i)*qsl/c5)
        endif
c cc4 shifts seismogram to desired tsart time
        cc4=cmplx(cos(-ts0*pi*f(i)),sin(-ts0*pi*f(i)))
	ccc(i)=cc1*cc2*cc3*cc4
	c10=0.0
	if(u(i).gt.0.1) c10=exp(-pi*f(i)*distkm/(q(i)*u(i)))
c Here is where we add in the new treatment for attenuation!
c	c10=exp(-pi*f(i)*q(i))
c cc6 is the attenuation term
	cc6=cmplx(c10,0.0)
	ccc2(i)=cc6*ccc(i)
	if(lor.eq.0) ccc2r(i)=ccc2(i)*cmplx(0.0,-ra(i))
c	print *,ccc2(i),butter("b",3,f(i),f1,f2),f(i),f1,f2
        xtpr = 0.
        if (f(i).ge.frq4(2).and.f(i).le.frq4(3)) then
            xtpr = 1.
        else if (f(i).gt.frq4(1).and.f(i).le.frq4(2)) then
            xtpr = 0.5*(1.-cos(pi*(f(i)-frq4(1))/(frq4(2)-frq4(1))))
        else if (f(i).gt.frq4(3).and.f(i).le.frq4(4)) then
            xtpr = 0.5*(1.+cos(pi*(f(i)-frq4(3))/(frq4(4)-frq4(3))))
        endif
c        write(*,*) f(i),xtpr
	ccc2(i)=ccc2(i)*xtpr
c	ccc2(i)=ccc2(i)*butter("b",3,f(i),f1,f2)
      enddo
      call conj(ccc2,lcent)
      call cool(np,ccc2,1.0)

      do i=1,lx1
	x(i)=100.*real(ccc2(i))*df
      enddo

      return
      end
c#

      subroutine phvpath(evla,evlo,stla,stlo,per,nper,ityp,c,u,q)
c
c     -- computes path averaged phasevelocities for the Harvard model --
c
      real per(*),c(*),u(*),q(*)
      real ttime(4096),f(4096),qinv(4096)
      real*8 ela,elo,sla,slo,baz,az,dist,dsegment
      real evla,evlo,stla,stlo,gla,glo
      real*8 grla,grlo,graz
      ela=evla
      elo=evlo
      sla=stla
      slo=stlo
      if(nper .gt.4096) then
	print*,'phvpath: nper is too large'
	stop
      endif
      do j=1,nper
c    Don't try to extrapolate to periods longer than 250 s
c    or shorter than 25 s.
        if (per(j).gt.250.) then
           f(j) = 0.004
        elseif (per(j).lt.25.) then
           f(j) = 0.04
        else
           f(j) = 1./per(j)
        endif
	ttime(j)=0.0
      enddo
c
c     -- compute path averaged phasevelocity ---------------------------
c      call phv_hrvt(sla,slo,per,nper,ityp,c,u,q)
      call path_gdm52(evlo,evla,stlo,stla,ityp,f,c,u,nper)
c      write(*,*) evlo,evla,stlo,stla
c      do j=1,nper
c         write(*,*) f(j),c(j),u(j)
c      enddo
c      return
c     -- compute path integrated attenuation ---------------------------
c     -- dr is the target spacing for the segments ---------------------
      dr=1.0
      call thetasr(ela,elo,sla,slo,baz,az,dist)
      nsegments=dist/dr
      dsegment=dist/nsegments
c     print*,nsegments,dsegment
      aseg=0.0
      do i=0,nsegments
	call gettp(ela,elo,az,dsegment*i,grla,grlo,graz)
        gla = grla
        glo = grlo
c Calculate group velocity and 1/Q at (glo,gla)
        call point_gdm52(glo,gla,ityp,f,c,u,nper)
        call point_de06(glo,gla,f,qinv,nper)
	do j=1,nper
	  if(i.eq.0 .or.i.eq.nsegments) then
	    ttime(j)=ttime(j)+0.5*111.*dsegment*qinv(j)/u(j)
	    aseg=aseg+0.5*111.*dsegment
	  else
	    ttime(j)=ttime(j)+111.*dsegment*qinv(j)/u(j)
	    aseg=aseg+111.*dsegment
	  endif
        enddo
      enddo
c Save integrated attenuation in q() for use in surfw_simple
      do j=1,nper
	q(j)=ttime(j)
      enddo
      return
      end
c     ------------------------------------------------------------------


      subroutine phvpath0(evla,evlo,stla,stlo,per,nper,ityp,c,u,q)
c
c     -- computes path averaged phasevelocities for the Harvard model --
c
      real per(*),c(*),u(*),q(*)
      real ttime(4096)
      real*8 ela,elo,sla,slo,baz,az,dist,dsegment
      real*8 grla,grlo,graz
      ela=evla
      elo=evlo
      sla=stla
      slo=stlo
      if(nper .gt.4096) then
	print*,'phvpath: nper is too large'
	stop
      endif
      do j=1,nper
	ttime(j)=0.0
      enddo
c     -- compute path averaged phasevelocity ---------------------------
c
c     -- dr is the target spacing for the segments ---------------------
      dr=1.0
      call thetasr(ela,elo,sla,slo,baz,az,dist)
      nsegments=dist/dr
      dsegment=dist/nsegments
c     print*,nsegments,dsegment
      aseg=0.0
      do i=0,nsegments
	call gettp(ela,elo,az,dsegment*i,grla,grlo,graz)
	call phv_hrvt(grla,grlo,per,nper,ityp,c,u,q)
	do j=1,nper
	  if(i.eq.0 .or.i.eq.nsegments) then
	    ttime(j)=ttime(j)+0.5*111.*dsegment/c(j)
	    aseg=aseg+0.5*111.*dsegment
	  else
	    ttime(j)=ttime(j)+111.*dsegment/c(j)
	    aseg=aseg+111.*dsegment
	  endif
        enddo
      enddo

      do j=1,nper
	c(j)=111.*dist/ttime(j)
      enddo
      return
      end
c     ------------------------------------------------------------------

      subroutine thetasr(tts,pps,ttr,ppr,bazm,azm,dist)
c     ------------------------------------------------------------------
c     -- back azimuth, azimuth and distance between source and receiver:
c     -- source coordinate (ts,ps,), receiver coordinate (tr,pr); ------
c     -- back azimuth (azimuth of source from receiver); ---------------
c     -- azimuth (azimuth of receiver from source);  -------------------
c     -- They are all in degrees, measured clockwise from north. -------
      implicit real*8 (a-h,o-z)
      double precision x,xd,dph,dis,sang,cang,rdph,rps,rts,rtr
      if(tts.eq.ttr.and.pps.eq.ppr) then
	 dist=0.0
	 azm=0.0
	 bazm=0.0
	 return
      endif
        
      if(pps.lt.0.0) then
         ps=pps+360.
        else
         ps=pps
        endif
      if(ppr.lt.0.0) then
         pr=ppr+360.
      else
         pr=ppr
        endif
      ts=90.-tts
      tr=90.-ttr

      conv=3.141592653589793/180.0d0
      rps=ps*conv
      rts=ts*conv
      rpr=pr*conv
      rtr=tr*conv
      xd=pr-ps
      if(xd .gt. -360.0 .and. xd .le. -180.0) then
        icase=1
        dph=xd+360.0
      elseif(xd .gt. -180.0 .and. xd .le. 0.0) then
        icase=2
        dph= -xd
      elseif(xd .gt. 0.0 .and. xd .le. 180.0) then
        icase=3
        dph=xd
      elseif(xd .gt. 180.0 .and. xd .le. 360.0 ) then
        icase=4
        dph = 360.0 - xd
      endif

      rdph=dph*conv
      x=dcos(rts)*dcos(rtr) + dsin(rts)*dsin(rtr)*dcos(rdph)
      dis=dacos(x)
      dist=dis/conv

      if(dis.ne.0.0d0) then
         sang=dsin(rts)*dsin(rdph)/dsin(dis)
         cang=(dcos(rts)-x*dcos(rtr))/(dsin(dis)*dsin(rtr))
         bazm=datan2(sang,cang)/conv
         if(icase .eq. 2 .or. icase .eq. 4) bazm= -bazm

         sang=dsin(rtr)*dsin(rdph)/dsin(dis)
         cang=(dcos(rtr)-x*dcos(rts))/(dsin(rts)*dsin(dis))
         azm=datan2(sang,cang)/conv
         if(icase .eq. 1 .or. icase .eq. 3) azm = -azm
c       -- change of convention: clockwise from north
         azm = -azm
         bazm = -bazm
      else
	 azm=0.0
	 bazm=0.0
      endif
      return
      end
c
      subroutine gettp(tthe,pphi,azm,dis,athe,aphi,aazm)
c     -- for a given (the,phi) and azm and dis, calculates (athe,aphi) -
c     -- and azimuth azm (actaully baz) at the new point.  -------------
c     -- Note that if azm=0.0, this routine causes zero divide. --------
      								
      double precision  tthe,the,phi,pphi,azm,dis,athe,aphi,aazm
      double precision  rthe,rphi,rdis,ct,cd,st,sd,razm,cb,ctp,stp
      double precision  sb,sc,sa,cc,cang,ca,aang
      the=90.-tthe
      phi=pphi
      if(phi.lt.0.0) phi=phi+360.
      conv=3.141592653589793/180.0d0
      rthe=the*conv
      rphi=phi*conv
      rdis=dis*conv
      ct=dcos(rthe)
      cd=dcos(rdis)
      st=dsin(rthe)
      sd=dsin(rdis)
      razm=azm*conv
      cb=dcos(razm)
      
      ctp=ct*cd+st*sd*cb
      athe=dacos(ctp)/conv
      if(athe .lt. 0.0) athe=athe+180.
      stp=dsin(athe*conv)
      
      sb=dsin(razm)
      sc=st*sb/stp
      sa=sd*sb/stp
      
      if(the .eq.90.0) then
        cc = -cd*sc*cb/sb
        cang=datan2(sc,cc)/conv
        ca= -cb*cc+sb*sc*cd
        aang=datan2(sa,ca)/conv
        aphi = phi + aang
      else
        if(the .ne. 90.0) then
          ca=(ctp*st-sd*cb)/stp/ct
          aang=datan2(sa,ca)/conv
          cc=(ct*sd-st*cd*cb)/stp
          cang=datan2(sc,cc)/conv
          aphi=phi+aang
        endif
      endif
      if(aphi .lt. 0.0) aphi= aphi+360.0
      if(aphi .gt. 360.0) aphi=aphi-360.0
      aazm=360. - cang
      if(aazm .gt. 360.) aazm=aazm - 360.0
      if(aazm .lt. 0.0 ) aazm=aazm + 360.0
      athe=90.-athe
      return
      end
c
c#
c     -- read excitation functions from file ---------------------------
      subroutine rdexcite(exfnm,it)
      include 'sub.excite.inc'
      character*80 exfnm
      
      if(it.gt.Nmod) then
	 print*,'sub.excite.f: it is larger than Nmod - ',it
	 stop
      endif
10    format(a80)
      open(1,file=exfnm)
      rewind 1
c     -- odd itype means toroidal mode ---------------------------------
      itype=mod(it,2)
      do j=1,Mdep
        read(1,*,end=64) zxc(j,it),vxcp(j,it),vxcs(j,it),
     +			rhxc(j,it),nxct(j,it)
c	print*, zxc(j,it),vxcp(j,it),vxcs(j,it),rhxc(j,it),nxct(j,it)

	if(nxct(j,it).gt.Edim) then
	   print*,'sub.excite.f: nxct is larger than Edim'
	   stop
        endif

        do i=1,nxct(j,it)
          if(itype.eq.1) then
	    read(1,*) ntxc,txc(i,j,it),pxc(i,j,it),qxc(i,j,it)
          else
	    read(1,*) ntxc,txc(i,j,it),axc(i,j,it),sxc(i,j,it),
     +		       pxc(i,j,it),qxc(i,j,it),rxc(i,j,it)
          endif
	enddo
      enddo
      print*,'sub.rdexcite.f: Reached maximum number of depth points'
      stop
64    continue
      nxcz(it)=j-1
c      print *,'it,nxcz ',it,nxcz(it)
      close(1)
      end


c     -- interpolate excitation function for the right depth -----------
      subroutine getexcite(depth,T,s,p,q,r,n,it)
      real s(*),p(*),q(*),r(*),T(*)
      include 'sub.excite.inc'
      
      if(it.gt.Nmod) then
	 print*,'sub.exite.f - getexcite: it is larger than Nmod'
	 stop
      endif
      dz=9999.
      do i=1,nxcz(it)
	duz=abs(depth-zxc(i,it))
	if(duz.lt.dz) then
	  iz=i
	  dz=duz
        endif
      enddo
c      write(*,*) 'iz,dz=',iz,dz
      if(zxc(iz,it).lt.depth) then
	idir=1
      else
	idir=-1
      endif

      if(iz.eq.nxcz(it)) then
        idir=-1.
      elseif(iz.eq.1) then
	idir=1
      elseif(zxc(iz,it).gt.zxc(iz+1,it)) then
	idir=-1.*idir
      endif
c      print*, iz,it,idir
c      print*,'excite ',depth,zxc(iz,it),zxc(iz,it),nxcz(it),iz,idir

      do j=1,n
	ddt=(zxc(iz,it)-depth)/(zxc(iz,it)-zxc(iz+idir,it))
c        print *,'ddt ',ddt
	s1=ainterp(txc(1,iz,it),sxc(1,iz,it),T(j),nxct(iz,it))
	p1=ainterp(txc(1,iz,it),pxc(1,iz,it),T(j),nxct(iz,it))
	q1=ainterp(txc(1,iz,it),qxc(1,iz,it),T(j),nxct(iz,it))
	r1=ainterp(txc(1,iz,it),rxc(1,iz,it),T(j),nxct(iz,it))
	s2=ainterp(txc(1,iz+idir,it),sxc(1,iz+idir,it),
     +             T(j),nxct(iz+idir,it))
	p2=ainterp(txc(1,iz+idir,it),pxc(1,iz+idir,it),
     +             T(j),nxct(iz+idir,it))
	q2=ainterp(txc(1,iz+idir,it),qxc(1,iz+idir,it),
     +             T(j),nxct(iz+idir,it))
	r2=ainterp(txc(1,iz+idir,it),rxc(1,iz+idir,it),
     +             T(j),nxct(iz+idir,it))
	s(j)=s1+ddt*(s2-s1)
	p(j)=p1+ddt*(p2-p1)
	q(j)=q1+ddt*(q2-q1)
	r(j)=r1+ddt*(r2-r1)
      enddo
      end

c#
c     -- subroutines to read the harvard surface wave data sets --------

      subroutine readhrv(coefpath)

c     -- reads the Harvard model parameters ----------------------------
c     ------------------------------------------------------------------
      implicit none

      integer*4 im, M, il, L, Ldim, maxL, minL, taper, ip, np, idx
      integer*4 nprem
      parameter(Ldim=40)
      common /hrvcoef/ clm,slm,tlm,norm

      real*8 norm, junk, pi, lat, lon, filt
      real*8 clm(Ldim+1,Ldim+1,18), slm(Ldim+1,Ldim+1,18),tlm(18)

      character*256 coeffile(18),file, xyzfile, ajunk
      data tlm /35.,37.,40.,45.,50.,60.,75.,100.,150.,
     1          35.,37.,40.,45.,50.,60.,75.,100.,150./
      data coeffile /'L35_SH.txt','L37_SH.txt','L40_SH.txt',
     1               'L45_SH.txt','L50_SH.txt','L60_SH.txt',
     2               'L75_SH.txt','L100_SH.txt','L150_SH.txt',
     3               'R35_SH.txt','R37_SH.txt','R40_SH.txt',
     4               'R45_SH.txt','R50_SH.txt','R60_SH.txt',
     5               'R75_SH.txt','R100_SH.txt','R150_SH.txt'/

c      common /hrvcoef/ clm,slm,tlm,norm
c      coefpath='/d/atws/1/hongkie/Share/Harvard-phv'
      character*(*) coefpath
Cf2py intent(in) coefpath
      pi = 3.1415927d0

c     ------------------------------------------------------------------

      np=18
      minL=0
      maxL=40
      taper=0
      norm=1./pi

 10   format(a80)

      idx=index(coefpath,' ')-1
      do 20 ip=1,np

	file=coefpath(1:idx)//'/'//coeffile(ip)
        open(unit=10,file=file)
	rewind 10
        do L = 0, maxL
          if ((L .gt. taper).and.(taper .gt. 0)) then
            filt = exp(log(0.01)*(L-taper)**2 / (maxL-taper)**2)
          else 
            filt = 1.0
          end if
          il = L+1
          do M = 0, L
            im = M+1
	    if (m.eq.0) then
               read(10,*) junk, junk, clm(il,im,ip)
	    else
               read(10,*) junk, junk, clm(il,im,ip), slm(il,im,ip)
	    endif
            if (L .lt. minL) then
               clm(il,im,ip) = 0.0d0
               slm(il,im,ip) = 0.0d0
            else
               clm(il,im,ip) = clm(il,im,ip) * filt
               slm(il,im,ip) = slm(il,im,ip) * filt
            end if
          end do
        end do
	close(10)
20    continue
      print*, '\nSurfw: Read phase velocity corrections from ',coefpath
      return
      end
c     ------------------------------------------------------------------

c     ------------------------------------------------------------------

      subroutine phv_hrvt(lat,lon,T,nt,wtype,c,u,q)

      implicit none

      integer*4 im,M,il,L,Ldim,maxL,minL,ip1,ip2, wtype,it0,ilm,i,j,nt
      integer*4 ip,nprem(12)

      parameter(Ldim=40)

      real*8 norm, junk, pi, lat, lon, filt
      real*8 dlon, dlat, dphi, phi, theta, dtheta
      real*8 Plm, A1,A2,plgndr,A(9)
      real*8 clm(Ldim+1,Ldim+1,18), slm(Ldim+1,Ldim+1,18)
      real*8 cmphi, smphi,tlm(18),costh,conv,dinterp0
      real*4 T(*),c(*),u(*),q(*),cpr,ainterp
      real*4 tprem(50,12),cprem(50,12),uprem(50,12),qprem(50,12)
      real*4 fprem(100)
      common /hrvcoef/clm,slm,tlm,norm
      data nprem /36,36,35,34,31,34,27,33,25,32,23,31/
      data (tprem(i,1),i=1,36) /
     1   926.93,819.20,737.40,619.86,538.23,477.47,430.07,375.28,333.40,
     2   300.18,265.12,237.46,215.04,192.35,173.98,156.09,141.54,127.65,
     3   114.78,104.27, 94.54, 85.67, 77.66, 70.49, 63.65, 57.68, 52.16,
     4    47.39, 43.06, 39.01, 35.44, 32.21, 29.25, 26.57, 24.14, 21.92/
      data (cprem(i,1),i=1,36) /
     1     6.64,  6.52,  6.39,  6.15,  5.95,  5.78,  5.64,  5.47,  5.34,
     2     5.23,  5.12,  5.03,  4.96,  4.90,  4.84,  4.79,  4.75,  4.72,
     3     4.68,  4.65,  4.63,  4.60,  4.58,  4.56,  4.54,  4.52,  4.50,
     4     4.48,  4.46,  4.43,  4.40,  4.37,  4.32,  4.27,  4.21,  4.15/
      data (uprem(i,1),i=1,36) /
     1     5.83,  5.53,  5.30,  4.99,  4.79,  4.66,  4.57,  4.48,  4.43,
     2     4.41,  4.39,  4.38,  4.38,  4.38,  4.38,  4.38,  4.38,  4.38,
     3     4.38,  4.38,  4.38,  4.37,  4.36,  4.36,  4.34,  4.32,  4.30,
     4     4.26,  4.22,  4.16,  4.09,  3.99,  3.88,  3.76,  3.63,  3.51/
      data (qprem(i,1),i=1,36) /
     1   205.42,195.63,187.07,173.23,162.92,155.23,149.48,143.38,139.31,
     2   136.54,134.14,132.67,131.79,131.22,131.04,131.18,131.59,132.36,
     3   133.52,134.97,136.91,139.44,142.71,146.88,152.64,160.02,170.19,
     4   183.45,201.83,228.57,265.17,314.95,378.90,449.41,514.09,561.10/
      data (tprem(i,3),i=1,35) /
     1   808.99,694.86,630.72,571.27,519.32,438.55,381.68,339.77,307.16,
     2   269.47,240.79,218.26,194.87,176.74,159.04,142.59,129.59,117.36,
     3   106.06, 95.77, 86.50, 78.21, 70.82, 64.26, 58.07, 52.67, 47.70,
     4    43.19, 39.13, 35.50, 32.25, 29.26, 26.53, 24.06, 21.84/
      data (cprem(i,3),i=1,35) /
     1    32.99, 16.46, 14.10, 12.74, 11.86, 10.74,  9.99,  9.43,  8.99,
     2     8.49,  8.11,  7.80,  7.47,  7.19,  6.90,  6.61,  6.37,  6.15,
     3     5.94,  5.77,  5.61,  5.47,  5.36,  5.26,  5.16,  5.08,  5.01,
     4     4.94,  4.88,  4.83,  4.78,  4.74,  4.71,  4.68,  4.66/
      data (uprem(i,3),i=1,35) /
     1     2.58,  5.36,  6.28,  6.85,  7.10,  6.98,  6.61,  6.33,  6.16,
     2     5.98,  5.80,  5.62,  5.37,  5.15,  4.93,  4.74,  4.63,  4.56,
     3     4.51,  4.48,  4.46,  4.45,  4.43,  4.42,  4.41,  4.40,  4.38,
     4     4.37,  4.35,  4.35,  4.35,  4.37,  4.40,  4.43,  4.45/
      data (qprem(i,3),i=1,35) /
     1   259.95,252.84,249.65,246.27,242.16,232.14,223.24,217.33,213.16,
     2   207.31,200.41,192.37,181.00,170.41,159.67,150.75,145.23,141.58,
     3   139.55,138.70,138.52,138.54,138.34,137.59,135.88,133.10,128.98,
     4   123.57,117.22,110.58,104.37, 99.02, 94.96, 92.27, 90.67/
      data (tprem(i,5),i=1,31) /
     1   457.07,402.40,363.14,323.88,289.21,260.90,228.72,204.83,186.14,
     2   166.54,151.08,135.72,121.36,110.15, 99.92, 90.82, 81.94, 74.20,
     3    67.36, 60.86, 55.20, 49.98, 45.23, 40.93, 37.06, 33.59, 30.49,
     4    27.64, 25.07, 22.76, 20.67/
      data (cprem(i,5),i=1,31) /
     1    58.39, 18.09, 14.70, 13.01, 12.04, 11.37, 10.61, 10.02,  9.56,
     2     9.07,  8.69,  8.31,  7.95,  7.65,  7.35,  7.05,  6.74,  6.46,
     3     6.22,  6.01,  5.82,  5.66,  5.51,  5.39,  5.28,  5.19,  5.12,
     4     5.06,  5.00,  4.95,  4.90/
      data (uprem(i,5),i=1,31) /
     1     1.29,  4.61,  6.09,  7.15,  7.54,  7.41,  6.97,  6.65,  6.43,
     2     6.22,  6.07,  5.91,  5.70,  5.46,  5.16,  4.88,  4.68,  4.58,
     3     4.54,  4.50,  4.47,  4.43,  4.41,  4.42,  4.44,  4.47,  4.49,
     4     4.50,  4.49,  4.46,  4.43/
      data (qprem(i,5),i=1,31) /
     1   203.93,213.10,223.40,234.49,240.96,241.68,237.47,231.54,225.21,
     2   217.49,211.51,205.53,197.40,186.62,172.75,159.46,148.64,141.93,
     3   137.78,134.88,132.78,131.03,129.63,128.68,128.20,127.96,127.50,
     4   126.26,123.84,120.38,116.67/
      data (tprem(i,7),i=1,27) /
     1   312.16,277.22,250.21,222.50,197.85,178.31,158.79,143.74,128.78,
     2   116.84,105.40, 94.88, 85.39, 76.96, 69.66, 62.98, 57.01, 51.62,
     3    46.70, 42.42, 38.54, 34.92, 31.73, 28.82, 26.17, 23.78, 21.60/
      data (cprem(i,7),i=1,27) /
     1    85.49, 19.25, 15.24, 13.33, 12.26, 11.51, 10.73, 10.13,  9.56,
     2     9.14,  8.73,  8.35,  8.01,  7.71,  7.41,  7.10,  6.78,  6.49,
     3     6.23,  6.03,  5.85,  5.69,  5.55,  5.42,  5.30,  5.20,  5.13/
      data (uprem(i,7),i=1,27) /
     1      .92,  4.39,  5.96,  7.20,  7.55,  7.16,  6.69,  6.52,  6.40,
     2     6.28,  6.10,  5.92,  5.79,  5.59,  5.27,  4.89,  4.63,  4.54,
     3     4.53,  4.53,  4.51,  4.45,  4.40,  4.37,  4.38,  4.42,  4.49/
      data (qprem(i,7),i=1,27) /
     1   215.72,227.62,237.10,244.53,242.68,232.77,222.67,219.46,218.73,
     2   216.96,212.51,207.71,204.65,198.65,184.69,165.59,150.38,141.44,
     3   136.78,134.48,132.97,131.38,129.47,127.41,125.92,125.82,127.47/
      data (tprem(i,9),i=1,25) /
     1   232.31,209.26,185.41,165.24,147.17,132.95,119.26,106.63, 96.78,
     2    87.56, 79.03, 71.32, 64.53, 58.61, 53.14, 48.04, 43.61, 39.47,
     3    35.77, 32.46, 29.43, 26.69, 24.24, 22.00, 19.96/
      data (cprem(i,9),i=1,25) /
     1   114.88, 20.14, 14.89, 13.09, 12.09, 11.36, 10.66, 10.01,  9.51,
     2     9.05,  8.66,  8.31,  8.00,  7.72,  7.42,  7.09,  6.77,  6.48,
     3     6.23,  6.03,  5.85,  5.69,  5.55,  5.42,  5.31/
      data (uprem(i,9),i=1,25) /
     1      .67,  3.98,  5.86,  7.20,  7.44,  7.07,  6.76,  6.47,  6.27,
     2     6.18,  6.13,  6.00,  5.80,  5.57,  5.21,  4.81,  4.61,  4.55,
     3     4.55,  4.55,  4.52,  4.47,  4.43,  4.41,  4.41/
      data (qprem(i,9),i=1,25) /
     1   220.86,220.88,222.84,229.27,232.24,231.39,229.30,222.95,217.81,
     2   218.00,219.86,216.59,208.80,199.50,183.95,163.61,149.39,141.08,
     3   136.61,134.10,132.52,131.61,131.45,132.00,132.74/
      data (tprem(i,11),i=1,23) /
     1   186.79,169.56,151.29,134.51,122.05,109.82, 98.84, 88.74, 80.52,
     2    72.94, 66.20, 59.73, 54.08, 49.10, 44.48, 40.26, 36.59, 33.15,
     3    30.09, 27.28, 24.74, 22.47, 20.41/
      data (cprem(i,11),i=1,23) /
     1   142.87, 20.53, 15.12, 13.23, 12.38, 11.57, 10.80, 10.14,  9.65,
     2     9.22,  8.83,  8.43,  8.09,  7.80,  7.53,  7.23,  6.90,  6.58,
     3     6.32,  6.10,  5.92,  5.76,  5.61/
      data (uprem(i,11),i=1,23) /
     1      .51,  3.78,  5.74,  7.35,  7.61,  6.97,  6.60,  6.57,  6.54,
     2     6.35,  6.06,  5.87,  5.80,  5.73,  5.49,  4.97,  4.59,  4.52,
     3     4.55,  4.57,  4.55,  4.51,  4.44/
      data (qprem(i,11),i=1,23) /
     1   199.53,204.46,218.20,236.65,240.47,230.57,224.36,229.06,233.58,
     2   229.39,218.49,211.49,208.64,205.10,196.20,174.06,154.05,143.09,
     3   137.81,135.43,134.77,134.48,133.68/
      data (tprem(i,2),i=1,36) /
     1   963.19,811.83,707.46,633.60,536.94,473.27,426.19,374.07,335.83,
     2   297.67,268.43,239.58,216.42,193.84,175.39,157.75,143.23,129.53,
     3   116.87,105.36, 95.01, 85.77, 77.55, 70.27, 63.83, 57.81, 52.55,
     4    47.73, 43.37, 39.29, 35.69, 32.42, 29.39, 26.72, 24.26, 22.05/
      data (cprem(i,2),i=1,36) /
     1     6.39,  6.57,  6.66,  6.65,  6.48,  6.27,  6.06,  5.78,  5.54,
     2     5.27,  5.06,  4.84,  4.68,  4.54,  4.43,  4.34,  4.27,  4.20,
     3     4.15,  4.11,  4.07,  4.04,  4.02,  4.00,  3.98,  3.97,  3.96,
     4     3.95,  3.94,  3.93,  3.91,  3.90,  3.89,  3.87,  3.84,  3.81/
      data (uprem(i,2),i=1,36) /
     1     7.88,  7.55,  6.95,  6.24,  5.25,  4.81,  4.53,  4.19,  3.93,
     2     3.72,  3.61,  3.57,  3.57,  3.60,  3.62,  3.65,  3.68,  3.70,
     3     3.72,  3.74,  3.76,  3.78,  3.80,  3.82,  3.83,  3.84,  3.84,
     4     3.84,  3.83,  3.81,  3.79,  3.76,  3.71,  3.65,  3.57,  3.47/
      data (qprem(i,2),i=1,36) /
     1   347.36,342.06,337.39,332.76,322.05,307.19,288.74,259.06,232.42,
     2   205.36,186.63,170.73,159.60,149.71,142.07,135.03,129.54,124.81,
     3   121.15,118.79,117.90,118.58,120.91,124.99,130.91,139.42,150.39,
     4   164.94,183.87,209.18,241.14,282.08,334.54,396.59,468.32,542.30/
      data (tprem(i,4),i=1,34) /
     1   852.62,729.78,657.01,555.77,465.46,391.38,336.05,299.53,263.58,
     2   236.16,214.18,190.87,172.52,154.56,138.11,125.41,113.72,103.09,
     3    93.46, 84.76, 76.92, 69.89, 63.15, 57.22, 52.01, 47.18, 42.76,
     4    38.79, 35.23, 31.95, 28.96, 26.26, 23.84, 21.67/
      data (cprem(i,4),i=1,34) /
     1    10.43,  9.97,  9.37,  8.47,  8.19,  8.18,  8.22,  8.10,  7.79,
     2     7.53,  7.33,  7.11,  6.93,  6.73,  6.51,  6.32,  6.12,  5.93,
     3     5.75,  5.59,  5.45,  5.33,  5.22,  5.12,  5.05,  4.98,  4.91,
     4     4.86,  4.80,  4.76,  4.71,  4.67,  4.63,  4.60/
      data (uprem(i,4),i=1,34) /
     1     8.71,  6.95,  5.44,  6.14,  7.72,  8.40,  8.22,  6.36,  5.92,
     2     5.82,  5.75,  5.64,  5.49,  5.27,  5.00,  4.77,  4.59,  4.47,
     3     4.40,  4.37,  4.36,  4.35,  4.36,  4.37,  4.37,  4.37,  4.36,
     4     4.34,  4.32,  4.30,  4.29,  4.29,  4.30,  4.31/
      data (qprem(i,4),i=1,34) /
     1   271.09,291.88,345.67,379.45,378.34,365.26,293.43,165.99,155.62,
     2   155.80,156.79,157.16,155.54,150.90,143.33,135.82,128.78,123.25,
     3   119.47,117.14,115.89,115.30,115.01,114.70,114.08,112.87,110.91,
     4   108.19,104.93,101.39, 98.03, 95.21, 93.11, 91.78/
      data (tprem(i,6),i=1,34) /
     1   805.03,725.06,594.94,536.21,448.68,388.78,344.84,308.56,273.42,
     2   244.36,220.81,192.96,174.06,156.64,139.57,126.20,113.55,102.05,
     3    91.90, 83.12, 75.52, 68.31, 61.99, 56.03, 50.85, 46.11, 41.85,
     4    38.05, 34.49, 31.32, 28.38, 25.79, 23.44, 21.29/
      data (cprem(i,6),i=1,34) /
     1    14.21, 12.27, 10.35,  9.95,  9.39,  8.95,  8.60,  8.37,  8.37,
     2     8.40,  8.43,  8.47,  8.36,  8.11,  7.86,  7.64,  7.42,  7.20,
     3     6.97,  6.74,  6.50,  6.27,  6.06,  5.88,  5.73,  5.58,  5.45,
     4     5.33,  5.22,  5.12,  5.05,  4.98,  4.93,  4.88/
      data (uprem(i,6),i=1,34) /
     1     6.10,  5.19,  7.18,  7.42,  7.07,  6.67,  6.48,  7.63,  8.64,
     2     8.73,  8.73,  8.69,  6.45,  6.32,  6.15,  5.98,  5.78,  5.55,
     3     5.26,  4.96,  4.73,  4.61,  4.57,  4.56,  4.52,  4.45,  4.37,
     4     4.32,  4.33,  4.37,  4.41,  4.43,  4.42,  4.38/
      data (qprem(i,6),i=1,34) /
     1   415.44,380.16,237.93,211.61,188.24,176.18,174.29,257.84,390.01,
     2   406.98,411.52,408.68,188.13,179.05,171.04,165.61,161.39,157.52,
     3   152.41,146.12,140.64,136.93,135.11,134.08,133.08,131.47,129.09,
     4   126.50,124.37,123.07,122.14,120.99,119.02,116.04/
      data (tprem(i,8),i=1,33) /
     1   903.98,705.62,545.46,447.52,392.21,354.65,310.44,273.44,242.39,
     2   216.92,196.14,177.92,159.47,144.48,128.41,115.57,104.23, 93.85,
     3    84.46, 76.10, 68.71, 62.31, 56.48, 51.06, 46.33, 42.07, 38.16,
     4    34.60, 31.42, 28.51, 25.89, 23.48, 21.33/
      data (cprem(i,8),i=1,33) /
     1    17.71, 16.21, 16.31, 16.26, 15.70, 13.28, 11.21, 10.10,  9.44,
     2     9.00,  8.68,  8.49,  8.51,  8.53,  8.54,  8.55,  8.26,  7.97,
     3     7.71,  7.46,  7.24,  7.02,  6.78,  6.51,  6.24,  6.00,  5.81,
     4     5.66,  5.53,  5.41,  5.31,  5.21,  5.12/
      data (uprem(i,8),i=1,33) /
     1     6.95, 16.88, 16.26, 15.79,  5.54,  5.28,  5.54,  6.06,  6.38,
     2     6.50,  6.51,  8.59,  8.68,  8.67,  8.66,  8.63,  6.13,  5.99,
     3     5.85,  5.71,  5.56,  5.29,  4.89,  4.53,  4.37,  4.38,  4.45,
     4     4.50,  4.49,  4.46,  4.41,  4.39,  4.38/
      data (qprem(i,8),i=1,33) /
     1   366.63, 90.91, 89.99, 89.92,275.53,263.60,249.88,236.20,222.38,
     2   210.03,200.05,401.73,421.66,423.24,424.67,423.73,214.05,199.72,
     3   189.49,183.77,179.41,171.59,158.83,145.69,137.14,132.54,130.66,
     4   130.33,130.49,130.57,130.21,129.19,127.78/
      data (tprem(i,10),i=1,32) /
     1   707.90,580.62,488.05,438.67,380.67,331.83,294.39,258.76,232.82,
     2   211.16,186.25,165.67,149.04,132.64,119.81,107.51, 96.32, 87.29,
     3    78.55, 70.85, 64.15, 57.86, 52.48, 47.55, 43.19, 39.18, 35.50,
     4    32.26, 29.28, 26.59, 24.11, 21.87/
      data (cprem(i,10),i=1,32) /
     1    37.70, 27.58, 23.43, 20.28, 16.18, 16.08, 16.00, 14.73, 12.74,
     2    11.49, 10.48,  9.86,  9.42,  9.01,  8.68,  8.56,  8.57,  8.57,
     3     8.29,  8.01,  7.75,  7.48,  7.23,  6.99,  6.74,  6.49,  6.25,
     4     6.04,  5.85,  5.69,  5.54,  5.41/
      data (uprem(i,10),i=1,32) /
     1    11.63, 12.85, 13.21,  4.95, 15.50, 15.34, 15.22,  5.97,  5.71,
     2     6.07,  6.55,  6.73,  6.71,  6.55,  6.36,  8.64,  8.63,  6.44,
     3     6.28,  6.03,  5.77,  5.53,  5.36,  5.14,  4.85,  4.64,  4.56,
     4     4.51,  4.47,  4.44,  4.38,  4.33/
      data (qprem(i,10),i=1,32) /
     1   355.05,434.15,480.14,290.18, 89.80, 89.68, 89.60,302.40,268.22,
     2   260.27,258.94,258.82,256.42,248.45,235.76,426.37,427.17,242.99,
     3   239.53,226.95,210.27,194.36,183.88,172.65,158.43,146.77,140.44,
     4   137.94,136.89,135.17,131.67,127.88/
      data (tprem(i,12),i=1,31) /
     1   583.50,478.17,420.25,369.91,332.15,283.64,240.56,212.95,187.61,
     2   166.78,150.58,134.10,121.09,108.61, 98.37, 88.53, 79.75, 72.29,
     3    65.32, 59.19, 53.77, 48.77, 44.27, 40.23, 36.51, 33.07, 30.00,
     4    27.21, 24.73, 22.46, 20.42/
      data (cprem(i,12),i=1,31) /
     1    45.74, 33.49, 21.17, 19.68, 18.54, 16.60, 15.85, 15.04, 13.77,
     2    12.31, 11.31, 10.47,  9.87,  9.33,  8.94,  8.61,  8.58,  8.58,
     3     8.34,  8.10,  7.88,  7.64,  7.38,  7.13,  6.87,  6.60,  6.34,
     4     6.12,  5.92,  5.74,  5.59/
      data (uprem(i,12),i=1,31) /
     1    28.52,  2.73, 13.15, 12.69, 11.84,  8.26, 15.08,  9.39,  7.36,
     2     6.40,  6.52,  6.49,  6.34,  6.34,  6.43,  6.45,  8.62,  8.61,
     3     6.35,  6.30,  6.04,  5.69,  5.44,  5.23,  4.89,  4.62,  4.58,
     4     4.51,  4.41,  4.43,  4.47/
      data (qprem(i,12),i=1,31) /
     1    90.75,317.86,489.18,502.49,506.32,418.20, 89.48,385.70,345.56,
     2   284.69,268.85,253.87,240.42,234.78,238.17,242.73,428.08,428.30,
     3   248.95,251.80,242.51,222.23,202.12,187.62,171.50,156.04,146.18,
     4   138.72,132.87,130.79,131.43/

      pi = dacos(-1.0d0)
      conv=pi/180.d0
      maxL=40

      do j=1,9
	A(j)=0.0d0
      enddo
      do j=1,nprem(wtype)
	fprem(j)=1./tprem(j,wtype)
      enddo


      phi = lon * conv
      theta=(90.-lat)*conv
      costh=dcos(theta)

c     -- set love or rayleigh ------------------------------------------
      if (wtype.eq.1) then
	it0=0
      elseif (wtype.eq.2) then
	it0=9
      endif

c     ------------------------------------------------------------------
      if(wtype .lt. 3) then
        do L = 0, maxL
          do M = 0, L
            il = L+1
            im = M+1
	    Plm=plgndr(l,m,costh)
            cmphi = dcos(dble(M)*phi)
            smphi = dsin(dble(M)*phi)
	    do ip=1,9
              A(ip)=A(ip)+
     1	          Plm*(clm(il,im,ip+it0)*cmphi+slm(il,im,ip+it0)*smphi)
            end do
          end do
        end do
      endif

c     -- determine the period brackets ---------------------------------
      do j=1,nt
	cpr=ainterp(fprem,cprem(1,wtype),1./T(j),nprem(wtype))
	u(j)=ainterp(fprem,uprem(1,wtype),1./T(j),nprem(wtype))
	q(j)=ainterp(fprem,qprem(1,wtype),1./T(j),nprem(wtype))
        if(wtype .lt. 3) then
          c(j)=dinterp0(tlm,A,1.d0*T(j),9)
          c(j)=cpr*(1.+c(j)/100.)
        else
          c(j)=cpr
	endif
      enddo

      return
      end
c     ------------------------------------------------------------------
c#

      real*8 function plgndr(l,m,x)
      implicit real*8 (a-h,o-z)
      if(m.lt.0.or.m.gt.l.or.dabs(x).gt.1.)pause 'bad arguments'
      pmm=1.0d0
      if(m.gt.0) then
        somx2=dsqrt((1.-x)*(1.+x))
        fact=1.
        do 11 i=1,m
          pmm=-pmm*fact*somx2
          fact=fact+2.
11      continue
      endif
      if(l.eq.m) then
        plgndr=pmm
      else
        pmmp1=x*(2*m+1)*pmm
        if(l.eq.m+1) then
          plgndr=pmmp1
        else
          do 12 ll=m+2,l
            pll=(x*(2*ll-1)*pmmp1-(ll+m-1)*pmm)/(ll-m)
            pmm=pmmp1
            pmmp1=pll
12        continue
          plgndr=pll
        endif
      endif
      do 15 i=l-m+1,l+m,1
        plgndr=plgndr/sqrt(i*1.)
15    continue
      anorm=dsqrt(((2*l+1)/(4*3.1415927d0)))
      plgndr=plgndr*anorm
      return
      end

C#

      real function ainterp(x,y,x0,nx)
 
c     linear interpolation/extrapolation. x should be monotonous
c     otherwise it should work fine.
c 
c                             Hong Kie Thio, January, 1996
      real x(*),y(*)
      j=0
      isign=1
      if(x(1).gt.x(nx)) isign=-1
      do 10, i=1,nx
	if(isign*(x0-x(i)).gt..0)j=j+1
10    continue
      if(j.ge.nx) then
  	 ainterp=y(nx)+(y(nx)-y(nx-1))*(x0-x(nx))/(x(nx)-x(nx-1))
      else
	 if(j.eq.0) j=1
         ainterp=y(j)+(y(j+1)-y(j))*(x0-x(j))/(x(j+1)-x(j))
      endif
      return
      end

      real*8 function dinterp0(x,y,x0,nx)
 
c     linear interpolation only. x should be monotonous
c     otherwise it should work fine.
c 
c                             Hong Kie Thio, January, 1996
      real*8 x(*),y(*),x0
      j=0
      isign=1
      if(x(1).gt.x(nx)) isign=-1
      do 10, i=1,nx
	if(isign*(x0-x(i)).gt..0)j=j+1
10    continue
      if(j.ge.nx) then
  	 dinterp0=y(nx)
      else if(j.eq.0) then
         dinterp0=y(1)
      else
         dinterp0=y(j)+(y(j+1)-y(j))*(x0-x(j))/(x(j+1)-x(j))
      endif
      return
      end
c#

      subroutine wsac1(filename,x,nx,b,dt,nerr)
      real x(*),b,dt,fhd(70)
      integer nx,nerr,ihd(40)
      character*80 filename
      character*8 chd(24)
      data fhd /70*-12345./
      data ihd /40*-12345/
      data chd /24*'-12345'/

c     -- intitalize necessary header variables ------------------------------------------
      ihd(7)=6
      ihd(16)=1
      ihd(36)=1
      ihd(37)=0
      ihd(38)=1
      ihd(39)=1
      ihd(40)=0
c     ------------------------------------------------------------------------------------

      ihd(10)=nx
      fhd(6)=b
      fhd(7)=b+(nx-1)*dt
      fhd(1)=dt

c     following works on Solaris f77
c     open(1,file=filename,form='BINARY',status='REPLACE')
c     following works on OS X xlf
      open(1,file=filename,access='STREAM',form='UNFORMATTED',
c      open(1,file=filename,form='UNFORMATTED',
     +       status='REPLACE')
      write(1) fhd
      write(1) ihd
      write(1) chd
      write(1) (x(i),i=1,nx)
      close(1)
      end


c#
c      program testpath
c      parameter(nt=14)
cc
c      parameter (maxukn=8000,mxspl=20)
cc
c      dimension numvertex(2)
c      dimension numker(2)
c      dimension verlat(maxukn,2)
c      dimension verlon(maxukn,2)
c      dimension verrad(maxukn,2)
c      dimension splmod(maxukn,5,mxspl,2)
c      dimension fker(mxspl,2)
c      common/gdm52/numvertex,numker,fker,verlat,verlon,verrad,splmod
c      real ela,elo,sla,slo
c      real c(nt),t(nt),f(nt),u(14)
c      data t/30,40,50,60,70,80,90,100,125,150,175,200,225,250/
c      call read_gdm52()
c      ela=20.
c      elo=18.
c      sla=34.
c      slo=-118.
c      lora=2
c      do i=1,nt
c         f(i) = 1./T(i)
c      enddo
c      call path_gdm52(elo,ela,slo,sla,lora,f,c,u,nt)
c      do i=1,nt
c        write(*,*) T(i),c(i),u(i)
c      enddo
c      end
c         1         0         1      1442        12     17304
c         2         1         2      1442        12     51912
c      program test
c      parameter(nt=14)
cc
c      parameter (maxukn=8000,mxspl=20)
cc
c      dimension numvertex(2)
c      dimension numker(2)
c      dimension verlat(maxukn,2)
c      dimension verlon(maxukn,2)
c      dimension verrad(maxukn,2)
c      dimension splmod(maxukn,5,mxspl,2)
c      dimension fker(mxspl,2)
c      common/gdm52/numvertex,numker,fker,verlat,verlon,verrad,splmod
c      real lat,lon,dv
c      real T(14), c(20), u(20), q(20),f(nt)
c      data T /30,40,50,60,70,80,90,100,125,150,175,200,225,250/
c
cc
c      lat=30.
c      lon=95.
c      lora=2
cc
c      call read_gdm52()
c      do i=1,nt
c         f(i) = 1./T(i)
c      enddo
c      call point_gdm52(lon,lat,lora,f,c,u,nt)
c      do i=1,nt
c        write(*,*) T(i),c(i),u(i)
c      enddo
c      end


c***********************************************************************
      subroutine read_gdm52(dispdir)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c---- open and read model file
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      dimension lora(2)
      dimension ncoeff(2)
      character*(*) dispdir
c
      parameter (maxukn=8000,mxspl=20)
c
      dimension numvertex(2)
      dimension numker(2)
      dimension verlat(maxukn,2)
      dimension verlon(maxukn,2)
      dimension verrad(maxukn,2)
      dimension splmod(maxukn,5,mxspl,2)
      dimension fker(mxspl,2)
      common/gdm52/numvertex,numker,fker,verlat,verlon,verrad,splmod
      integer iprtlv
      common /plevel/iprtlv
c
c
      character*80 basisfile(2)
      dimension lbasisfile(2)
      character*128 string
      integer iostat,imdl,ios,iani(2)
      logical exists
      dimension dsparr(100000,2)
      character*80 dispfile(2)

      data iprtlv/0/

c      dispdir = '/Users/phil/repos/ffipy/philscode/surfw/GDM52/'
      dispfile(1)='LOVE_400_100.disp'
      dispfile(2)='RAYL_320_80_32000_8000.disp'

      inquire(file=dispdir//dispfile(1),exist=exists)
      if(.not.exists) then
	   write(*,*) 'Love dispersion model file '//dispdir//
     &dispfile(1)(1:17)//' does not exist'
	   stop
      endif
c
      inquire(file=dispdir//dispfile(2),exist=exists)
      if(.not.exists) then
	   write(*,*) 'Rayleigh dispersion model file '//dispdir//
     &dispfile(2)(1:27)//' does not exist'
	   stop 
      endif
c
c---- read in the dispersion model
c
      do imdl=1,2
          ios=0
          open(1,file=dispdir//dispfile(imdl),iostat=ios)
          do while(ios.eq.0) 
	    read(1,"(a)",iostat=ios) string
	    if(ios.eq.0) then
	      lstr=lnblnk(string)
	      if(string(1:9).eq.'#ANITYPE:') then
	        read(string(10:lstr),*) iani(imdl)
              endif
	      if(string(1:10).eq.'#WAVETYPE:') then
	        read(string(11:lstr),*) lora(imdl)
              endif
	      if(string(1:7).eq.'#BASIS:') then
	        read(string(8:lstr),*) basisfile(imdl)
              endif
	      lbasisfile(imdl)=lnblnk(basisfile(imdl))
	      if(string(1:11).eq.'#NUMVERTEX:') then
	        read(string(12:lstr),*) numvertex(imdl)
	        do i=1,numvertex(imdl)
	          read(1,*) verlat(i,imdl),verlon(i,imdl),verrad(i,imdl)
	        enddo
              endif
	      if(string(1:8).eq.'#NUMKER:') then
	        read(string(9:lstr),*) numker(imdl)
	        do i=1,numker(imdl)
	          read(1,*) idummy,fker(i,imdl)
	        enddo
              endif
	      if(string(1:8).eq.'#NCOEFF:') then
	        read(string(9:lstr),*) ncoeff(imdl)
	        do i=1,ncoeff(imdl)
	          read(1,*) idummy,dsparr(i,imdl)
	        enddo
              endif
            endif
          enddo
          close(1)
c         1         0         1      1442        12     17304
c         2         1         2      1442        12     51912
c          write(6,"(6i10)") imdl,iani(imdl),lora(imdl),numvertex(imdl),
c     #                      numker(imdl),ncoeff(imdl)
c
c---- put the model in new array
c
          ic=0
          do ik=1,numker(imdl)
	    if(iani(imdl).eq.0) then
	      nv=1
            else if(iani(imdl).eq.1) then
	      nv=3
            else
	      stop ' not ready for this iani'
            endif
	    do iv=1,nv   
	      do is=1,numvertex(imdl)
	        ic=ic+1
	        splmod(is,iv,ik,imdl)=dsparr(ic,imdl)
	      enddo
            enddo
          enddo
      enddo
      write(*,*) 'Read dispersion model GDM52 from files:'
      write(*,*) '         ',dispdir//dispfile(1)
      write(*,*) '         ',dispdir//dispfile(2)
      return
      end
c***********************************************************************
      subroutine point_gdm52(xlon,xlat,lora,f,vel1,velg1,nf)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      parameter (maxukn=8000,mxspl=20,maxcon=300)
      parameter (twopi=6.2831853)
      parameter (geoco=0.993277)
c
      dimension f(nf),vel1(nf),velg1(nf)
      dimension numvertex(2)
      dimension numker(2)
      dimension verlat(maxukn,2)
      dimension verlon(maxukn,2)
      dimension verrad(maxukn,2)
      dimension splmod(maxukn,5,mxspl,2)
      dimension fker(mxspl,2)
      common/gdm52/numvertex,numker,fker,verlat,verlon,verrad,splmod
c
      dimension icon(maxcon)
      dimension con(maxcon)
      dimension conhat(5,maxcon)
      dimension contil(5,maxcon)
      dimension iconhat(maxcon)
      dimension icontil(maxcon)
      dimension splv(mxspl)
      dimension splvd(mxspl)
      dimension splpt(mxspl)
      integer lora,iker,is,isp
      real zlat,xlat,xlon
c
c
c---- need to convert latitude to geocentric coordinates for splcon
c
      zlat=atand(geoco*tand(xlat))
      call splcon(zlat,xlon,
     #        numvertex(lora),verlat(1,lora),verlon(1,lora),
     #        verrad(1,lora),
     #        ncon,icon,con)
c
c---- calculate frequency spline for this location
c
      do iker=1,numker(lora)
	 splpt(iker)=0.
	 do isp=1,ncon
            is=icon(isp)
	    splpt(iker)=splpt(iker)+splmod(is,1,iker,lora)*con(isp)
         enddo
      enddo
c
c---- loop on frequencies
c
      frq0 = 0.
      do ifreq=1,nf
        if (f(ifreq).ne.frq0) then
	  omega=f(ifreq)*2.*3.14159
	  call vbspl(1000.*f(ifreq),numker(lora),fker(1,lora),splv,splvd)
          vel=premgephvelo(lora,omega)
          velg=premgegrvelo(lora,omega)
	  slow=1./vel
	  slowg=1./velg
	  dslow=0.
	  dslowg=0.
	  do iker=1,numker(lora)
	    dslow=dslow+splv(iker)*splpt(iker)
	    dslowg=dslowg+splvd(iker)*splpt(iker)
	  enddo
	  dslowg=dslow+omega*1000.*dslowg/twopi
        endif
c
        vel0  = vel
	velg0 = velg
	vel1(ifreq)  = 1./(slow+dslow)
	velg1(ifreq) = 1./(slowg+dslowg)
	vel2=-100.*(dslow/slow)/(1.+dslow/slow)
	velg2=-100.*(dslowg/slowg)/(1.+dslowg/slowg)
        frq0 = f(ifreq)
c
c	 write(*,"(7f8.3)") f(ifreq),vel0,vel1(ifreq),vel2,velg0,velg1(ifreq),velg2
      enddo

      return
      end
c***********************************************************************
      subroutine path_gdm52(xlon,xlat,xlon2,xlat2,lora,f,vel1,velg1,nf)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      parameter (maxukn=8000,mxspl=20,maxcon=300)
      parameter (twopi=6.2831853)
c
      dimension f(nf),vel1(nf),velg1(nf)
      dimension numvertex(2)
      dimension numker(2)
      dimension verlat(maxukn,2)
      dimension verlon(maxukn,2)
      dimension verrad(maxukn,2)
      dimension splmod(maxukn,5,mxspl,2)
      dimension fker(mxspl,2)
      common/gdm52/numvertex,numker,fker,verlat,verlon,verrad,splmod
c
      dimension icon(maxcon)
      dimension con(maxcon)
      dimension conhat(5,maxcon)
      dimension contil(5,maxcon)
      dimension iconhat(maxcon)
      dimension icontil(maxcon)
      dimension splv(mxspl)
      dimension splvd(mxspl)
      dimension splpt(mxspl)
      integer lora,inclazi,ifreq,iker,is,isp
c
      inclazi=1
c
c---- conversion to geocentric coordinates occurs in splaziave
c
      call splaziave(xlat,xlon,xlat2,xlon2,numvertex(lora),
     #             verlat(1,lora),verlon(1,lora),verrad(1,lora),
     #             0,ncontil,icontil,contil,nconhat,iconhat,conhat)
      do iker=1,numker(lora)
	 splpt(iker)=0.
	 do isp=1,ncontil
            is=icontil(isp)
	    splpt(iker)=splpt(iker)+
     #                        splmod(is,1,iker,lora)*contil(1,isp)
	    if(inclazi.eq.1) then
	       splpt(iker)=splpt(iker)+
     #                        splmod(is,2,iker,lora)*contil(2,isp)
	       splpt(iker)=splpt(iker)+
     #                        splmod(is,3,iker,lora)*contil(3,isp)
	    endif
         enddo
      enddo
c
c---- loop on frequencies
c
      frq0 = 0.
      do ifreq=1,nf
        if (f(ifreq).ne.frq0) then
	  omega=f(ifreq)*2.*3.14159
	  call vbspl(1000.*f(ifreq),numker(lora),fker(1,lora),splv,splvd)
          vel=premgephvelo(lora,omega)
          velg=premgegrvelo(lora,omega)
	  slow=1./vel
	  slowg=1./velg
	  dslow=0.
	  dslowg=0.
	  do iker=1,numker(lora)
	    dslow=dslow+splv(iker)*splpt(iker)
	    dslowg=dslowg+splvd(iker)*splpt(iker)
	  enddo
	  dslowg=dslow+omega*1000.*dslowg/twopi
        endif
c
        vel0  = vel
	velg0 = velg
	vel1(ifreq)  = 1./(slow+dslow)
	velg1(ifreq) = 1./(slowg+dslowg)
	vel2=-100.*(dslow/slow)/(1.+dslow/slow)
	velg2=-100.*(dslowg/slowg)/(1.+dslowg/slowg)
        frq0 = f(ifreq)
c
c	      write(4,"(7f8.3)") f,vel0,vel1,vel2,velg0,velg1,velg2
      enddo
      return
      end
c
c---- subroutines follow below
c
      subroutine splcon(xlat,xlon,nver,verlat,verlon,verrad,
     #                  ncon,icon,con)
      dimension verlat(1)
      dimension verlon(1)
      dimension verrad(1)
      dimension icon(1)
      dimension con(1)
c
c---- all calculations done in double precision
c---- but results are single
c
      real*8 dd
      real*8 rn
      real*8 dr
      real*8 dsind
      real*8 dcosd
      real*8 dacosd
c
      ncon=0
      do iver=1,nver
        if(xlat.gt.verlat(iver)-2.0*verrad(iver)) then
          if(xlat.lt.verlat(iver)+2.0*verrad(iver)) then
            dd=dsind(dble(verlat(iver)))*dsind(dble(xlat))
            dd=dd+dcosd(dble(verlat(iver)))*dcosd(dble(xlat))*
     #         dcosd(dble(xlon-verlon(iver)))
	    if(dd.gt.1.d0) dd=1.d0
            dd=dacosd(dd)
            if(dd.gt.dble(verrad(iver))*2.d0) then
            else
              ncon=ncon+1
              icon(ncon)=iver
              rn=dd/dble(verrad(iver))
              dr=rn-1.d0
              if(rn.le.1.d0) then
                con(ncon)=(0.75d0*rn-1.5d0)*(rn**2)+1.d0
              else if(rn.gt.1.d0) then
                con(ncon)=((-0.25d0*dr+0.75d0)*dr-0.75d0)*dr+0.25d0
              else
                con(ncon)=0.
              endif
            endif
          endif
        endif
      enddo
      return
      end

      subroutine splaziave(eplat,eplon,stlat,stlon,nver,verlat,verlon,
     &                     verrad,
     #           ifgc,ncontil,icontil,contil,nconhat,iconhat,conhat)
c
      parameter (geoco=0.993277)
c
      dimension verlat(1)
      dimension verlon(1)
      dimension verrad(1)
      dimension icontil(1)
      dimension iconhat(1)
      dimension contil(5,1)
      dimension conhat(5,1)
c
      parameter (maxdim=10000)
      dimension avec(5,maxdim)
      dimension iavec(maxdim)
      parameter (maxcon=100)
      dimension iconpt(maxcon)
      dimension conpt(maxcon)
c
      elat=atand(geoco*tand(eplat))
      elon=eplon
      slat=atand(geoco*tand(stlat))
      slon=stlon
      call delazgc(elat,elon,slat,slon,delta,azatep,az2)
c
c---- do the minor arc
c
      nseg=int(2.*delta)+1
      dseg=delta/float(nseg)
      segweight=dseg/delta
      do iver=1,nver
        avec(1,iver)=0.
        avec(2,iver)=0.
        avec(3,iver)=0.
        avec(4,iver)=0.
        avec(5,iver)=0.
        iavec(iver)=0
      enddo
      do iseg=1,nseg
        delta2=dseg*0.5+float(iseg-1)*dseg
        call pdaz(elat,elon,azatep,delta2,seglat,seglon)
        call delazgc(seglat,seglon,slat,slon,dd,azatseg,az2)
        call splcon(seglat,seglon,nver,verlat,verlon,verrad,
     &               nconpt,iconpt,conpt)
        if(nconpt.gt.maxcon) then
          stop 'too many contributing splines in splcon'
        endif
            do ic=1,nconpt
              iver=iconpt(ic)
              avec(1,iver)=avec(1,iver)+conpt(ic)*segweight
              iavec(iver)=iavec(iver)+1
              call delazgc(seglat,seglon,verlat(iver),verlon(iver),
     #                     ddd,aztobs,aztoseg)
              gamma=aztoseg-180.+azatseg-aztobs
              cos2fac=cosd(2.*gamma)
              sin2fac=sind(2.*gamma)
              cos4fac=cosd(4.*gamma)
              sin4fac=sind(4.*gamma)
              avec(2,iver)=avec(2,iver)+conpt(ic)*segweight*cos2fac
              avec(3,iver)=avec(3,iver)+conpt(ic)*segweight*sin2fac
              avec(4,iver)=avec(4,iver)+conpt(ic)*segweight*cos4fac
              avec(5,iver)=avec(5,iver)+conpt(ic)*segweight*sin4fac
            enddo
          enddo
          ncontil=0
          do iver=1,nver
            if(iavec(iver).gt.0) then
              ncontil=ncontil+1
              icontil(ncontil)=iver
              contil(1,ncontil)=avec(1,iver)
              contil(2,ncontil)=avec(2,iver)
              contil(3,ncontil)=avec(3,iver)
              contil(4,ncontil)=avec(4,iver)
              contil(5,ncontil)=avec(5,iver)
            endif
          enddo
c
c---- the great circle
c
          if(ifgc.ne.0) then
            nseg=int(360.)+1
            dseg=360./float(nseg)
            segweight=dseg/360.
            do iver=1,nver
              avec(1,iver)=0.
              avec(2,iver)=0.
              avec(3,iver)=0.
              avec(4,iver)=0.
              avec(5,iver)=0.
              iavec(iver)=0
            enddo
            do iseg=1,nseg
              delta2=dseg*0.5+float(iseg-1)*dseg
              call pdaz(elat,elon,azatep,delta2,seglat,seglon)
              call delazgc(seglat,seglon,slat,slon,dd,azatseg,az2)
              call splcon(seglat,seglon,nver,verlat,verlon,verrad,
     &                    nconpt,iconpt,conpt)
              if(nconpt.gt.maxcon) then
                stop 'too many contributing splines in splcon'
              endif
              do ic=1,nconpt
                iver=iconpt(ic)
                avec(1,iver)=avec(1,iver)+conpt(ic)*segweight
                iavec(iver)=iavec(iver)+1
                call delazgc(seglat,seglon,verlat(iver),verlon(iver),
     #                     ddd,aztobs,aztoseg)
                gamma=aztoseg-180.+azatseg-aztobs
                cos2fac=cosd(2.*gamma)
                sin2fac=sind(2.*gamma)
                cos4fac=cosd(4.*gamma)
                sin4fac=sind(4.*gamma)
                avec(2,iver)=avec(2,iver)+conpt(ic)*segweight*cos2fac
                avec(3,iver)=avec(3,iver)+conpt(ic)*segweight*sin2fac
                avec(4,iver)=avec(4,iver)+conpt(ic)*segweight*cos4fac
                avec(5,iver)=avec(5,iver)+conpt(ic)*segweight*sin4fac
              enddo
            enddo
            nconhat=0
            do iver=1,nver
              if(iavec(iver).gt.0) then
                nconhat=nconhat+1
                iconhat(nconhat)=iver
                conhat(1,nconhat)=avec(1,iver)
                conhat(2,nconhat)=avec(2,iver)
                conhat(3,nconhat)=avec(3,iver)
                conhat(4,nconhat)=avec(4,iver)
                conhat(5,nconhat)=avec(5,iver)
              endif
            enddo
          else
            nconhat=0.
          endif
      return
      end

      subroutine delazgc(eplat,eplong,stlat,stlong,delta,azep,azst)

      real*8 deplat,deplong,dstlat,dstlong,ddelta,dazep,dazst
c
      deplat=dble(eplat)
      deplong=dble(eplong)
      dstlat=dble(stlat)
      dstlong=dble(stlong)
c
      call ddelazgc(deplat,deplong,dstlat,dstlong,ddelta,dazep,dazst)
c
      delta=sngl(ddelta)
      azep=sngl(dazep)
      azst=sngl(dazst)
      return
      end

      subroutine ddelazgc(eplat,eplong,stlat,stlong,delta,azep,azst)
c
c---- modeified from delaz to use geocentric coordinates
c
      implicit double precision (a-h,o-z)
      data hpi,twopi,rad,reprad/1.57079632675d0,
     16.283185307d0,.0174532925d0,57.2957795d0/
      dtan(x)=dsin(x)/dcos(x)
      darcos(x)=datan2(dsqrt(1.d0-x*x),x)
      el=eplat*rad
c
      el=hpi-el
c
      stl=stlat*rad
c
      stl=hpi-stl
      elon=eplong*rad
      slon=stlong*rad
      as=dcos(stl)
      bs=dsin(stl)
      cs=dcos(slon)
      ds=dsin(slon)
      a=dcos(el)
      b=dsin(el)
      c=dcos(elon)
      d=dsin(elon)
      cdel=a*as+b*bs*(c*cs+d*ds)
      if(dabs(cdel).gt.1.d0) cdel=dsign(1.d0,cdel)
      delt=darcos(cdel)
      delta=delt*reprad
      sdel=dsin(delt)
      caze=(as-a*cdel)/(sdel*b)
      if(dabs(caze).gt.1.d0) caze=dsign(1.d0,caze)
      aze=darcos(caze)
      if(bs.gt.0.d0) cazs=(a-as*cdel)/(bs*sdel)
      if(bs.eq.0.d0) cazs=dsign(1.d0,cazs)
      if(dabs(cazs).gt.1.d0) cazs=dsign(1.d0,cazs)
      azs=darcos(cazs)
      dif=ds*c-cs*d
      if(dif.lt.0.d0) aze=twopi-aze
      azep=reprad*aze
      if(dif.gt.0.d0) azs=twopi-azs
      azst=reprad*azs
      return
      end
c
      subroutine pdaz(xla,xlo,az,del,yla,ylo)
      double precision xla8,xlo8,del8,az8,yla8,ylo8
c
      xla8=dble(xla)
      xlo8=dble(xlo)
      del8=dble(del)
      az8=dble(az)
c
      call dpdaz(xla8,xlo8,az8,del8,yla8,ylo8)
c
      yla=sngl(yla8)
      ylo=sngl(ylo8)
c
      return
      end

      SUBROUTINE DPDAZ(EPLA,EPLO,AZIM,DELTA,XLAT,XLON)
C
C     INPUT PARAMETERS:
C        EPLA : EPICENTRAL LATITUDE
C        EPLO : EPICENTRAL LONGITUDE
C        AZIM : AZIMUTH OF RECEIVER AS SEEN FROM EPICENTER
C        DELTA: ANGULAR DISTANCE
C     OUTPUT PARAMETERS:
C        XLAT : RECEIVER LATITUDE
C        XLON : RECEIVER LONGITUDE
C
C     AZIMUTH IS MEASURED COUNTERCLOCKWISE FROM THE NORTH
C     ALL ANGLES ARE IN DEGREES
C
C       [HOFFMAN TRADITIONAL ROUTINE]
C
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (RADIAN=180.D0/3.1415926535D0)
C      DATA RADIAN/57.29578/
      PH21=(180.D0-AZIM)/RADIAN
      TH1=(90.D0-EPLA)/RADIAN
      PH1=EPLO/RADIAN
      DEL=DELTA/RADIAN
      STH1=DSIN(TH1)
      CTH1=DCOS(TH1)
      SPH1=DSIN(PH1)
      CPH1=DCOS(PH1)
      SDEL=DSIN(DEL)
      CDEL=DCOS(DEL)
      CPH21=DCOS(PH21)
      SPH21=DSIN(PH21)
      CTH2=-SDEL*CPH21*STH1+CDEL*CTH1
      STH2=DSQRT(1.D0-CTH2*CTH2)
      CPH2=(SDEL*(CPH21*CTH1*CPH1-SPH21*SPH1)+CDEL*STH1*CPH1)
      SPH2=(SDEL*(CPH21*CTH1*SPH1+SPH21*CPH1)+CDEL*STH1*SPH1)
      TH2=DATAN2(STH2,CTH2)
      PH2=DATAN2(SPH2,CPH2)
      XLAT=90.D0-TH2*RADIAN
      XLON=PH2*RADIAN
      RETURN
      END


      function premgephvelo(modetype,omega)
      common /plevel/iprtlv
      logical splined
      dimension tcube(3,641)
      dimension scube(3,691)
      dimension work(3,1000)
      dimension tomega( 641)
      dimension tpvel( 641)
      dimension somega( 691)
      dimension spvel( 691)
      save
      data ntspl,nsspl / 641, 691/
      data splined/.false./
c
c     there are 641 toroidal modes to spline
c     there are 691 spheroidal modes to spline
c
      data tomega /
     #0.00238238,0.00368297,0.00481079,0.00583234,0.00677852,0.00766992,
     #0.00852074,0.00934068,0.01013648,0.01091298,0.01167381,0.01242183,
     #0.01315929,0.01388806,0.01460966,0.01532535,0.01603616,0.01674295,
     #0.01744643,0.01814720,0.01884573,0.01954244,0.02023764,0.02093163,
     #0.02162463,0.02231682,0.02300837,0.02369939,0.02439001,0.02508030,
     #0.02577035,0.02646020,0.02714991,0.02783952,0.02852906,0.02921855,
     #0.02990802,0.03059749,0.03128696,0.03197645,0.03266597,0.03335551,
     #0.03404509,0.03473471,0.03542437,0.03611406,0.03680379,0.03749356,
     #0.03818337,0.03887321,0.03956308,0.04025298,0.04094291,0.04163286,
     #0.04232283,0.04301282,0.04370283,0.04439285,0.04508287,0.04577290,
     #0.04646293,0.04715297,0.04784299,0.04853301,0.04922302,0.04991302,
     #0.05060299,0.05129296,0.05198289,0.05267281,0.05336269,0.05405255,
     #0.05474237,0.05543216,0.05612191,0.05681162,0.05750128,0.05819090,
     #0.05888047,0.05957000,0.06025946,0.06094887,0.06163822,0.06232752,
     #0.06301675,0.06370591,0.06439501,0.06508404,0.06577300,0.06646188,
     #0.06715069,0.06783942,0.06852807,0.06921664,0.06990512,0.07059351,
     #0.07128182,0.07197003,0.07265816,0.07334618,0.07403411,0.07472194,
     #0.07540967,0.07609729,0.07678480,0.07747222,0.07815952,0.07884670,
     #0.07953378,0.08022073,0.08090757,0.08159429,0.08228088,0.08296735,
     #0.08365369,0.08433990,0.08502598,0.08571193,0.08639774,0.08708341,
     #0.08776894,0.08845434,0.08913958,0.08982468,0.09050963,0.09119443,
     #0.09187907,0.09256356,0.09324789,0.09393206,0.09461607,0.09529991,
     #0.09598359,0.09666709,0.09735043,0.09803358,0.09871656,0.09939937,
     #0.10008199,0.10076442,0.10144667,0.10212874,0.10281061,0.10349228,
     #0.10417376,0.10485504,0.10553612,0.10621700,0.10689766,0.10757812,
     #0.10825837,0.10893840,0.10961822,0.11029781,0.11097719,0.11165634,
     #0.11233526,0.11301395,0.11369240,0.11437062,0.11504860,0.11572634,
     #0.11640384,0.11708109,0.11775809,0.11843483,0.11911132,0.11978756,
     #0.12046353,0.12113924,0.12181467,0.12248984,0.12316474,0.12383936,
     #0.12451369,0.12518775,0.12586153,0.12653501,0.12720822,0.12788111,
     #0.12855372,0.12922603,0.12989803,0.13056973,0.13124110,0.13191219,
     #0.13258293,0.13325338,0.13392350,0.13459329,0.13526276,0.13593189,
     #0.13660070,0.13726917,0.13793729,0.13860507,0.13927251,0.13993959,
     #0.14060633,0.14127271,0.14193873,0.14260440,0.14326969,0.14393461,
     #0.14459915,0.14526334,0.14592715,0.14659056,0.14725360,0.14791626,
     #0.14857852,0.14924039,0.14990187,0.15056294,0.15122361,0.15188389,
     #0.15254375,0.15320320,0.15386224,0.15452085,0.15517905,0.15583684,
     #0.15649419,0.15715112,0.15780762,0.15846367,0.15911929,0.15977448,
     #0.16042922,0.16108352,0.16173737,0.16239077,0.16304372,0.16369621,
     #0.16434824,0.16499981,0.16565092,0.16630156,0.16695173,0.16760144,
     #0.16825067,0.16889942,0.16954769,0.17019549,0.17084281,0.17148964,
     #0.17213598,0.17278183,0.17342718,0.17407204,0.17471641,0.17536029,
     #0.17600366,0.17664653,0.17728890,0.17793076,0.17857210,0.17921296,
     #0.17985329,0.18049310,0.18113241,0.18177120,0.18240947,0.18304722,
     #0.18368445,0.18432115,0.18495734,0.18559299,0.18622813,0.18686274,
     #0.18749681,0.18813035,0.18876337,0.18939584,0.19002779,0.19065920,
     #0.19129008,0.19192041,0.19255021,0.19317947,0.19380820,0.19443637,
     #0.19506401,0.19569111,0.19631766,0.19694367,0.19756915,0.19819407,
     #0.19881845,0.19944228,0.20006557,0.20068830,0.20131050,0.20193215,
     #0.20255324,0.20317379,0.20379379,0.20441325,0.20503215,0.20565052,
     #0.20626834,0.20688559,0.20750232,0.20811848,0.20873410,0.20934917,
     #0.20996369,0.21057768,0.21119110,0.21180399,0.21241632,0.21302812,
     #0.21363936,0.21425006,0.21486022,0.21546982,0.21607888,0.21668741,
     #0.21729538,0.21790282,0.21850972,0.21911606,0.21972188,0.22032715,
     #0.22093189,0.22153607,0.22213973,0.22274286,0.22334544,0.22394748,
     #0.22454900,0.22514997,0.22575043,0.22635035,0.22694974,0.22754858,
     #0.22814691,0.22874472,0.22934200,0.22993875,0.23053497,0.23113066,
     #0.23172584,0.23232050,0.23291464,0.23350826,0.23410136,0.23469394,
     #0.23528601,0.23587757,0.23646861,0.23705915,0.23764917,0.23823869,
     #0.23882769,0.23941620,0.24000420,0.24059169,0.24117868,0.24176517,
     #0.24235116,0.24293666,0.24352166,0.24410616,0.24469016,0.24527369,
     #0.24585672,0.24643925,0.24702130,0.24760287,0.24818395,0.24876454,
     #0.24934466,0.24992430,0.25050345,0.25108215,0.25166035,0.25223809,
     #0.25281534,0.25339213,0.25396845,0.25454432,0.25511968,0.25569463,
     #0.25626907,0.25684309,0.25741661,0.25798970,0.25856233,0.25913450,
     #0.25970623,0.26027748,0.26084831,0.26141867,0.26198861,0.26255807,
     #0.26312712,0.26369572,0.26426387,0.26483160,0.26539886,0.26596573,
     #0.26653212,0.26709813,0.26766366,0.26822880,0.26879349,0.26935777,
     #0.26992163,0.27048504,0.27104807,0.27161065,0.27217284,0.27273458,
     #0.27329594,0.27385688,0.27441740,0.27497754,0.27553725,0.27609655,
     #0.27665547,0.27721396,0.27777207,0.27832979,0.27888709,0.27944404,
     #0.28000057,0.28055668,0.28111243,0.28166780,0.28222278,0.28277737,
     #0.28333157,0.28388539,0.28443885,0.28499192,0.28554460,0.28609693,
     #0.28664887,0.28720045,0.28775164,0.28830248,0.28885296,0.28940305,
     #0.28995281,0.29050219,0.29105121,0.29159987,0.29214820,0.29269615,
     #0.29324377,0.29379100,0.29433790,0.29488447,0.29543066,0.29597652,
     #0.29652205,0.29706722,0.29761207,0.29815656,0.29870072,0.29924455,
     #0.29978806,0.30033121,0.30087405,0.30141655,0.30195871,0.30250058,
     #0.30304208,0.30358329,0.30412418,0.30466473,0.30520496,0.30574489,
     #0.30628452,0.30682382,0.30736279,0.30790147,0.30843982,0.30897790,
     #0.30951566,0.31005308,0.31059024,0.31112707,0.31166363,0.31219986,
     #0.31273583,0.31327146,0.31380683,0.31434190,0.31487668,0.31541115,
     #0.31594536,0.31647927,0.31701288,0.31754622,0.31807926,0.31861204,
     #0.31914455,0.31967676,0.32020870,0.32074037,0.32127178,0.32180288,
     #0.32233375,0.32286432,0.32339463,0.32392466,0.32445446,0.32498395,
     #0.32551321,0.32604221,0.32657093,0.32709941,0.32762760,0.32815558,
     #0.32868326,0.32921073,0.32973790,0.33026487,0.33079156,0.33131799,
     #0.33184418,0.33237013,0.33289585,0.33342132,0.33394656,0.33447152,
     #0.33499628,0.33552077,0.33604506,0.33656907,0.33709288,0.33761644,
     #0.33813977,0.33866286,0.33918574,0.33970839,0.34023082,0.34075302,
     #0.34127498,0.34179673,0.34231824,0.34283954,0.34336063,0.34388149,
     #0.34440213,0.34492257,0.34544277,0.34596276,0.34648257,0.34700215,
     #0.34752151,0.34804067,0.34855962,0.34907836,0.34959689,0.35011521,
     #0.35063332,0.35115126,0.35166898,0.35218650,0.35270384,0.35322094,
     #0.35373789,0.35425460,0.35477114,0.35528749,0.35580364,0.35631961,
     #0.35683537,0.35735095,0.35786632,0.35838154,0.35889655,0.35941136,
     #0.35992602,0.36044046,0.36095476,0.36146885,0.36198276,0.36249650,
     #0.36301005,0.36352345,0.36403665,0.36454967,0.36506253,0.36557522,
     #0.36608770,0.36660007,0.36711222,0.36762422,0.36813605,0.36864769,
     #0.36915919,0.36967054,0.37018168,0.37069270,0.37120351,0.37171420,
     #0.37222472,0.37273505,0.37324524,0.37375528,0.37426516,0.37477487,
     #0.37528443,0.37579384,0.37630311,0.37681219,0.37732115,0.37782994,
     #0.37833858,0.37884709,0.37935543,0.37986362,0.38037169,0.38087958,
     #0.38138735,0.38189498,0.38240242,0.38290977,0.38341695,0.38392398,
     #0.38443089,0.38493764,0.38544428,0.38595077,0.38645712,0.38696334,
     #0.38746941,0.38797534,0.38848114,0.38898683,0.38949236,0.38999778,
     #0.39050305,0.39100820,0.39151320,0.39201811,0.39252287/
c
      data tpvel /
     #6.07126904,6.70405579,6.81100893,6.75596619,6.64399719,6.51534510,
     #6.38654566,6.26415634,6.15042973,6.04579115,5.94990969,5.86218214,
     #5.78192043,5.70844126,5.64110088,5.57930231,5.52250528,5.47022200,
     #5.42201090,5.37747908,5.33627367,5.29807901,5.26261330,5.22962379,
     #5.19888639,5.17019796,5.14337873,5.11826563,5.09471321,5.07259130,
     #5.05178118,5.03217745,5.01368380,4.99621344,4.97968864,4.96403742,
     #4.94919538,4.93510389,4.92170954,4.90896368,4.89682102,4.88524055,
     #4.87418652,4.86362314,4.85351944,4.84384584,4.83457661,4.82568645,
     #4.81715298,4.80895519,4.80107403,4.79349089,4.78618860,4.77915239,
     #4.77236748,4.76582098,4.75949955,4.75339174,4.74748707,4.74177504,
     #4.73624563,4.73089075,4.72570038,4.72066927,4.71578741,4.71104908,
     #4.70644808,4.70197725,4.69763136,4.69340563,4.68929291,4.68528986,
     #4.68139172,4.67759371,4.67389154,4.67028189,4.66675997,4.66332388,
     #4.65996885,4.65669250,4.65349102,4.65036249,4.64730358,4.64431143,
     #4.64138365,4.63851833,4.63571310,4.63296556,4.63027382,4.62763548,
     #4.62504911,4.62251282,4.62002468,4.61758327,4.61518669,4.61283350,
     #4.61052275,4.60825205,4.60602093,4.60382795,4.60167170,4.59955025,
     #4.59746361,4.59541082,4.59338951,4.59140015,4.58944035,4.58751011,
     #4.58560801,4.58373356,4.58188534,4.58006382,4.57826614,4.57649326,
     #4.57474375,4.57301712,4.57131243,4.56962919,4.56796694,4.56632423,
     #4.56470108,4.56309795,4.56151199,4.55994415,4.55839396,4.55686045,
     #4.55534315,4.55384159,4.55235529,4.55088329,4.54942656,4.54798269,
     #4.54655313,4.54513693,4.54373312,4.54234171,4.54096174,4.53959417,
     #4.53823709,4.53689146,4.53555632,4.53423119,4.53291607,4.53161049,
     #4.53031445,4.52902651,4.52774811,4.52647829,4.52521563,4.52396154,
     #4.52271509,4.52147579,4.52024412,4.51901865,4.51779985,4.51658773,
     #4.51538134,4.51418066,4.51298618,4.51179695,4.51061344,4.50943470,
     #4.50826073,4.50709152,4.50592661,4.50476599,4.50360966,4.50245762,
     #4.50130892,4.50016356,4.49902153,4.49788332,4.49674797,4.49561548,
     #4.49448586,4.49335861,4.49223423,4.49111176,4.48999214,4.48887348,
     #4.48775768,4.48664331,4.48553038,4.48441887,4.48330832,4.48220015,
     #4.48109198,4.47998571,4.47887993,4.47777510,4.47667027,4.47556639,
     #4.47446299,4.47336006,4.47225666,4.47115421,4.47005129,4.46894789,
     #4.46784496,4.46674156,4.46563768,4.46453381,4.46342897,4.46232319,
     #4.46121645,4.46010971,4.45900154,4.45789242,4.45678234,4.45567131,
     #4.45455885,4.45344496,4.45233011,4.45121336,4.45009565,4.44897556,
     #4.44785452,4.44673157,4.44560719,4.44448042,4.44335175,4.44222164,
     #4.44108915,4.43995476,4.43881845,4.43767929,4.43653822,4.43539524,
     #4.43424988,4.43310165,4.43195152,4.43079901,4.42964411,4.42848635,
     #4.42732620,4.42616367,4.42499828,4.42382956,4.42265940,4.42148542,
     #4.42030907,4.41913033,4.41794825,4.41676331,4.41557646,4.41438627,
     #4.41319275,4.41199589,4.41079664,4.40959406,4.40838909,4.40718126,
     #4.40597010,4.40475559,4.40353823,4.40231752,4.40109444,4.39986801,
     #4.39863873,4.39740562,4.39616966,4.39493084,4.39368916,4.39244366,
     #4.39119577,4.38994408,4.38868952,4.38743210,4.38617134,4.38490772,
     #4.38364124,4.38237095,4.38109827,4.37982178,4.37854242,4.37726021,
     #4.37597513,4.37468672,4.37339544,4.37210131,4.37080383,4.36950302,
     #4.36819983,4.36689329,4.36558390,4.36427164,4.36295652,4.36163902,
     #4.36031818,4.35899401,4.35766745,4.35633802,4.35500574,4.35367107,
     #4.35233259,4.35099220,4.34964895,4.34830332,4.34695482,4.34560347,
     #4.34425020,4.34289312,4.34153461,4.34017277,4.33880901,4.33744240,
     #4.33607340,4.33470249,4.33332872,4.33195257,4.33057404,4.32919359,
     #4.32781076,4.32642508,4.32503748,4.32364798,4.32225609,4.32086229,
     #4.31946611,4.31806803,4.31666803,4.31526566,4.31386185,4.31245565,
     #4.31104755,4.30963755,4.30822563,4.30681229,4.30539751,4.30398035,
     #4.30256128,4.30114126,4.29971886,4.29829550,4.29686975,4.29544306,
     #4.29401445,4.29258490,4.29115343,4.28972101,4.28828716,4.28685141,
     #4.28541470,4.28397655,4.28253746,4.28109694,4.27965498,4.27821207,
     #4.27676821,4.27532291,4.27387667,4.27242947,4.27098131,4.26953220,
     #4.26808167,4.26663113,4.26517916,4.26372671,4.26227283,4.26081848,
     #4.25936365,4.25790739,4.25645113,4.25499392,4.25353622,4.25207806,
     #4.25061893,4.24915934,4.24769974,4.24623919,4.24477816,4.24331713,
     #4.24185514,4.24039316,4.23893118,4.23746872,4.23600578,4.23454237,
     #4.23307896,4.23161507,4.23015165,4.22868824,4.22722340,4.22575998,
     #4.22429562,4.22283173,4.22136736,4.21990347,4.21843910,4.21697569,
     #4.21551180,4.21404791,4.21258450,4.21112061,4.20965815,4.20819473,
     #4.20673227,4.20526981,4.20380783,4.20234680,4.20088482,4.19942427,
     #4.19796324,4.19650316,4.19504356,4.19358444,4.19212580,4.19066763,
     #4.18920994,4.18775272,4.18629646,4.18484020,4.18338537,4.18193007,
     #4.18047667,4.17902327,4.17757034,4.17611885,4.17466784,4.17321730,
     #4.17176819,4.17031908,4.16887093,4.16742468,4.16597795,4.16453314,
     #4.16308928,4.16164541,4.16020298,4.15876150,4.15732098,4.15588140,
     #4.15444279,4.15300512,4.15156889,4.15013361,4.14869928,4.14726639,
     #4.14583445,4.14440346,4.14297342,4.14154482,4.14011765,4.13869095,
     #4.13726616,4.13584232,4.13441992,4.13299847,4.13157892,4.13015985,
     #4.12874269,4.12732601,4.12591171,4.12449837,4.12308598,4.12167501,
     #4.12026596,4.11885786,4.11745167,4.11604643,4.11464310,4.11324072,
     #4.11184025,4.11044073,4.10904312,4.10764647,4.10625172,4.10485888,
     #4.10346699,4.10207653,4.10068798,4.09930086,4.09791517,4.09653139,
     #4.09514904,4.09376860,4.09238958,4.09101200,4.08963585,4.08826208,
     #4.08688974,4.08551836,4.08414936,4.08278179,4.08141613,4.08005190,
     #4.07868958,4.07732916,4.07597017,4.07461309,4.07325745,4.07190371,
     #4.07055187,4.06920147,4.06785297,4.06650639,4.06516171,4.06381845,
     #4.06247711,4.06113768,4.05980015,4.05846453,4.05713081,4.05579853,
     #4.05446863,4.05313969,4.05181360,4.05048895,4.04916620,4.04784489,
     #4.04652643,4.04520893,4.04389429,4.04258108,4.04126883,4.03995991,
     #4.03865194,4.03734684,4.03604269,4.03474092,4.03344154,4.03214312,
     #4.03084707,4.02955294,4.02826118,4.02697134,4.02568293,4.02439642,
     #4.02311277,4.02183008,4.02055025,4.01927185,4.01799583,4.01672125,
     #4.01544952,4.01417875,4.01291084,4.01164436,4.01038027,4.00911808,
     #4.00785780,4.00659943,4.00534344,4.00408936,4.00283718,4.00158691,
     #4.00033903,3.99909329,3.99784899,3.99660683,3.99536729,3.99412966,
     #3.99289370,3.99166012,3.99042821,3.98919868,3.98797083,3.98674536,
     #3.98552179,3.98430014,3.98308110,3.98186374,3.98064876,3.97943521,
     #3.97822428,3.97701502,3.97580791,3.97460341,3.97340035,3.97219992,
     #3.97100115,3.96980453,3.96860981,3.96741748,3.96622729,3.96503854,
     #3.96385241,3.96266818,3.96148634,3.96030617,3.95912814,3.95795226,
     #3.95677853,3.95560694,3.95443749,3.95326972,3.95210457,3.95094109,
     #3.94977927,3.94862080,3.94746351,3.94630837,3.94515514,3.94400430,
     #3.94285560,3.94170856,3.94056392,3.93942142,3.93828058,3.93714237,
     #3.93600607,3.93487167,3.93373919,3.93260908,3.93148112,3.93035507,
     #3.92923093,3.92810941,3.92698932,3.92587137,3.92475605,3.92364240,
     #3.92253065,3.92142129,3.92031384,3.91920829,3.91810536,3.91700363,
     #3.91590476,3.91480780,3.91371226,3.91261911,3.91152835,3.91043925,
     #3.90935230,3.90826726,3.90718460,3.90610385,3.90502477,3.90394855,
     #3.90287375,3.90180063,3.90073037,3.89966178,3.89859509,3.89753056,
     #3.89646840,3.89540768,3.89434910,3.89329290,3.89223838/
c
      data somega /
     #0.00194327,0.00294407,0.00406573,0.00528063,0.00652341,0.00773970,
     #0.00888144,0.00991674,0.01084782,0.01170198,0.01250597,0.01327605,
     #0.01402037,0.01474283,0.01544549,0.01612975,0.01679691,0.01744832,
     #0.01808540,0.01870965,0.01932258,0.01992564,0.02052023,0.02110765,
     #0.02168905,0.02226549,0.02283790,0.02340707,0.02397370,0.02453840,
     #0.02510165,0.02566390,0.02622549,0.02678672,0.02734783,0.02790902,
     #0.02847045,0.02903226,0.02959453,0.03015736,0.03072079,0.03128488,
     #0.03184966,0.03241516,0.03298137,0.03354831,0.03411598,0.03468437,
     #0.03525348,0.03582330,0.03639382,0.03696501,0.03753688,0.03810939,
     #0.03868255,0.03925632,0.03983070,0.04040567,0.04098121,0.04155731,
     #0.04213396,0.04271114,0.04328884,0.04386705,0.04444575,0.04502493,
     #0.04560458,0.04618470,0.04676526,0.04734627,0.04792771,0.04850958,
     #0.04909186,0.04967454,0.05025763,0.05084112,0.05142500,0.05200925,
     #0.05259388,0.05317889,0.05376425,0.05434998,0.05493607,0.05552250,
     #0.05610928,0.05669640,0.05728386,0.05787165,0.05845978,0.05904823,
     #0.05963700,0.06022609,0.06081549,0.06140521,0.06199524,0.06258557,
     #0.06317620,0.06376713,0.06435835,0.06494987,0.06554167,0.06613376,
     #0.06672613,0.06731878,0.06791171,0.06850491,0.06909837,0.06969211,
     #0.07028610,0.07088035,0.07147486,0.07206962,0.07266463,0.07325988,
     #0.07385538,0.07445111,0.07504708,0.07564328,0.07623971,0.07683637,
     #0.07743324,0.07803034,0.07862765,0.07922517,0.07982290,0.08042083,
     #0.08101895,0.08161729,0.08221581,0.08281451,0.08341341,0.08401249,
     #0.08461174,0.08521118,0.08581078,0.08641055,0.08701049,0.08761058,
     #0.08821084,0.08881124,0.08941180,0.09001251,0.09061335,0.09121433,
     #0.09181546,0.09241670,0.09301808,0.09361959,0.09422123,0.09482297,
     #0.09542483,0.09602681,0.09662889,0.09723108,0.09783337,0.09843575,
     #0.09903824,0.09964081,0.10024347,0.10084622,0.10144904,0.10205195,
     #0.10265493,0.10325799,0.10386112,0.10446431,0.10506756,0.10567087,
     #0.10627424,0.10687767,0.10748114,0.10808467,0.10868824,0.10929185,
     #0.10989551,0.11049920,0.11110292,0.11170667,0.11231045,0.11291426,
     #0.11351810,0.11412195,0.11472582,0.11532971,0.11593360,0.11653751,
     #0.11714143,0.11774535,0.11834928,0.11895320,0.11955712,0.12016104,
     #0.12076496,0.12136886,0.12197275,0.12257662,0.12318048,0.12378433,
     #0.12438814,0.12499195,0.12559572,0.12619947,0.12680319,0.12740687,
     #0.12801053,0.12861414,0.12921771,0.12982126,0.13042475,0.13102821,
     #0.13163161,0.13223498,0.13283829,0.13344155,0.13404475,0.13464791,
     #0.13525100,0.13585404,0.13645701,0.13705993,0.13766278,0.13826557,
     #0.13886827,0.13947092,0.14007349,0.14067599,0.14127842,0.14188077,
     #0.14248304,0.14308524,0.14368735,0.14428937,0.14489132,0.14549318,
     #0.14609495,0.14669663,0.14729822,0.14789970,0.14850111,0.14910242,
     #0.14970362,0.15030473,0.15090574,0.15150666,0.15210746,0.15270817,
     #0.15330876,0.15390925,0.15450963,0.15510990,0.15571006,0.15631010,
     #0.15691003,0.15750985,0.15810955,0.15870912,0.15930858,0.15990791,
     #0.16050713,0.16110621,0.16170518,0.16230401,0.16290273,0.16350131,
     #0.16409975,0.16469806,0.16529624,0.16589430,0.16649221,0.16708998,
     #0.16768761,0.16828510,0.16888246,0.16947967,0.17007674,0.17067365,
     #0.17127043,0.17186706,0.17246354,0.17305987,0.17365605,0.17425206,
     #0.17484793,0.17544365,0.17603920,0.17663461,0.17722985,0.17782493,
     #0.17841986,0.17901461,0.17960921,0.18020363,0.18079790,0.18139200,
     #0.18198591,0.18257968,0.18317325,0.18376668,0.18435991,0.18495297,
     #0.18554586,0.18613857,0.18673111,0.18732347,0.18791564,0.18850763,
     #0.18909943,0.18969105,0.19028249,0.19087374,0.19146481,0.19205567,
     #0.19264635,0.19323686,0.19382715,0.19441725,0.19500716,0.19559687,
     #0.19618639,0.19677570,0.19736482,0.19795375,0.19854246,0.19913097,
     #0.19971928,0.20030737,0.20089526,0.20148295,0.20207043,0.20265770,
     #0.20324475,0.20383158,0.20441821,0.20500463,0.20559081,0.20617680,
     #0.20676255,0.20734809,0.20793341,0.20851851,0.20910338,0.20968802,
     #0.21027245,0.21085663,0.21144059,0.21202433,0.21260783,0.21319111,
     #0.21377414,0.21435696,0.21493952,0.21552186,0.21610394,0.21668580,
     #0.21726742,0.21784879,0.21842992,0.21901082,0.21959145,0.22017185,
     #0.22075200,0.22133189,0.22191155,0.22249095,0.22307010,0.22364898,
     #0.22422762,0.22480601,0.22538413,0.22596200,0.22653961,0.22711696,
     #0.22769405,0.22827087,0.22884743,0.22942372,0.22999975,0.23057552,
     #0.23115100,0.23172623,0.23230118,0.23287585,0.23345025,0.23402438,
     #0.23459823,0.23517181,0.23574510,0.23631811,0.23689085,0.23746331,
     #0.23803547,0.23860736,0.23917896,0.23975027,0.24032129,0.24089204,
     #0.24146247,0.24203263,0.24260248,0.24317205,0.24374132,0.24431030,
     #0.24487898,0.24544735,0.24601543,0.24658321,0.24715067,0.24771784,
     #0.24828471,0.24885127,0.24941753,0.24998347,0.25054911,0.25111443,
     #0.25167942,0.25224414,0.25280851,0.25337258,0.25393632,0.25449976,
     #0.25506288,0.25562567,0.25618815,0.25675029,0.25731212,0.25787362,
     #0.25843480,0.25899565,0.25955617,0.26011637,0.26067621,0.26123574,
     #0.26179492,0.26235378,0.26291230,0.26347050,0.26402834,0.26458585,
     #0.26514304,0.26569986,0.26625633,0.26681247,0.26736829,0.26792374,
     #0.26847884,0.26903358,0.26958799,0.27014205,0.27069578,0.27124912,
     #0.27180210,0.27235475,0.27290705,0.27345896,0.27401054,0.27456176,
     #0.27511260,0.27566311,0.27621323,0.27676299,0.27731240,0.27786142,
     #0.27841008,0.27895838,0.27950630,0.28005385,0.28060105,0.28114787,
     #0.28169429,0.28224036,0.28278604,0.28333136,0.28387630,0.28442085,
     #0.28496504,0.28550881,0.28605223,0.28659526,0.28713790,0.28768015,
     #0.28822201,0.28876352,0.28930461,0.28984532,0.29038563,0.29092556,
     #0.29146507,0.29200423,0.29254296,0.29308131,0.29361928,0.29415682,
     #0.29469398,0.29523075,0.29576710,0.29630306,0.29683861,0.29737377,
     #0.29790851,0.29844284,0.29897678,0.29951030,0.30004343,0.30057615,
     #0.30110845,0.30164033,0.30217183,0.30270287,0.30323353,0.30376378,
     #0.30429360,0.30482301,0.30535200,0.30588058,0.30640873,0.30693647,
     #0.30746379,0.30799067,0.30851716,0.30904320,0.30956882,0.31009403,
     #0.31061882,0.31114316,0.31166708,0.31219059,0.31271365,0.31323630,
     #0.31375849,0.31428027,0.31480163,0.31532255,0.31584302,0.31636307,
     #0.31688270,0.31740186,0.31792063,0.31843892,0.31895679,0.31947422,
     #0.31999123,0.32050776,0.32102388,0.32153955,0.32205480,0.32256958,
     #0.32308394,0.32359785,0.32411131,0.32462433,0.32513690,0.32564902,
     #0.32616070,0.32667193,0.32718271,0.32769307,0.32820296,0.32871240,
     #0.32922140,0.32972994,0.33023801,0.33074567,0.33125284,0.33175960,
     #0.33226588,0.33277172,0.33327708,0.33378202,0.33428648,0.33479050,
     #0.33529407,0.33579716,0.33629981,0.33680201,0.33730373,0.33780500,
     #0.33830583,0.33880618,0.33930609,0.33980551,0.34030452,0.34080303,
     #0.34130111,0.34179869,0.34229586,0.34279254,0.34328875,0.34378451,
     #0.34427980,0.34477463,0.34526902,0.34576294,0.34625638,0.34674937,
     #0.34724188,0.34773394,0.34822553,0.34871668,0.34920734,0.34969753,
     #0.35018727,0.35067654,0.35116535,0.35165370,0.35214156,0.35262898,
     #0.35311592,0.35360241,0.35408843,0.35457397,0.35505906,0.35554367,
     #0.35602784,0.35651150,0.35699475,0.35747749,0.35795978,0.35844159,
     #0.35892296,0.35940385,0.35988426,0.36036423,0.36084372,0.36132273,
     #0.36180130,0.36227939,0.36275703,0.36323419,0.36371088,0.36418709,
     #0.36466286,0.36513817,0.36561298,0.36608738,0.36656126,0.36703470,
     #0.36750767,0.36798018,0.36845222,0.36892381,0.36939493,0.36986557,
     #0.37033576,0.37080547,0.37127474,0.37174353,0.37221187,0.37267974,
     #0.37314713,0.37361407,0.37408057,0.37454659,0.37501213,0.37547722,
     #0.37594187,0.37640604,0.37686977,0.37733302,0.37779582,0.37825817,
     #0.37872005,0.37918144,0.37964243,0.38010293,0.38056296,0.38102254,
     #0.38148168,0.38194036,0.38239858,0.38285634,0.38331366,0.38377050,
     #0.38422689,0.38468283,0.38513833,0.38559338,0.38604796,0.38650209,
     #0.38695577,0.38740900,0.38786179,0.38831413,0.38876599,0.38921744,
     #0.38966841,0.39011896,0.39056903,0.39101869,0.39146787,0.39191663,
     #0.39236492/
c
      data spvel /
     #4.95222807,5.35905170,5.75617504,6.11689281,6.39394808,6.57461500,
     #6.65690279,6.65047598,6.58204365,6.48289680,6.37404108,6.26531172,
     #6.16026163,6.05978012,5.96382999,5.87215090,5.78449345,5.70067883,
     #5.62058926,5.54414797,5.47129488,5.40196800,5.33609867,5.27360106,
     #5.21437550,5.15830755,5.10527134,5.05513334,5.00775290,4.96298838,
     #4.92069674,4.88073778,4.84297371,4.80727339,4.77350760,4.74155664,
     #4.71130562,4.68264627,4.65547562,4.62969923,4.60522747,4.58197689,
     #4.55986977,4.53883410,4.51880169,4.49971104,4.48150301,4.46412420,
     #4.44752359,4.43165588,4.41647625,4.40194559,4.38802671,4.37468338,
     #4.36188507,4.34960032,4.33780146,4.32646227,4.31555891,4.30506706,
     #4.29496717,4.28523922,4.27586365,4.26682377,4.25810289,4.24968624,
     #4.24155903,4.23370790,4.22612047,4.21878481,4.21168900,4.20482302,
     #4.19817734,4.19174194,4.18550825,4.17946815,4.17361307,4.16793633,
     #4.16243029,4.15708828,4.15190411,4.14687109,4.14198446,4.13723803,
     #4.13262701,4.12814617,4.12379122,4.11955643,4.11543894,4.11143446,
     #4.10753870,4.10374784,4.10005808,4.09646702,4.09297037,4.08956575,
     #4.08624935,4.08301878,4.07987118,4.07680416,4.07381439,4.07090044,
     #4.06805897,4.06528902,4.06258678,4.05995131,4.05737972,4.05487108,
     #4.05242300,4.05003309,4.04770088,4.04542303,4.04319906,4.04102755,
     #4.03890657,4.03683424,4.03481007,4.03283119,4.03089809,4.02900839,
     #4.02716064,4.02535439,4.02358818,4.02186060,4.02017164,4.01851845,
     #4.01690102,4.01531887,4.01376963,4.01225281,4.01076889,4.00931501,
     #4.00789165,4.00649786,4.00513172,4.00379372,4.00248241,4.00119686,
     #3.99993753,3.99870253,3.99749184,3.99630427,3.99513960,3.99399638,
     #3.99287581,3.99177504,3.99069500,3.98963499,3.98859429,3.98757219,
     #3.98656797,3.98558187,3.98461270,3.98366022,3.98272443,3.98180413,
     #3.98089957,3.98001003,3.97913504,3.97827411,3.97742677,3.97659302,
     #3.97577262,3.97496486,3.97416902,3.97338581,3.97261381,3.97185326,
     #3.97110367,3.97036529,3.96963716,3.96891880,3.96821070,3.96751237,
     #3.96682310,3.96614289,3.96547151,3.96480918,3.96415472,3.96350837,
     #3.96287012,3.96223927,3.96161628,3.96100044,3.96039128,3.95978928,
     #3.95919394,3.95860457,3.95802212,3.95744562,3.95687485,3.95631027,
     #3.95575094,3.95519686,3.95464826,3.95410466,3.95356584,3.95303226,
     #3.95250297,3.95197845,3.95145845,3.95094275,3.95043087,3.94992304,
     #3.94941926,3.94891906,3.94842219,3.94792962,3.94743991,3.94695377,
     #3.94647050,3.94599080,3.94551396,3.94503999,3.94456863,3.94410014,
     #3.94363451,3.94317102,3.94271016,3.94225192,3.94179606,3.94134188,
     #3.94088960,3.94043994,3.93999195,3.93954611,3.93910193,3.93865943,
     #3.93821883,3.93777990,3.93734241,3.93690610,3.93647170,3.93603849,
     #3.93560648,3.93517542,3.93474603,3.93431735,3.93389010,3.93346381,
     #3.93303823,3.93261385,3.93219018,3.93176770,3.93134546,3.93092418,
     #3.93050361,3.93008351,3.92966413,3.92924500,3.92882681,3.92840886,
     #3.92799139,3.92757440,3.92715764,3.92674088,3.92632484,3.92590857,
     #3.92549276,3.92507720,3.92466187,3.92424607,3.92383099,3.92341566,
     #3.92299986,3.92258453,3.92216873,3.92175364,3.92133760,3.92092180,
     #3.92050552,3.92008924,3.91967273,3.91925597,3.91883898,3.91842103,
     #3.91800332,3.91758490,3.91716623,3.91674757,3.91632795,3.91590810,
     #3.91548753,3.91506648,3.91464496,3.91422296,3.91380024,3.91337681,
     #3.91295314,3.91252851,3.91210341,3.91167760,3.91125083,3.91082382,
     #3.91039538,3.90996718,3.90953708,3.90910673,3.90867567,3.90824366,
     #3.90781045,3.90737653,3.90694237,3.90650654,3.90607047,3.90563321,
     #3.90519428,3.90475512,3.90431499,3.90387368,3.90343118,3.90298772,
     #3.90254354,3.90209842,3.90165186,3.90120411,3.90075541,3.90030575,
     #3.89985514,3.89940262,3.89894962,3.89849567,3.89804029,3.89758348,
     #3.89712548,3.89666629,3.89620614,3.89574456,3.89528203,3.89481807,
     #3.89435291,3.89388585,3.89341831,3.89294934,3.89247870,3.89200711,
     #3.89153385,3.89105940,3.89058375,3.89010668,3.88962817,3.88914800,
     #3.88866687,3.88818383,3.88769984,3.88721442,3.88672757,3.88623929,
     #3.88574910,3.88525796,3.88476491,3.88427067,3.88377500,3.88327789,
     #3.88277936,3.88227892,3.88177705,3.88127375,3.88076901,3.88026237,
     #3.87975454,3.87924480,3.87873387,3.87822127,3.87770677,3.87719107,
     #3.87667346,3.87615466,3.87563348,3.87511134,3.87458754,3.87406206,
     #3.87353492,3.87300587,3.87247539,3.87194300,3.87140918,3.87087369,
     #3.87033629,3.86979747,3.86925721,3.86871481,3.86817050,3.86762452,
     #3.86707735,3.86652780,3.86597681,3.86542392,3.86486959,3.86431360,
     #3.86375523,3.86319566,3.86263394,3.86207080,3.86150551,3.86093879,
     #3.86036992,3.85979939,3.85922694,3.85865283,3.85807705,3.85749936,
     #3.85691977,3.85633826,3.85575461,3.85516953,3.85458255,3.85399342,
     #3.85340285,3.85281014,3.85221601,3.85161972,3.85102153,3.85042119,
     #3.84981894,3.84921551,3.84860945,3.84800172,3.84739184,3.84678054,
     #3.84616709,3.84555149,3.84493446,3.84431505,3.84369397,3.84307098,
     #3.84244609,3.84181905,3.84119034,3.84055948,3.83992624,3.83929157,
     #3.83865476,3.83801603,3.83737516,3.83673263,3.83608794,3.83544135,
     #3.83479309,3.83414221,3.83348942,3.83283496,3.83217835,3.83151984,
     #3.83085918,3.83019662,3.82953215,3.82886529,3.82819700,3.82752633,
     #3.82685328,3.82617879,3.82550240,3.82482338,3.82414269,3.82346010,
     #3.82277489,3.82208848,3.82139969,3.82070875,3.82001591,3.81932068,
     #3.81862354,3.81792450,3.81722331,3.81652021,3.81581497,3.81510782,
     #3.81439805,3.81368685,3.81297326,3.81225801,3.81154037,3.81082058,
     #3.81009912,3.80937529,3.80864930,3.80792141,3.80719161,3.80645943,
     #3.80572534,3.80498934,3.80425119,3.80351090,3.80276847,3.80202413,
     #3.80127716,3.80052876,3.79977822,3.79902554,3.79827094,3.79751396,
     #3.79675508,3.79599404,3.79523087,3.79446602,3.79369879,3.79292965,
     #3.79215813,3.79138470,3.79060888,3.78983164,3.78905177,3.78827047,
     #3.78748655,3.78670073,3.78591299,3.78512263,3.78433084,3.78353667,
     #3.78274059,3.78194213,3.78114200,3.78033972,3.77953529,3.77872920,
     #3.77792048,3.77710986,3.77629757,3.77548265,3.77466583,3.77384710,
     #3.77302670,3.77220368,3.77137899,3.77055216,3.76972318,3.76889229,
     #3.76805902,3.76722431,3.76638722,3.76554823,3.76470685,3.76386380,
     #3.76301885,3.76217151,3.76132274,3.76047134,3.75961828,3.75876307,
     #3.75790644,3.75704694,3.75618577,3.75532269,3.75445747,3.75359058,
     #3.75272131,3.75185061,3.75097752,3.75010276,3.74922585,3.74834681,
     #3.74746561,3.74658298,3.74569798,3.74481201,3.74392295,3.74303269,
     #3.74214005,3.74124575,3.74034905,3.73945093,3.73855066,3.73764873,
     #3.73674488,3.73583889,3.73493123,3.73402143,3.73310995,3.73219633,
     #3.73128104,3.73036408,3.72944498,3.72852421,3.72760129,3.72667670,
     #3.72575021,3.72482181,3.72389150,3.72295928,3.72202587,3.72109008,
     #3.72015309,3.71921325,3.71827245,3.71732998,3.71638489,3.71543884,
     #3.71449018,3.71354032,3.71258903,3.71163559,3.71068001,3.70972323,
     #3.70876455,3.70780396,3.70684195,3.70587802,3.70491266,3.70394492,
     #3.70297623,3.70200515,3.70103288,3.70005894,3.69908309,3.69810557,
     #3.69712639,3.69614601,3.69516373,3.69417953,3.69319391,3.69220662,
     #3.69121790,3.69022703,3.68923521,3.68824124,3.68724608,3.68624926,
     #3.68525100,3.68425083,3.68324947,3.68224645,3.68124151,3.68023515,
     #3.67922759,3.67821836,3.67720771,3.67619538,3.67518139,3.67416596,
     #3.67314959,3.67213178,3.67111158,3.67009091,3.66906810,3.66804409,
     #3.66701865,3.66599202,3.66496348,3.66393375,3.66290283,3.66187024,
     #3.66083670,3.65980101,3.65876460,3.65772653,3.65668750,3.65564680,
     #3.65460467,3.65356135,3.65251708,3.65147114,3.65042377,3.64937496,
     #3.64832568,3.64727449,3.64622235,3.64516830,3.64411402,3.64305782,
     #3.64200068,3.64094186,3.63988256,3.63882160,3.63775945,3.63669610,
     #3.63563156,3.63456607,3.63349915,3.63243151,3.63136244,3.63029218,
     #3.62922072,3.62814879,3.62707520,3.62600064,3.62492490,3.62384820,
     #3.62277031,3.62169170,3.62061167,3.61953068,3.61844873,3.61736584,
     #3.61628175,3.61519670,3.61411071,3.61302423,3.61193585,3.61084723,
     #3.60975718/
c
      if(.not.splined) then
        if(iprtlv.gt.1) then
        write(6,"('there are ',i5,' toroidal modes in spline')") ntspl
        write(6,"('there are ',i5,' spheroidal modes in spline')") nsspl
        endif
        call rspln(1,ntspl,tomega,tpvel,tcube,work)
        call rspln(1,nsspl,somega,spvel,scube,work)
        splined=.true.
      endif
c
      if(modetype.eq.1) then
        pvel=rsple(1,ntspl,tomega,tpvel,tcube,omega)
      else
        pvel=rsple(1,nsspl,somega,spvel,scube,omega)
      endif
      premgephvelo=pvel
      return
      end

      function premgegrvelo(modetype,omega)
      common /plevel/iprtlv
      logical splined
      dimension tcube(3,641)
      dimension scube(3,691)
      dimension work(3,1000)
c     there are 641 toroidal modes to spline
c     there are 691 spheroidal modes to spline
      dimension tomega( 641)
      dimension tpvel( 641)
      dimension somega( 691)
      dimension spvel( 691)
      save
      data ntspl,nsspl / 641, 691/
      data splined/.false./
c
      data tomega /
     #0.00238238,0.00368296,0.00481078,0.00583232,0.00677851,0.00766991,
     #0.00852072,0.00934066,0.01013646,0.01091296,0.01167379,0.01242180,
     #0.01315925,0.01388803,0.01460963,0.01532531,0.01603612,0.01674291,
     #0.01744640,0.01814716,0.01884570,0.01954241,0.02023762,0.02093162,
     #0.02162463,0.02231681,0.02300836,0.02369938,0.02438999,0.02508028,
     #0.02577032,0.02646016,0.02714987,0.02783948,0.02852901,0.02921851,
     #0.02990798,0.03059744,0.03128691,0.03197641,0.03266592,0.03335546,
     #0.03404504,0.03473467,0.03542432,0.03611401,0.03680375,0.03749353,
     #0.03818334,0.03887319,0.03956307,0.04025297,0.04094291,0.04163286,
     #0.04232284,0.04301283,0.04370283,0.04439285,0.04508287,0.04577290,
     #0.04646293,0.04715297,0.04784299,0.04853301,0.04922302,0.04991302,
     #0.05060300,0.05129296,0.05198290,0.05267281,0.05336270,0.05405255,
     #0.05474238,0.05543216,0.05612191,0.05681162,0.05750129,0.05819090,
     #0.05888048,0.05957000,0.06025946,0.06094887,0.06163823,0.06232752,
     #0.06301675,0.06370591,0.06439502,0.06508405,0.06577300,0.06646188,
     #0.06715070,0.06783942,0.06852807,0.06921664,0.06990512,0.07059351,
     #0.07128182,0.07197004,0.07265816,0.07334618,0.07403411,0.07472194,
     #0.07540967,0.07609729,0.07678481,0.07747222,0.07815952,0.07884671,
     #0.07953378,0.08022073,0.08090757,0.08159429,0.08228088,0.08296735,
     #0.08365369,0.08433990,0.08502598,0.08571193,0.08639774,0.08708341,
     #0.08776895,0.08845434,0.08913958,0.08982468,0.09050963,0.09119444,
     #0.09187908,0.09256357,0.09324790,0.09393207,0.09461608,0.09529992,
     #0.09598360,0.09666710,0.09735043,0.09803360,0.09871657,0.09939939,
     #0.10008200,0.10076445,0.10144669,0.10212876,0.10281062,0.10349231,
     #0.10417378,0.10485508,0.10553615,0.10621703,0.10689768,0.10757814,
     #0.10825840,0.10893843,0.10961825,0.11029785,0.11097722,0.11165637,
     #0.11233529,0.11301398,0.11369244,0.11437066,0.11504863,0.11572637,
     #0.11640386,0.11708110,0.11775810,0.11843483,0.11911132,0.11978755,
     #0.12046353,0.12113923,0.12181467,0.12248984,0.12316474,0.12383936,
     #0.12451369,0.12518775,0.12586153,0.12653501,0.12720822,0.12788111,
     #0.12855372,0.12922603,0.12989803,0.13056971,0.13124110,0.13191219,
     #0.13258293,0.13325338,0.13392350,0.13459329,0.13526276,0.13593189,
     #0.13660070,0.13726917,0.13793729,0.13860507,0.13927251,0.13993959,
     #0.14060633,0.14127271,0.14193873,0.14260440,0.14326969,0.14393461,
     #0.14459917,0.14526334,0.14592715,0.14659058,0.14725360,0.14791626,
     #0.14857852,0.14924039,0.14990187,0.15056294,0.15122361,0.15188389,
     #0.15254375,0.15320320,0.15386224,0.15452085,0.15517907,0.15583684,
     #0.15649420,0.15715112,0.15780762,0.15846367,0.15911929,0.15977448,
     #0.16042922,0.16108352,0.16173738,0.16239077,0.16304372,0.16369621,
     #0.16434824,0.16499981,0.16565092,0.16630156,0.16695175,0.16760144,
     #0.16825068,0.16889943,0.16954771,0.17019551,0.17084281,0.17148964,
     #0.17213598,0.17278183,0.17342719,0.17407206,0.17471643,0.17536029,
     #0.17600366,0.17664653,0.17728890,0.17793076,0.17857212,0.17921296,
     #0.17985329,0.18049312,0.18113242,0.18177120,0.18240948,0.18304722,
     #0.18368445,0.18432117,0.18495734,0.18559301,0.18622814,0.18686274,
     #0.18749681,0.18813036,0.18876338,0.18939586,0.19002780,0.19065921,
     #0.19129008,0.19192041,0.19255023,0.19317949,0.19380820,0.19443639,
     #0.19506402,0.19569112,0.19631767,0.19694369,0.19756915,0.19819407,
     #0.19881846,0.19944228,0.20006557,0.20068832,0.20131050,0.20193215,
     #0.20255326,0.20317380,0.20379381,0.20441326,0.20503217,0.20565054,
     #0.20626834,0.20688561,0.20750232,0.20811850,0.20873411,0.20934919,
     #0.20996371,0.21057770,0.21119112,0.21180400,0.21241634,0.21302813,
     #0.21363938,0.21425007,0.21486023,0.21546984,0.21607891,0.21668743,
     #0.21729541,0.21790284,0.21850973,0.21911608,0.21972190,0.22032717,
     #0.22093190,0.22153610,0.22213975,0.22274287,0.22334546,0.22394751,
     #0.22454903,0.22515000,0.22575045,0.22635037,0.22694975,0.22754861,
     #0.22814694,0.22874475,0.22934201,0.22993876,0.23053499,0.23113069,
     #0.23172587,0.23232053,0.23291467,0.23350829,0.23410138,0.23469397,
     #0.23528604,0.23587760,0.23646864,0.23705918,0.23764920,0.23823872,
     #0.23882772,0.23941623,0.24000423,0.24059172,0.24117871,0.24176520,
     #0.24235119,0.24293669,0.24352169,0.24410619,0.24469021,0.24527372,
     #0.24585675,0.24643929,0.24702133,0.24760288,0.24818397,0.24876456,
     #0.24934468,0.24992432,0.25050348,0.25108215,0.25166038,0.25223809,
     #0.25281537,0.25339216,0.25396848,0.25454432,0.25511971,0.25569463,
     #0.25626910,0.25684309,0.25741664,0.25798970,0.25856236,0.25913450,
     #0.25970623,0.26027751,0.26084831,0.26141870,0.26198861,0.26255810,
     #0.26312715,0.26369572,0.26426390,0.26483160,0.26539889,0.26596573,
     #0.26653215,0.26709813,0.26766369,0.26822880,0.26879352,0.26935777,
     #0.26992163,0.27048507,0.27104807,0.27161068,0.27217284,0.27273461,
     #0.27329597,0.27385691,0.27441743,0.27497754,0.27553725,0.27609658,
     #0.27665550,0.27721399,0.27777210,0.27832982,0.27888712,0.27944404,
     #0.28000057,0.28055671,0.28111246,0.28166780,0.28222278,0.28277737,
     #0.28333157,0.28388542,0.28443885,0.28499192,0.28554460,0.28609693,
     #0.28664887,0.28720045,0.28775167,0.28830251,0.28885296,0.28940308,
     #0.28995281,0.29050222,0.29105124,0.29159990,0.29214820,0.29269618,
     #0.29324377,0.29379103,0.29433793,0.29488447,0.29543069,0.29597655,
     #0.29652208,0.29706725,0.29761207,0.29815659,0.29870075,0.29924458,
     #0.29978806,0.30033123,0.30087405,0.30141655,0.30195874,0.30250058,
     #0.30304211,0.30358329,0.30412418,0.30466476,0.30520499,0.30574492,
     #0.30628452,0.30682382,0.30736279,0.30790147,0.30843985,0.30897790,
     #0.30951566,0.31005311,0.31059024,0.31112710,0.31166363,0.31219989,
     #0.31273583,0.31327149,0.31380683,0.31434190,0.31487668,0.31541115,
     #0.31594536,0.31647927,0.31701291,0.31754625,0.31807929,0.31861207,
     #0.31914458,0.31967679,0.32020873,0.32074040,0.32127178,0.32180291,
     #0.32233375,0.32286432,0.32339466,0.32392469,0.32445446,0.32498398,
     #0.32551324,0.32604221,0.32657096,0.32709941,0.32762763,0.32815558,
     #0.32868329,0.32921073,0.32973793,0.33026487,0.33079156,0.33131802,
     #0.33184421,0.33237016,0.33289587,0.33342135,0.33394656,0.33447155,
     #0.33499628,0.33552080,0.33604506,0.33656910,0.33709288,0.33761644,
     #0.33813980,0.33866289,0.33918577,0.33970842,0.34023082,0.34075302,
     #0.34127498,0.34179673,0.34231827,0.34283957,0.34336063,0.34388149,
     #0.34440213,0.34492257,0.34544280,0.34596279,0.34648257,0.34700215,
     #0.34752151,0.34804067,0.34855962,0.34907836,0.34959689,0.35011524,
     #0.35063335,0.35115129,0.35166901,0.35218653,0.35270384,0.35322097,
     #0.35373789,0.35425463,0.35477117,0.35528749,0.35580364,0.35631961,
     #0.35683537,0.35735095,0.35786635,0.35838154,0.35889655,0.35941139,
     #0.35992602,0.36044049,0.36095476,0.36146885,0.36198279,0.36249653,
     #0.36301008,0.36352345,0.36403665,0.36454970,0.36506253,0.36557522,
     #0.36608773,0.36660007,0.36711225,0.36762422,0.36813605,0.36864772,
     #0.36915922,0.36967054,0.37018171,0.37069270,0.37120354,0.37171420,
     #0.37222472,0.37273508,0.37324527,0.37375531,0.37426516,0.37477490,
     #0.37528446,0.37579387,0.37630311,0.37681222,0.37732115,0.37782997,
     #0.37833861,0.37884709,0.37935546,0.37986365,0.38037157,0.38087934,
     #0.38138697,0.38189444,0.38240176,0.38290912,0.38341632,0.38392338,
     #0.38443032,0.38493720,0.38544393,0.38595054,0.38645700,0.38696334,
     #0.38746941,0.38797536,0.38848117,0.38898683,0.38949239,0.38999778,
     #0.39050305,0.39100820,0.39151323,0.39201811,0.39252287/
c
      data tpvel /
     #9.17879105,7.61300802,6.79586744,6.23279190,5.82625723,5.52815056,
     #5.30436897,5.13152695,4.99460268,4.88413000,4.79397631,4.71993494,
     #4.65893507,4.60861206,4.56708860,4.53283310,4.50458956,4.48132133,
     #4.46217108,4.44642878,4.43350840,4.42292309,4.41427135,4.40722036,
     #4.40149355,4.39686537,4.39314318,4.39017200,4.38781977,4.38597870,
     #4.38455772,4.38348246,4.38268995,4.38212681,4.38175201,4.38152790,
     #4.38142490,4.38141823,4.38148689,4.38161278,4.38178253,4.38198423,
     #4.38220692,4.38244295,4.38268566,4.38292980,4.38317013,4.38340330,
     #4.38362646,4.38383722,4.38403368,4.38421440,4.38437796,4.38452339,
     #4.38465071,4.38475895,4.38484764,4.38491726,4.38496685,4.38499737,
     #4.38500786,4.38499975,4.38497210,4.38492537,4.38486052,4.38477707,
     #4.38467503,4.38455534,4.38441801,4.38426352,4.38409138,4.38390303,
     #4.38369751,4.38347578,4.38323832,4.38298416,4.38271475,4.38243008,
     #4.38212967,4.38181448,4.38148451,4.38113928,4.38077974,4.38040543,
     #4.38001728,4.37961435,4.37919807,4.37876701,4.37832260,4.37786436,
     #4.37739229,4.37690687,4.37640762,4.37589502,4.37536860,4.37482929,
     #4.37427664,4.37371063,4.37313128,4.37253857,4.37193251,4.37131357,
     #4.37068129,4.37003613,4.36937761,4.36870575,4.36802053,4.36732244,
     #4.36661053,4.36588621,4.36514807,4.36439657,4.36363173,4.36285353,
     #4.36206198,4.36125660,4.36043787,4.35960531,4.35875988,4.35789967,
     #4.35702658,4.35613966,4.35523844,4.35432339,4.35339451,4.35245132,
     #4.35149384,4.35052252,4.34953642,4.34853601,4.34752131,4.34649181,
     #4.34544754,4.34438896,4.34331512,4.34222651,4.34112310,4.34000397,
     #4.33887005,4.33772087,4.33655596,4.33537579,4.33417940,4.33296824,
     #4.33174038,4.33049870,4.32923985,4.32796431,4.32667351,4.32536554,
     #4.32404280,4.32270384,4.32134485,4.31997299,4.31858015,4.31717348,
     #4.31574821,4.31430721,4.31284904,4.31137228,4.30987501,4.30836296,
     #4.30682945,4.30527735,4.30371189,4.30212307,4.30052185,4.29890203,
     #4.29726362,4.29560757,4.29393244,4.29223919,4.29052734,4.28879690,
     #4.28704786,4.28527975,4.28349304,4.28168678,4.27986193,4.27801800,
     #4.27615452,4.27427244,4.27237034,4.27044964,4.26850891,4.26654911,
     #4.26456976,4.26257086,4.26055193,4.25851393,4.25645638,4.25437880,
     #4.25228167,4.25016451,4.24802780,4.24587154,4.24369526,4.24149942,
     #4.23928404,4.23704863,4.23479319,4.23251820,4.23022366,4.22790909,
     #4.22557497,4.22322130,4.22084808,4.21845484,4.21604252,4.21361065,
     #4.21115923,4.20868826,4.20619822,4.20368862,4.20115995,4.19861221,
     #4.19604540,4.19345951,4.19085503,4.18823195,4.18558979,4.18292904,
     #4.18024969,4.17755270,4.17483759,4.17210436,4.16935349,4.16658449,
     #4.16379833,4.16099405,4.15817308,4.15533447,4.15247917,4.14960670,
     #4.14671803,4.14381266,4.14089108,4.13795328,4.13499975,4.13203001,
     #4.12904501,4.12604475,4.12302923,4.11999893,4.11695385,4.11389446,
     #4.11082077,4.10773277,4.10463142,4.10151577,4.09838724,4.09524584,
     #4.09209108,4.08892393,4.08574438,4.08255243,4.07934856,4.07613325,
     #4.07290697,4.06966877,4.06642008,4.06316090,4.05989122,4.05661154,
     #4.05332184,4.05002260,4.04671431,4.04339695,4.04007053,4.03673553,
     #4.03339291,4.03004169,4.02668333,4.02331686,4.01994371,4.01656389,
     #4.01317739,4.00978422,4.00638533,4.00298023,3.99956989,3.99615407,
     #3.99273348,3.98930812,3.98587823,3.98244429,3.97900629,3.97556448,
     #3.97211933,3.96867108,3.96521997,3.96176624,3.95831013,3.95485163,
     #3.95139146,3.94792962,3.94446611,3.94100189,3.93753648,3.93407035,
     #3.93060374,3.92713714,3.92367053,3.92020392,3.91673803,3.91327286,
     #3.90980840,3.90634513,3.90288329,3.89942312,3.89596462,3.89250803,
     #3.88905382,3.88560200,3.88215256,3.87870598,3.87526250,3.87182212,
     #3.86838508,3.86495161,3.86152196,3.85809612,3.85467458,3.85125709,
     #3.84784412,3.84443569,3.84103203,3.83763337,3.83423972,3.83085132,
     #3.82746840,3.82409072,3.82071900,3.81735301,3.81399298,3.81063914,
     #3.80729127,3.80395031,3.80061555,3.79728770,3.79396653,3.79065228,
     #3.78734517,3.78404522,3.78075242,3.77746701,3.77418900,3.77091885,
     #3.76765609,3.76440144,3.76115465,3.75791574,3.75468493,3.75146246,
     #3.74824810,3.74504209,3.74184465,3.73865581,3.73547578,3.73230410,
     #3.72914147,3.72598767,3.72284269,3.71970677,3.71657991,3.71346235,
     #3.71035409,3.70725489,3.70416498,3.70108438,3.69801354,3.69495201,
     #3.69190025,3.68885827,3.68582559,3.68280292,3.67979002,3.67678714,
     #3.67379379,3.67081046,3.66783714,3.66487408,3.66192079,3.65897751,
     #3.65604472,3.65312171,3.65020895,3.64730668,3.64441442,3.64153242,
     #3.63866091,3.63579965,3.63294888,3.63010836,3.62727833,3.62445855,
     #3.62164950,3.61885071,3.61606264,3.61328483,3.61051774,3.60776114,
     #3.60501504,3.60227942,3.59955454,3.59684014,3.59413648,3.59144330,
     #3.58876085,3.58608890,3.58342767,3.58077717,3.57813716,3.57550764,
     #3.57288909,3.57028103,3.56768370,3.56509686,3.56252074,3.55995536,
     #3.55740047,3.55485630,3.55232286,3.54979992,3.54728770,3.54478598,
     #3.54229498,3.53981447,3.53734469,3.53488541,3.53243685,3.52999854,
     #3.52757120,3.52515411,3.52274776,3.52035189,3.51796627,3.51559138,
     #3.51322699,3.51087308,3.50852966,3.50619674,3.50387406,3.50156188,
     #3.49925995,3.49696875,3.49468756,3.49241686,3.49015641,3.48790646,
     #3.48566651,3.48343706,3.48121786,3.47900867,3.47680974,3.47462106,
     #3.47244263,3.47027421,3.46811604,3.46596766,3.46382976,3.46170163,
     #3.45958352,3.45747542,3.45537734,3.45328927,3.45121121,3.44914293,
     #3.44708443,3.44503593,3.44299698,3.44096804,3.43894887,3.43693948,
     #3.43493962,3.43294954,3.43096924,3.42899823,3.42703724,3.42508554,
     #3.42314363,3.42121100,3.41928816,3.41737461,3.41547036,3.41357565,
     #3.41169047,3.40981436,3.40794778,3.40609026,3.40424228,3.40240335,
     #3.40057349,3.39875293,3.39694166,3.39513922,3.39334607,3.39156175,
     #3.38978672,3.38802052,3.38626313,3.38451481,3.38277531,3.38104486,
     #3.37932301,3.37760997,3.37590575,3.37421036,3.37252355,3.37084532,
     #3.36917591,3.36751485,3.36586261,3.36421871,3.36258340,3.36095667,
     #3.35933828,3.35772824,3.35612655,3.35453320,3.35294843,3.35137153,
     #3.34980321,3.34824300,3.34669089,3.34514689,3.34361100,3.34208345,
     #3.34056377,3.33905196,3.33754826,3.33605266,3.33456469,3.33308482,
     #3.33161283,3.33014846,3.32869196,3.32724333,3.32580233,3.32436895,
     #3.32294321,3.32152534,3.32011485,3.31871200,3.31731653,3.31592870,
     #3.31454825,3.31317520,3.31180978,3.31045151,3.30910063,3.30775714,
     #3.30642080,3.30509162,3.30376983,3.30245519,3.30114746,3.29984713,
     #3.29855371,3.29726744,3.29598808,3.29471564,3.29345036,3.29219174,
     #3.29094028,3.28969550,3.28845763,3.28722644,3.28600216,3.28478456,
     #3.28357363,3.28236938,3.28117180,3.27998066,3.27879620,3.27761841,
     #3.27644706,3.27528214,3.27412367,3.27297163,3.27182603,3.27068663,
     #3.26955366,3.26842713,3.26730680,3.26619267,3.26508474,3.26398301,
     #3.26288748,3.26179790,3.26071453,3.25963736,3.25856590,3.25750065,
     #3.25644135,3.25538802,3.25434065,3.25329900,3.25226355,3.25123358,
     #3.25020957,3.24919128,3.24817872,3.24717188,3.24617076,3.24517536,
     #3.24418545,3.24320126,3.24222279,3.24124956,3.24028206,3.23932004,
     #3.23836327,3.23741221,3.23646641,3.23552608,3.23459101,3.23366094,
     #3.23273635,3.23181653,3.23090172,3.22999334,3.22909021,3.22819233,
     #3.22729945,3.22641158,3.22552848,3.22465086,3.22377849,3.22291112,
     #3.22204828,3.22119045,3.22033763,3.21948957,3.21864653,3.21780825,
     #3.21697474,3.21614623,3.21532249,3.21450329,3.21368909/
c
      data somega /
     #0.00194327,0.00294407,0.00406573,0.00528063,0.00652341,0.00773970,
     #0.00888144,0.00991674,0.01084782,0.01170198,0.01250596,0.01327604,
     #0.01402037,0.01474283,0.01544548,0.01612975,0.01679691,0.01744832,
     #0.01808540,0.01870965,0.01932258,0.01992564,0.02052024,0.02110765,
     #0.02168906,0.02226550,0.02283790,0.02340707,0.02397371,0.02453840,
     #0.02510166,0.02566391,0.02622550,0.02678673,0.02734784,0.02790903,
     #0.02847045,0.02903225,0.02959453,0.03015735,0.03072079,0.03128488,
     #0.03184966,0.03241515,0.03298136,0.03354830,0.03411597,0.03468436,
     #0.03525347,0.03582329,0.03639381,0.03696500,0.03753686,0.03810938,
     #0.03868253,0.03925630,0.03983068,0.04040565,0.04098120,0.04155730,
     #0.04213395,0.04271113,0.04328883,0.04386703,0.04444573,0.04502491,
     #0.04560457,0.04618468,0.04676525,0.04734625,0.04792769,0.04850956,
     #0.04909183,0.04967453,0.05025762,0.05084110,0.05142498,0.05200923,
     #0.05259386,0.05317887,0.05376424,0.05434996,0.05493604,0.05552248,
     #0.05610926,0.05669638,0.05728384,0.05787164,0.05845976,0.05904821,
     #0.05963698,0.06022607,0.06081548,0.06140519,0.06199522,0.06258555,
     #0.06317618,0.06376711,0.06435833,0.06494985,0.06554165,0.06613375,
     #0.06672611,0.06731876,0.06791168,0.06850488,0.06909835,0.06969208,
     #0.07028607,0.07088032,0.07147484,0.07206959,0.07266460,0.07325986,
     #0.07385536,0.07445109,0.07504706,0.07564326,0.07623969,0.07683635,
     #0.07743323,0.07803032,0.07862763,0.07922515,0.07982288,0.08042081,
     #0.08101894,0.08161727,0.08221579,0.08281449,0.08341339,0.08401247,
     #0.08461173,0.08521116,0.08581076,0.08641053,0.08701047,0.08761057,
     #0.08821081,0.08881123,0.08941178,0.09001248,0.09061333,0.09121431,
     #0.09181543,0.09241669,0.09301807,0.09361958,0.09422120,0.09482295,
     #0.09542482,0.09602679,0.09662887,0.09723106,0.09783335,0.09843574,
     #0.09903822,0.09964079,0.10024345,0.10084620,0.10144903,0.10205194,
     #0.10265492,0.10325797,0.10386109,0.10446429,0.10506754,0.10567085,
     #0.10627422,0.10687765,0.10748113,0.10808466,0.10868822,0.10929184,
     #0.10989549,0.11049918,0.11110290,0.11170666,0.11231044,0.11291425,
     #0.11351808,0.11412194,0.11472581,0.11532969,0.11593359,0.11653750,
     #0.11714142,0.11774534,0.11834926,0.11895319,0.11955711,0.12016103,
     #0.12076494,0.12136885,0.12197273,0.12257661,0.12318047,0.12378432,
     #0.12438814,0.12499194,0.12559570,0.12619945,0.12680317,0.12740687,
     #0.12801051,0.12861413,0.12921771,0.12982126,0.13042475,0.13102821,
     #0.13163161,0.13223498,0.13283828,0.13344154,0.13404475,0.13464791,
     #0.13525100,0.13585404,0.13645701,0.13705993,0.13766277,0.13826555,
     #0.13886827,0.13947092,0.14007349,0.14067599,0.14127842,0.14188077,
     #0.14248304,0.14308523,0.14368734,0.14428937,0.14489131,0.14549316,
     #0.14609493,0.14669661,0.14729820,0.14789970,0.14850111,0.14910242,
     #0.14970362,0.15030473,0.15090574,0.15150666,0.15210746,0.15270817,
     #0.15330876,0.15390925,0.15450963,0.15510990,0.15571006,0.15631010,
     #0.15691003,0.15750985,0.15810955,0.15870912,0.15930858,0.15990791,
     #0.16050713,0.16110621,0.16170518,0.16230401,0.16290273,0.16350131,
     #0.16409975,0.16469806,0.16529626,0.16589430,0.16649221,0.16708998,
     #0.16768761,0.16828512,0.16888246,0.16947967,0.17007674,0.17067367,
     #0.17127044,0.17186707,0.17246355,0.17305988,0.17365605,0.17425208,
     #0.17484795,0.17544366,0.17603922,0.17663462,0.17722987,0.17782494,
     #0.17841986,0.17901462,0.17960921,0.18020365,0.18079790,0.18139200,
     #0.18198593,0.18257968,0.18317327,0.18376668,0.18435992,0.18495299,
     #0.18554588,0.18613859,0.18673111,0.18732348,0.18791565,0.18850763,
     #0.18909945,0.18969107,0.19028251,0.19087376,0.19146483,0.19205569,
     #0.19264637,0.19323687,0.19382717,0.19441727,0.19500718,0.19559689,
     #0.19618641,0.19677573,0.19736485,0.19795376,0.19854248,0.19913098,
     #0.19971929,0.20030740,0.20089529,0.20148297,0.20207044,0.20265771,
     #0.20324476,0.20383161,0.20441823,0.20500465,0.20559084,0.20617682,
     #0.20676258,0.20734811,0.20793343,0.20851852,0.20910339,0.20968804,
     #0.21027246,0.21085666,0.21144062,0.21202436,0.21260786,0.21319114,
     #0.21377417,0.21435697,0.21493955,0.21552187,0.21610397,0.21668583,
     #0.21726744,0.21784882,0.21842995,0.21901083,0.21959148,0.22017187,
     #0.22075202,0.22133192,0.22191158,0.22249097,0.22307011,0.22364901,
     #0.22422765,0.22480604,0.22538416,0.22596203,0.22653964,0.22711699,
     #0.22769408,0.22827090,0.22884746,0.22942375,0.22999978,0.23057553,
     #0.23115103,0.23172624,0.23230121,0.23287587,0.23345028,0.23402441,
     #0.23459826,0.23517184,0.23574513,0.23631814,0.23689088,0.23746334,
     #0.23803550,0.23860739,0.23917899,0.23975030,0.24032132,0.24089207,
     #0.24146250,0.24203266,0.24260251,0.24317208,0.24374135,0.24431032,
     #0.24487899,0.24544737,0.24601544,0.24658322,0.24715069,0.24771787,
     #0.24828473,0.24885128,0.24941754,0.24998349,0.25054911,0.25111443,
     #0.25167945,0.25224414,0.25280854,0.25337261,0.25393635,0.25449979,
     #0.25506291,0.25562569,0.25618815,0.25675032,0.25731215,0.25787362,
     #0.25843480,0.25899565,0.25955617,0.26011637,0.26067623,0.26123574,
     #0.26179495,0.26235381,0.26291233,0.26347050,0.26402837,0.26458588,
     #0.26514304,0.26569986,0.26625636,0.26681250,0.26736829,0.26792374,
     #0.26847884,0.26903361,0.26958802,0.27014208,0.27069578,0.27124912,
     #0.27180213,0.27235475,0.27290705,0.27345899,0.27401054,0.27456176,
     #0.27511263,0.27566311,0.27621323,0.27676299,0.27731240,0.27786142,
     #0.27841008,0.27895838,0.27950633,0.28005388,0.28060105,0.28114787,
     #0.28169429,0.28224036,0.28278604,0.28333136,0.28387630,0.28442085,
     #0.28496504,0.28550881,0.28605223,0.28659526,0.28713790,0.28768015,
     #0.28822201,0.28876352,0.28930461,0.28984532,0.29038563,0.29092556,
     #0.29146507,0.29200423,0.29254296,0.29308131,0.29361928,0.29415682,
     #0.29469398,0.29523075,0.29576710,0.29630306,0.29683861,0.29737374,
     #0.29790848,0.29844284,0.29897678,0.29951030,0.30004343,0.30057612,
     #0.30110845,0.30164033,0.30217180,0.30270287,0.30323353,0.30376378,
     #0.30429360,0.30482301,0.30535200,0.30588058,0.30640873,0.30693647,
     #0.30746377,0.30799067,0.30851713,0.30904320,0.30956882,0.31009403,
     #0.31061879,0.31114316,0.31166708,0.31219056,0.31271365,0.31323627,
     #0.31375849,0.31428027,0.31480160,0.31532252,0.31584302,0.31636307,
     #0.31688267,0.31740186,0.31792060,0.31843892,0.31895679,0.31947422,
     #0.31999120,0.32050776,0.32102388,0.32153955,0.32205477,0.32256958,
     #0.32308391,0.32359782,0.32411128,0.32462430,0.32513687,0.32564899,
     #0.32616070,0.32667193,0.32718271,0.32769305,0.32820293,0.32871237,
     #0.32922137,0.32972991,0.33023801,0.33074567,0.33125284,0.33175957,
     #0.33226585,0.33277169,0.33327708,0.33378202,0.33428648,0.33479050,
     #0.33529404,0.33579716,0.33629981,0.33680201,0.33730373,0.33780500,
     #0.33830583,0.33880618,0.33930609,0.33980554,0.34030452,0.34080303,
     #0.34130108,0.34179869,0.34229583,0.34279251,0.34328875,0.34378448,
     #0.34427980,0.34477463,0.34526899,0.34576291,0.34625635,0.34674934,
     #0.34724188,0.34773391,0.34822550,0.34871665,0.34920731,0.34969750,
     #0.35018724,0.35067654,0.35116532,0.35165367,0.35214156,0.35262895,
     #0.35311592,0.35360238,0.35408840,0.35457397,0.35505903,0.35554364,
     #0.35602781,0.35651150,0.35699472,0.35747746,0.35795975,0.35844156,
     #0.35892293,0.35940382,0.35988423,0.36036420,0.36084369,0.36132273,
     #0.36180127,0.36227936,0.36275700,0.36323416,0.36371085,0.36418709,
     #0.36466286,0.36513814,0.36561298,0.36608735,0.36656123,0.36703467,
     #0.36750764,0.36798015,0.36845219,0.36892378,0.36939490,0.36986554,
     #0.37033573,0.37080544,0.37127471,0.37174350,0.37221184,0.37267971,
     #0.37314710,0.37361404,0.37408054,0.37454656,0.37501213,0.37547722,
     #0.37594184,0.37640601,0.37686974,0.37733302,0.37779579,0.37825814,
     #0.37872002,0.37918144,0.37964240,0.38010290,0.38056293,0.38102254,
     #0.38148165,0.38194034,0.38239855,0.38285631,0.38331363,0.38377050,
     #0.38422689,0.38468283,0.38513833,0.38559335,0.38604796,0.38650209,
     #0.38695577,0.38740900,0.38786179,0.38831404,0.38876596,0.38921741,
     #0.38966838,0.39011893,0.39056900,0.39101866,0.39146784,0.39191660,
     #0.39236489/
c
      data spvel /
     #6.39109182,6.70532084,7.51172256,7.88421440,7.88004065,7.55123425,
     #6.94628572,6.23629856,5.64826536,5.25375748,4.99700928,4.81332636,
     #4.66454124,4.53290462,4.41150379,4.29837894,4.19354010,4.09755564,
     #4.01093674,3.93391252,3.86639047,3.80799985,3.75816417,3.71617436,
     #3.68125391,3.65261030,3.62946939,3.61110115,3.59683299,3.58605838,
     #3.57823658,3.57289243,3.56961179,3.56803775,3.56786323,3.56882691,
     #3.57070708,3.57331800,3.57650280,3.58013010,3.58409119,3.58829594,
     #3.59267044,3.59715414,3.60169816,3.60626268,3.61081648,3.61533427,
     #3.61979699,3.62418938,3.62850189,3.63272595,3.63685632,3.64088964,
     #3.64482474,3.64866138,3.65240026,3.65604329,3.65959239,3.66305065,
     #3.66642118,3.66970706,3.67291212,3.67603970,3.67909312,3.68207645,
     #3.68499327,3.68784618,3.69063902,3.69337511,3.69605756,3.69868898,
     #3.70127225,3.70380998,3.70630431,3.70875788,3.71117282,3.71355057,
     #3.71589375,3.71820331,3.72048140,3.72272921,3.72494817,3.72713947,
     #3.72930408,3.73144293,3.73355699,3.73564744,3.73771453,3.73975921,
     #3.74178171,3.74378276,3.74576283,3.74772191,3.74966073,3.75157928,
     #3.75347781,3.75535655,3.75721574,3.75905490,3.76087475,3.76267505,
     #3.76445556,3.76621652,3.76795793,3.76967955,3.77138138,3.77306318,
     #3.77472496,3.77636671,3.77798820,3.77958941,3.78117013,3.78273034,
     #3.78426957,3.78578806,3.78728557,3.78876185,3.79021692,3.79165053,
     #3.79306269,3.79445314,3.79582214,3.79716921,3.79849434,3.79979753,
     #3.80107880,3.80233788,3.80357480,3.80478930,3.80598187,3.80715203,
     #3.80830002,3.80942559,3.81052876,3.81160975,3.81266832,3.81370473,
     #3.81471848,3.81571031,3.81667995,3.81762743,3.81855297,3.81945634,
     #3.82033777,3.82119751,3.82203531,3.82285166,3.82364607,3.82441926,
     #3.82517099,3.82590127,3.82661057,3.82729888,3.82796621,3.82861257,
     #3.82923841,3.82984376,3.83042860,3.83099318,3.83153772,3.83206248,
     #3.83256721,3.83305264,3.83351827,3.83396482,3.83439207,3.83480024,
     #3.83518982,3.83556056,3.83591270,3.83624673,3.83656240,3.83685994,
     #3.83713984,3.83740187,3.83764648,3.83787346,3.83808351,3.83827639,
     #3.83845258,3.83861184,3.83875465,3.83888125,3.83899140,3.83908558,
     #3.83916378,3.83922625,3.83927321,3.83930469,3.83932066,3.83932185,
     #3.83930802,3.83927917,3.83923578,3.83917785,3.83910561,3.83901882,
     #3.83891821,3.83880353,3.83867526,3.83853292,3.83837700,3.83820772,
     #3.83802557,3.83782983,3.83762145,3.83739996,3.83716583,3.83691835,
     #3.83665895,3.83638716,3.83610320,3.83580685,3.83549857,3.83517814,
     #3.83484602,3.83450222,3.83414650,3.83377957,3.83340096,3.83301115,
     #3.83261013,3.83219790,3.83177471,3.83134031,3.83089542,3.83043957,
     #3.82997298,3.82949567,3.82900810,3.82851005,3.82800150,3.82748294,
     #3.82695389,3.82641482,3.82586575,3.82530665,3.82473755,3.82415891,
     #3.82357025,3.82297182,3.82236385,3.82174635,3.82111931,3.82048273,
     #3.81983685,3.81918144,3.81851673,3.81784296,3.81715989,3.81646776,
     #3.81576633,3.81505609,3.81433678,3.81360865,3.81287146,3.81212544,
     #3.81137061,3.81060719,3.80983472,3.80905390,3.80826426,3.80746603,
     #3.80665922,3.80584383,3.80501986,3.80418754,3.80334663,3.80249739,
     #3.80163980,3.80077386,3.79989958,3.79901695,3.79812622,3.79722714,
     #3.79631972,3.79540420,3.79448032,3.79354858,3.79260850,3.79166055,
     #3.79070425,3.78974009,3.78876781,3.78778744,3.78679919,3.78580284,
     #3.78479862,3.78378606,3.78276587,3.78173757,3.78070140,3.77965736,
     #3.77860546,3.77754569,3.77647805,3.77540255,3.77431893,3.77322793,
     #3.77212882,3.77102184,3.76990724,3.76878476,3.76765442,3.76651645,
     #3.76537037,3.76421690,3.76305556,3.76188636,3.76070929,3.75952482,
     #3.75833225,3.75713229,3.75592422,3.75470877,3.75348520,3.75225401,
     #3.75101495,3.74976850,3.74851418,3.74725223,3.74598265,3.74470520,
     #3.74342012,3.74212718,3.74082661,3.73951840,3.73820257,3.73687887,
     #3.73554754,3.73420858,3.73286176,3.73150730,3.73014522,3.72877526,
     #3.72739792,3.72601247,3.72461963,3.72321892,3.72181058,3.72039437,
     #3.71897078,3.71753931,3.71609998,3.71465302,3.71319842,3.71173620,
     #3.71026611,3.70878839,3.70730281,3.70580983,3.70430875,3.70280027,
     #3.70128393,3.69975972,3.69822788,3.69668818,3.69514084,3.69358587,
     #3.69202304,3.69045258,3.68887448,3.68728852,3.68569493,3.68409348,
     #3.68248439,3.68086743,3.67924285,3.67761040,3.67597032,3.67432237,
     #3.67266679,3.67100334,3.66933227,3.66765332,3.66596675,3.66427255,
     #3.66257048,3.66086054,3.65914297,3.65741754,3.65568447,3.65394378,
     #3.65219522,3.65043879,3.64867473,3.64690280,3.64512324,3.64333606,
     #3.64154100,3.63973808,3.63792753,3.63610935,3.63428330,3.63244963,
     #3.63060808,3.62875891,3.62690210,3.62503719,3.62316465,3.62128448,
     #3.61939669,3.61750102,3.61559796,3.61368704,3.61176825,3.60984206,
     #3.60790801,3.60596633,3.60401678,3.60205984,3.60009503,3.59812260,
     #3.59614253,3.59415507,3.59215975,3.59015679,3.58814621,3.58612800,
     #3.58410215,3.58206868,3.58002758,3.57797885,3.57592273,3.57385898,
     #3.57178760,3.56970882,3.56762242,3.56552839,3.56342697,3.56131792,
     #3.55920124,3.55707741,3.55494595,3.55280685,3.55066037,3.54850674,
     #3.54634523,3.54417658,3.54200053,3.53981709,3.53762603,3.53542781,
     #3.53322220,3.53100920,3.52878904,3.52656150,3.52432656,3.52208447,
     #3.51983500,3.51757836,3.51531458,3.51304364,3.51076531,3.50847983,
     #3.50618720,3.50388765,3.50158072,3.49926686,3.49694586,3.49461770,
     #3.49228263,3.48994064,3.48759151,3.48523545,3.48287225,3.48050237,
     #3.47812533,3.47574162,3.47335076,3.47095323,3.46854901,3.46613812,
     #3.46372032,3.46129560,3.45886445,3.45642638,3.45398188,3.45153046,
     #3.44907260,3.44660830,3.44413733,3.44165969,3.43917561,3.43668532,
     #3.43418837,3.43168497,3.42917514,3.42665911,3.42413664,3.42160797,
     #3.41907310,3.41653204,3.41398454,3.41143107,3.40887141,3.40630555,
     #3.40373373,3.40115595,3.39857221,3.39598227,3.39338660,3.39078498,
     #3.38817739,3.38556409,3.38294506,3.38032031,3.37768960,3.37505341,
     #3.37241173,3.36976409,3.36711121,3.36445260,3.36178875,3.35911918,
     #3.35644436,3.35376406,3.35107851,3.34838772,3.34569168,3.34299040,
     #3.34028411,3.33757257,3.33485603,3.33213449,3.32940769,3.32667613,
     #3.32393980,3.32119846,3.31845236,3.31570148,3.31294584,3.31018567,
     #3.30742073,3.30465150,3.30187750,3.29909897,3.29631591,3.29352856,
     #3.29073691,3.28794074,3.28514051,3.28233624,3.27952766,3.27671480,
     #3.27389812,3.27107739,3.26825261,3.26542401,3.26259160,3.25975513,
     #3.25691509,3.25407124,3.25122380,3.24837279,3.24551821,3.24266028,
     #3.23979855,3.23693371,3.23406553,3.23119402,3.22831917,3.22544122,
     #3.22255993,3.21967578,3.21678877,3.21389842,3.21100545,3.20810938,
     #3.20521069,3.20230913,3.19940495,3.19649816,3.19358873,3.19067669,
     #3.18776226,3.18484521,3.18192601,3.17900443,3.17608070,3.17315459,
     #3.17022610,3.16729617,3.16436362,3.16142941,3.15849304,3.15555501,
     #3.15261531,3.14967370,3.14673042,3.14378548,3.14083910,3.13789105,
     #3.13494158,3.13199067,3.12903857,3.12608504,3.12313032,3.12017441,
     #3.11721730,3.11425924,3.11130023,3.10834002,3.10537910,3.10241723,
     #3.09945488,3.09649158,3.09352779,3.09056330,3.08759832,3.08463264,
     #3.08166671,3.07870030,3.07573366,3.07276678,3.06979966,3.06683254,
     #3.06386518,3.06089783,3.05793047,3.05496335,3.05199623,3.04902935,
     #3.04606271,3.04309654,3.04013062,3.03716493,3.03419995,3.03123546,
     #3.02827168,3.02530837,3.02234578,3.01938391,3.01642299,3.01346302,
     #3.01050377,3.00754547,3.00458837,3.00163221,2.99867725,2.99572349,
     #2.99277115,2.98982000,2.98687005,2.98392177,2.98097467,2.97802925,
     #2.97508550,2.97214317,2.96920252,2.96626377,2.96332669,2.96039128,
     #2.95745802,2.95452666,2.95159721,2.94866967,2.94574428,2.94282103,
     #2.93990016,2.93698120,2.93406463,2.93115044,2.92823863,2.92532921,
     #2.92242241,2.91951799,2.91661620,2.91371703,2.91082048,2.90792656,
     #2.90503550,2.90214729,2.89926195,2.89637947,2.89349985,2.89062333,
     #2.88774967,2.88487911,2.88201189,2.87914634,2.87628627,2.87342858,
     #2.87057400,2.86772275,2.86487508,2.86203074,2.85918975,2.85635233,
     #2.85351849/
c
      if(.not.splined) then
        if(iprtlv.gt.1) then
        write(6,"('there are ',i5,' toroidal modes in spline')") ntspl
        write(6,"('there are ',i5,' spheroidal modes in spline')") nsspl
        endif
        call rspln(1,ntspl,tomega,tpvel,tcube,work)
        call rspln(1,nsspl,somega,spvel,scube,work)
        splined=.true.
      endif
c
      if(modetype.eq.1) then
        pvel=rsple(1,ntspl,tomega,tpvel,tcube,omega)
      else
        pvel=rsple(1,nsspl,somega,spvel,scube,omega)
      endif
      premgegrvelo=pvel
      return
      end


      subroutine vbspl(x,np,xarr,splcon,splcond)
c
c---- this subroutine returns the spline contributions at a particular value of x
c
      dimension xarr(np)
      dimension splcon(np)
      dimension splcond(np)
c
c---- iflag=1 ==>> second derivative is 0 at end points
c---- iflag=0 ==>> first derivative is 0 at end points
c
      iflag=1
c
c---- first, find out within which interval x falls
c
      interval=0
      ik=1
      do while(interval.eq.0.and.ik.lt.np)
        ik=ik+1
        if(x.ge.xarr(ik-1).and.x.le.xarr(ik)) interval=ik-1
      enddo
      if(x.gt.xarr(np)) then
        interval=np
      endif
c
      if(interval.eq.0) then
cc        write(6,"('low value:',2f10.3)") x,xarr(1)
      else if(interval.gt.0.and.interval.lt.np) then
cc        write(6,"('bracket:',i5,3f10.3)") interval,xarr(interval),x,xarr(interval+1)
      else
cc        write(6,"('high value:',2f10.3)") xarr(np),x
      endif
c
      do ib=1,np
        val=0.
        vald=0.
        if(ib.eq.1) then
c
          r1=(x-xarr(1))/(xarr(2)-xarr(1))
          r2=(xarr(3)-x)/(xarr(3)-xarr(1))
          r4=(xarr(2)-x)/(xarr(2)-xarr(1))
          r5=(x-xarr(1))/(xarr(2)-xarr(1))
          r6=(xarr(3)-x)/(xarr(3)-xarr(1))
         r10=(xarr(2)-x)/(xarr(2)-xarr(1))
         r11=(x-xarr(1))  /(xarr(2)-xarr(1))
         r12=(xarr(3)-x)/(xarr(3)-xarr(2))
         r13=(xarr(2)-x)/(xarr(2)-xarr(1))
c
          r1d=1./(xarr(2)-xarr(1))
          r2d=-1./(xarr(3)-xarr(1))
          r4d=-1./(xarr(2)-xarr(1))
          r5d=1./(xarr(2)-xarr(1))
          r6d=-1./(xarr(3)-xarr(1))
         r10d=-1./(xarr(2)-xarr(1))
         r11d=1./(xarr(2)-xarr(1))
         r12d=-1./(xarr(3)-xarr(2))
         r13d=-1./(xarr(2)-xarr(1))
c

          if(interval.eq.ib.or.interval.eq.0) then
           if(iflag.eq.0) then 
             val=r1*r4*r10 + r2*r5*r10 + r2*r6*r11 +r13**3
             vald=r1d*r4*r10+r1*r4d*r10+r1*r4*r10d
             vald=vald+r2d*r5*r10+r2*r5d*r10+r2*r5*r10d
             vald=vald+r2d*r6*r11+r2*r6d*r11+r2*r6*r11d
             vald=vald+3.*r13d*r13**2
           else if(iflag.eq.1) then
             val=0.6667*(r1*r4*r10 + r2*r5*r10 + r2*r6*r11 + 1.5*r13**3)
             vald=r1d*r4*r10+r1*r4d*r10+r1*r4*r10d
             vald=vald+r2d*r5*r10+r2*r5d*r10+r2*r5*r10d
             vald=vald+r2d*r6*r11+r2*r6d*r11+r2*r6*r11d
             vald=vald+4.5*r13d*r13**2
             vald=0.6667*vald
           endif
          else if(interval.eq.ib+1) then
               if(iflag.eq.0) then
                 val=r2*r6*r12
                 vald=r2d*r6*r12+r2*r6d*r12+r2*r6*r12d
               else if(iflag.eq.1) then
                 val=0.6667*r2*r6*r12
                 vald=0.6667*(r2d*r6*r12+r2*r6d*r12+r2*r6*r12d)
               endif
          else
            val=0.
          endif
c
        else if(ib.eq.2) then
c
          rr1=(x-xarr(1))/(xarr(2)-xarr(1))
          rr2=(xarr(3)-x)/(xarr(3)-xarr(1))
          rr4=(xarr(2)-x)/(xarr(2)-xarr(1))
          rr5=(x-xarr(1))/(xarr(2)-xarr(1))
          rr6=(xarr(3)-x)/(xarr(3)-xarr(1))
         rr10=(xarr(2)-x)/(xarr(2)-xarr(1))
         rr11=(x-xarr(1))  /(xarr(2)-xarr(1))
         rr12=(xarr(3)-x)/(xarr(3)-xarr(2))
c
          rr1d=1./(xarr(2)-xarr(1))
          rr2d=-1./(xarr(3)-xarr(1))
          rr4d=-1./(xarr(2)-xarr(1))
          rr5d=1./(xarr(2)-xarr(1))
          rr6d=-1./(xarr(3)-xarr(1))
         rr10d=-1./(xarr(2)-xarr(1))
         rr11d=1./(xarr(2)-xarr(1))
         rr12d=-1./(xarr(3)-xarr(2))
c
          r1=(x-xarr(ib-1))/(xarr(ib+1)-xarr(ib-1))
          r2=(xarr(ib+2)-x)/(xarr(ib+2)-xarr(ib-1))
          r3=(x-xarr(ib-1))/(xarr(ib)-xarr(ib-1))
          r4=(xarr(ib+1)-x)/(xarr(ib+1)-xarr(ib-1))
          r5=(x-xarr(ib-1))/(xarr(ib+1)-xarr(ib-1))
          r6=(xarr(ib+2)-x)/(xarr(ib+2)-xarr(ib))
          r8=(xarr(ib)-x)/  (xarr(ib)-xarr(ib-1))
          r9=(x-xarr(ib-1))/(xarr(ib)-xarr(ib-1))
         r10=(xarr(ib+1)-x)/(xarr(ib+1)-xarr(ib))
         r11=(x-xarr(ib))  /(xarr(ib+1)-xarr(ib))
         r12=(xarr(ib+2)-x)/(xarr(ib+2)-xarr(ib+1))
c
          r1d=1./(xarr(ib+1)-xarr(ib-1))
          r2d=-1./(xarr(ib+2)-xarr(ib-1))
          r3d=1./(xarr(ib)-xarr(ib-1))
          r4d=-1./(xarr(ib+1)-xarr(ib-1))
          r5d=1./(xarr(ib+1)-xarr(ib-1))
          r6d=-1./(xarr(ib+2)-xarr(ib))
          r8d=-1./  (xarr(ib)-xarr(ib-1))
          r9d=1./(xarr(ib)-xarr(ib-1))
         r10d=-1./(xarr(ib+1)-xarr(ib))
         r11d=1./(xarr(ib+1)-xarr(ib))
         r12d=-1./(xarr(ib+2)-xarr(ib+1))
c
          if(interval.eq.ib-1.or.interval.eq.0) then
           val=r1*r3*r8 + r1*r4*r9 + r2*r5*r9
           vald=r1d*r3*r8+r1*r3d*r8+r1*r3*r8d
           vald=vald+r1d*r4*r9+r1*r4d*r9+r1*r4*r9d
           vald=vald+r2d*r5*r9+r2*r5d*r9+r2*r5*r9d
           if(iflag.eq.1) then
            val=val+0.3333*(rr1*rr4*rr10 + rr2*rr5*rr10 + rr2*rr6*rr11)
            vald=vald+0.3333*(rr1d*rr4*rr10+rr1*rr4d*rr10+rr1*rr4*rr10d)
            vald=vald+0.3333*(rr2d*rr5*rr10+rr2*rr5d*rr10+rr2*rr5*rr10d)
            vald=vald+0.3333*(rr2d*rr6*rr11+rr2*rr6d*rr11+rr2*rr6*rr11d)
           endif
          else if(interval.eq.ib) then
           val=r1*r4*r10 + r2*r5*r10 + r2*r6*r11
           vald=r1d*r4*r10+r1*r4d*r10+r1*r4*r10d
           vald=vald+r2d*r5*r10+r2*r5d*r10+r2*r5*r10d
           vald=vald+r2d*r6*r11+r2*r6d*r11+r2*r6*r11d
           if(iflag.eq.1) then
            val=val+0.3333*rr2*rr6*rr12
            vald=vald+0.3333*(rr2d*rr6*rr12+rr2*rr6d*rr12+rr2*rr6*rr12d)
           endif
          else if(interval.eq.ib+1) then
               val=r2*r6*r12
               vald=r2d*r6*r12+r2*r6d*r12+r2*r6*r12d
          else
               val=0.
          endif
        else if(ib.eq.np-1) then
c
          rr1=(x-xarr(np-2))/(xarr(np)-xarr(np-2))
          rr2=(xarr(np)-x)/(xarr(np)-xarr(np-1))
          rr3=(x-xarr(np-2))/(xarr(np)-xarr(np-2))
          rr4=(xarr(np)-x)/(xarr(np)-xarr(np-1))
          rr5=(x-xarr(np-1))/(xarr(np)-xarr(np-1))
          rr7=(x-xarr(np-2))/(xarr(np-1)-xarr(np-2))
          rr8=(xarr(np)-x)/  (xarr(np)-xarr(np-1))
          rr9=(x-xarr(np-1))/(xarr(np)-xarr(np-1))
c
          rr1d=1./(xarr(np)-xarr(np-2))
          rr2d=-1./(xarr(np)-xarr(np-1))
          rr3d=1./(xarr(np)-xarr(np-2))
          rr4d=-1./(xarr(np)-xarr(np-1))
          rr5d=1./(xarr(np)-xarr(np-1))
          rr7d=1./(xarr(np-1)-xarr(np-2))
          rr8d=-1./  (xarr(np)-xarr(np-1))
          rr9d=1./(xarr(np)-xarr(np-1))
c
          r1=(x-xarr(ib-2))/(xarr(ib+1)-xarr(ib-2))
          r2=(xarr(ib+1)-x)/(xarr(ib+1)-xarr(ib-1))
          r3=(x-xarr(ib-2))/(xarr(ib)-xarr(ib-2))
          r4=(xarr(ib+1)-x)/(xarr(ib+1)-xarr(ib-1))
          r5=(x-xarr(ib-1))/(xarr(ib+1)-xarr(ib-1))
          r6=(xarr(ib+1)-x)/(xarr(ib+1)-xarr(ib))
          r7=(x-xarr(ib-2))/(xarr(ib-1)-xarr(ib-2))
          r8=(xarr(ib)-x)/  (xarr(ib)-xarr(ib-1))
          r9=(x-xarr(ib-1))/(xarr(ib)-xarr(ib-1))
         r10=(xarr(ib+1)-x)/(xarr(ib+1)-xarr(ib))
         r11=(x-xarr(ib))  /(xarr(ib+1)-xarr(ib))
c
          r1d=1./(xarr(ib+1)-xarr(ib-2))
          r2d=-1./(xarr(ib+1)-xarr(ib-1))
          r3d=1./(xarr(ib)-xarr(ib-2))
          r4d=-1./(xarr(ib+1)-xarr(ib-1))
          r5d=1./(xarr(ib+1)-xarr(ib-1))
          r6d=-1./(xarr(ib+1)-xarr(ib))
          r7d=1./(xarr(ib-1)-xarr(ib-2))
          r8d=-1./(xarr(ib)-xarr(ib-1))
          r9d=1./(xarr(ib)-xarr(ib-1))
         r10d=-1./(xarr(ib+1)-xarr(ib))
         r11d=1./(xarr(ib+1)-xarr(ib))
c
          if(interval.eq.ib-2) then
               val=r1*r3*r7
               vald=r1d*r3*r7+r1*r3d*r7+r1*r3*r7d
          else if(interval.eq.ib-1) then
             val=r1*r3*r8 + r1*r4*r9 + r2*r5*r9
             vald=r1d*r3*r8+r1*r3d*r8+r1*r3*r8d
             vald=vald+r1d*r4*r9+r1*r4d*r9+r1*r4*r9d
             vald=vald+r2d*r5*r9+r2*r5d*r9+r2*r5*r9d
             if(iflag.eq.1) then
               val=val+0.3333*rr1*rr3*rr7
               vald=vald+0.3333*(rr1d*rr3*rr7+rr1*rr3d*rr7+rr1*rr3*rr7d)
             endif
          else if(interval.eq.ib.or.interval.eq.np) then
             val=r1*r4*r10 + r2*r5*r10 + r2*r6*r11
             vald=r1d*r4*r10+r1*r4d*r10+r1*r4*r10d
             vald=vald+r2d*r5*r10+r2*r5d*r10+r2*r5*r10d
             vald=vald+r2d*r6*r11+r2*r6d*r11+r2*r6*r11d
             if(iflag.eq.1) then
               val=val+0.3333*(rr1*rr3*rr8 + rr1*rr4*rr9 + rr2*rr5*rr9)
               vald=vald+0.3333*(rr1d*rr3*rr8+rr1*rr3d*rr8+rr1*rr3*rr8d)
               vald=vald+0.3333*(rr1d*rr4*rr9+rr1*rr4d*rr9+rr1*rr4*rr9d)
               vald=vald+0.3333*(rr2d*rr5*rr9+rr2*rr5d*rr9+rr2*rr5*rr9d)
             endif
          else
            val=0.
          endif
        else if(ib.eq.np) then
c
          r1=(x-xarr(np-2))/(xarr(np)-xarr(np-2))
          r2=(xarr(np)-x)/(xarr(np)-xarr(np-1))
          r3=(x-xarr(np-2))/(xarr(np)-xarr(np-2))
          r4=(xarr(np)-x)/(xarr(np)-xarr(np-1))
          r5=(x-xarr(np-1))/(xarr(np)-xarr(np-1))
          r7=(x-xarr(np-2))/(xarr(np-1)-xarr(np-2))
          r8=(xarr(np)-x)/  (xarr(np)-xarr(np-1))
          r9=(x-xarr(np-1))/(xarr(np)-xarr(np-1))
          r13=(x-xarr(np-1))/(xarr(np)-xarr(np-1))
c
          r1d=1./(xarr(np)-xarr(np-2))
          r2d=-1./(xarr(np)-xarr(np-1))
          r3d=1./(xarr(np)-xarr(np-2))
          r4d=-1./(xarr(np)-xarr(np-1))
          r5d=1./(xarr(np)-xarr(np-1))
          r7d=1./(xarr(np-1)-xarr(np-2))
          r8d=-1./  (xarr(np)-xarr(np-1))
          r9d=1./(xarr(np)-xarr(np-1))
          r13d=1./(xarr(np)-xarr(np-1))
c
          if(interval.eq.np-2) then
               if(iflag.eq.0) then
                 val=r1*r3*r7
                 vald=r1d*r3*r7+r1*r3d*r7+r1*r3*r7d
               else if(iflag.eq.1) then
                 val=0.6667*r1*r3*r7
                 vald=0.6667*(r1d*r3*r7+r1*r3d*r7+r1*r3*r7d)
               endif
          else if(interval.eq.np-1.or.interval.eq.np) then
               if(iflag.eq.0) then
                 val=r1*r3*r8 + r1*r4*r9 + r2*r5*r9 + r13**3
                 vald=r1d*r3*r8+r1*r3d*r8+r1*r3*r8d
                 vald=vald+r1d*r4*r9+r1*r4d*r9+r1*r4*r9d
                 vald=vald+r2d*r5*r9+r2*r5d*r9+r2*r5*r9d
                 vald=vald+3.*r13d*r13**2
               else if(iflag.eq.1) then
                val=0.6667*(r1*r3*r8 + r1*r4*r9 + r2*r5*r9 + 1.5*r13**3)
                 vald=r1d*r3*r8+r1*r3d*r8+r1*r3*r8d
                 vald=vald+r1d*r4*r9+r1*r4d*r9+r1*r4*r9d
                 vald=vald+r2d*r5*r9+r2*r5d*r9+r2*r5*r9d
                 vald=vald+4.5*r13d*r13**2
                 vald=0.6667*vald
               endif
          else
            val=0.
          endif
        else
c
          r1=(x-xarr(ib-2))/(xarr(ib+1)-xarr(ib-2))
          r2=(xarr(ib+2)-x)/(xarr(ib+2)-xarr(ib-1))
          r3=(x-xarr(ib-2))/(xarr(ib)-xarr(ib-2))
          r4=(xarr(ib+1)-x)/(xarr(ib+1)-xarr(ib-1))
          r5=(x-xarr(ib-1))/(xarr(ib+1)-xarr(ib-1))
          r6=(xarr(ib+2)-x)/(xarr(ib+2)-xarr(ib))
          r7=(x-xarr(ib-2))/(xarr(ib-1)-xarr(ib-2))
          r8=(xarr(ib)-x)/  (xarr(ib)-xarr(ib-1))
          r9=(x-xarr(ib-1))/(xarr(ib)-xarr(ib-1))
         r10=(xarr(ib+1)-x)/(xarr(ib+1)-xarr(ib))
         r11=(x-xarr(ib))  /(xarr(ib+1)-xarr(ib))
         r12=(xarr(ib+2)-x)/(xarr(ib+2)-xarr(ib+1))
c
          r1d=1./(xarr(ib+1)-xarr(ib-2))
          r2d=-1./(xarr(ib+2)-xarr(ib-1))
          r3d=1./(xarr(ib)-xarr(ib-2))
          r4d=-1./(xarr(ib+1)-xarr(ib-1))
          r5d=1./(xarr(ib+1)-xarr(ib-1))
          r6d=-1./(xarr(ib+2)-xarr(ib))
          r7d=1./(xarr(ib-1)-xarr(ib-2))
          r8d=-1./  (xarr(ib)-xarr(ib-1))
          r9d=1./(xarr(ib)-xarr(ib-1))
         r10d=-1./(xarr(ib+1)-xarr(ib))
         r11d=1./(xarr(ib+1)-xarr(ib))
         r12d=-1./(xarr(ib+2)-xarr(ib+1))
c
          if(interval.eq.ib-2) then
               val=r1*r3*r7
               vald=r1d*r3*r7+r1*r3d*r7+r1*r3*r7d
          else if(interval.eq.ib-1) then
               val=r1*r3*r8 + r1*r4*r9 + r2*r5*r9
               vald=r1d*r3*r8+r1*r3d*r8+r1*r3*r8d
               vald=vald+r1d*r4*r9+r1*r4d*r9+r1*r4*r9d
               vald=vald+r2d*r5*r9+r2*r5d*r9+r2*r5*r9d
          else if(interval.eq.ib) then
               val=r1*r4*r10 + r2*r5*r10 + r2*r6*r11
               vald=r1d*r4*r10+r1*r4d*r10+r1*r4*r10d
               vald=vald+r2d*r5*r10+r2*r5d*r10+r2*r5*r10d
               vald=vald+r2d*r6*r11+r2*r6d*r11+r2*r6*r11d
          else if(interval.eq.ib+1) then
               val=r2*r6*r12
               vald=r2d*r6*r12+r2*r6d*r12+r2*r6*r12d
          else
            val=0.
          endif
        endif
        splcon(ib)=val
        splcond(ib)=vald
      enddo
      return
      end

      SUBROUTINE RSPLN(I1,I2,X,Y,Q,F)
C
C C$C$C$C$C$ CALLS ONLY LIBRARY ROUTINES C$C$C$C$C$
C
C   SUBROUTINE RSPLN COMPUTES CUBIC SPLINE INTERPOLATION COEFFICIENTS
C   FOR Y(X) BETWEEN GRID POINTS I1 AND I2 SAVING THEM IN Q.  THE
C   INTERPOLATION IS CONTINUOUS WITH CONTINUOUS FIRST AND SECOND
C   DERIVITIVES.  IT AGREES EXACTLY WITH Y AT GRID POINTS AND WITH THE
C   THREE POINT FIRST DERIVITIVES AT BOTH END POINTS (I1 AND I2).
C   X MUST BE MONOTONIC BUT IF TWO SUCCESSIVE VALUES OF X ARE EQUAL
C   A DISCONTINUITY IS ASSUMED AND SEPERATE INTERPOLATION IS DONE ON
C   EACH STRICTLY MONOTONIC SEGMENT.  THE ARRAYS MUST BE DIMENSIONED AT
C   LEAST - X(I2), Y(I2), Q(3,I2), AND F(3,I2).  F IS WORKING STORAGE
C   FOR RSPLN.
C                                                     -RPB
C
      DIMENSION X(1),Y(1),Q(3,1),F(3,1),YY(3)
      EQUIVALENCE (YY(1),Y0)
      DATA SMALL/1.E-5/,YY/0.,0.,0./
      J1=I1+1
      Y0=0.
C   BAIL OUT IF THERE ARE LESS THAN TWO POINTS TOTAL.
      IF(I2-I1)13,17,8
 8    A0=X(J1-1)
C   SEARCH FOR DISCONTINUITIES.
      DO 3 I=J1,I2
      B0=A0
      A0=X(I)
      IF(ABS((A0-B0)/AMAX1(A0,B0)).LT.SMALL) GO TO 4
 3    CONTINUE
 17   J1=J1-1
      J2=I2-2
      GO TO 5
 4    J1=J1-1
      J2=I-3
C   SEE IF THERE ARE ENOUGH POINTS TO INTERPOLATE (AT LEAST THREE).
 5    IF(J2+1-J1)9,10,11
C   ONLY TWO POINTS.  USE LINEAR INTERPOLATION.
 10   J2=J2+2
      Y0=(Y(J2)-Y(J1))/(X(J2)-X(J1))
      DO 15 J=1,3
      Q(J,J1)=YY(J)
 15   Q(J,J2)=YY(J)
      GO TO 12
C   MORE THAN TWO POINTS.  DO SPLINE INTERPOLATION.
 11   A0=0.
      H=X(J1+1)-X(J1)
      H2=X(J1+2)-X(J1)
      Y0=H*H2*(H2-H)
      H=H*H
      H2=H2*H2
C   CALCULATE DERIVITIVE AT NEAR END.
      B0=(Y(J1)*(H-H2)+Y(J1+1)*H2-Y(J1+2)*H)/Y0
      B1=B0
C   EXPLICITLY REDUCE BANDED MATRIX TO AN UPPER BANDED MATRIX.
      DO 1 I=J1,J2
      H=X(I+1)-X(I)
      Y0=Y(I+1)-Y(I)
      H2=H*H
      HA=H-A0
      H2A=H-2.*A0
      H3A=2.*H-3.*A0
      H2B=H2*B0
      Q(1,I)=H2/HA
      Q(2,I)=-HA/(H2A*H2)
      Q(3,I)=-H*H2A/H3A
      F(1,I)=(Y0-H*B0)/(H*HA)
      F(2,I)=(H2B-Y0*(2.*H-A0))/(H*H2*H2A)
      F(3,I)=-(H2B-3.*Y0*HA)/(H*H3A)
      A0=Q(3,I)
 1    B0=F(3,I)
C   TAKE CARE OF LAST TWO ROWS.
      I=J2+1
      H=X(I+1)-X(I)
      Y0=Y(I+1)-Y(I)
      H2=H*H
      HA=H-A0
      H2A=H*HA
      H2B=H2*B0-Y0*(2.*H-A0)
      Q(1,I)=H2/HA
      F(1,I)=(Y0-H*B0)/H2A
      HA=X(J2)-X(I+1)
      Y0=-H*HA*(HA+H)
      HA=HA*HA
C   CALCULATE DERIVITIVE AT FAR END.
      Y0=(Y(I+1)*(H2-HA)+Y(I)*HA-Y(J2)*H2)/Y0
      Q(3,I)=(Y0*H2A+H2B)/(H*H2*(H-2.*A0))
      Q(2,I)=F(1,I)-Q(1,I)*Q(3,I)
C   SOLVE UPPER BANDED MATRIX BY REVERSE ITERATION.
      DO 2 J=J1,J2
      K=I-1
      Q(1,I)=F(3,K)-Q(3,K)*Q(2,I)
      Q(3,K)=F(2,K)-Q(2,K)*Q(1,I)
      Q(2,K)=F(1,K)-Q(1,K)*Q(3,K)
 2    I=K
      Q(1,I)=B1
C   FILL IN THE LAST POINT WITH A LINEAR EXTRAPOLATION.
 9    J2=J2+2
      DO 14 J=1,3
 14   Q(J,J2)=YY(J)
C   SEE IF THIS DISCONTINUITY IS THE LAST.
 12   IF(J2-I2)6,13,13
C   NO.  GO BACK FOR MORE.
 6    J1=J2+2
      IF(J1-I2)8,8,7
C   THERE IS ONLY ONE POINT LEFT AFTER THE LATEST DISCONTINUITY.
 7    DO 16 J=1,3
 16   Q(J,I2)=YY(J)
C   FINI.
 13   RETURN
      END
 
      FUNCTION RSPLE(I1,I2,X,Y,Q,S)
C
C C$C$C$C$C$ CALLS ONLY LIBRARY ROUTINES C$C$C$C$C$
C
C   RSPLE RETURNS THE VALUE OF THE FUNCTION Y(X) EVALUATED AT POINT S
C   USING THE CUBIC SPLINE COEFFICIENTS COMPUTED BY RSPLN AND SAVED IN
C   Q.  IF S IS OUTSIDE THE INTERVAL (X(I1),X(I2)) RSPLE EXTRAPOLATES
C   USING THE FIRST OR LAST INTERPOLATION POLYNOMIAL.  THE ARRAYS MUST
C   BE DIMENSIONED AT LEAST - X(I2), Y(I2), AND Q(3,I2).
C
C                                                     -RPB
      DIMENSION X(1),Y(1),Q(3,1)
      DATA I/1/
      II=I2-1
C   GUARANTEE I WITHIN BOUNDS.
      I=MAX0(I,I1)
      I=MIN0(I,II)
C   SEE IF X IS INCREASING OR DECREASING.
      IF(X(I2)-X(I1))1,2,2
C   X IS DECREASING.  CHANGE I AS NECESSARY.
 1    IF(S-X(I))3,3,4
 4    I=I-1
      IF(I-I1)11,6,1
 3    IF(S-X(I+1))5,6,6
 5    I=I+1
      IF(I-II)3,6,7
C   X IS INCREASING.  CHANGE I AS NECESSARY.
 2    IF(S-X(I+1))8,8,9
 9    I=I+1
      IF(I-II)2,6,7
 8    IF(S-X(I))10,6,6
 10   I=I-1
      IF(I-I1)11,6,8
 7    I=II
      GO TO 6
 11   I=I1
C   CALCULATE RSPLE USING SPLINE COEFFICIENTS IN Y AND Q.
 6    H=S-X(I)
      RSPLE=Y(I)+H*(Q(1,I)+H*(Q(2,I)+H*Q(3,I)))
      RETURN
      END
 
      function cosd(arg)
      parameter (twopi=6.28318530718)
      parameter (degtorad=twopi/360.)
      cosd=cos(arg*degtorad)
      return
      end
      function acosd(arg)
      parameter (twopi=6.28318530718)
      parameter (degtorad=twopi/360.)
      parameter (radtodeg=360./twopi)
      acosd=radtodeg*acos(arg)
      return
      end
      double precision function dcosd(darg)
      real*8 darg
      real*8 dtwopi
      real*8 ddegtorad
      parameter (dtwopi=6.28318530718d0)
      parameter (ddegtorad=dtwopi/360.d0)
      dcosd=dcos(darg*ddegtorad)
      return
      end
      double precision function dacosd(darg)
      real*8 darg
      real*8 dtwopi
      real*8 dradtodeg
      real*8 ddegtorad
      parameter (dtwopi=6.28318530718d0)
      parameter (ddegtorad=dtwopi/360.d0)
      parameter (dradtodeg=360.d0/dtwopi)
      dacosd=dradtodeg*dacos(darg)
      return
      end
      function tand(arg)
      parameter (twopi=6.28318530718)
      parameter (degtorad=twopi/360.)
      tand=tan(arg*degtorad)
      return
      end
      function atand(arg)
      parameter (twopi=6.28318530718)
      parameter (degtorad=twopi/360.)
      parameter (radtodeg=360./twopi)
      atand=radtodeg*atan(arg)
      return
      end
      function sind(arg)
      parameter (twopi=6.28318530718)
      parameter (degtorad=twopi/360.)
      sind=sin(arg*degtorad)
      return
      end
      double precision function dsind(darg)
      real*8 darg
      real*8 dtwopi
      real*8 ddegtorad
      parameter (dtwopi=6.28318530718d0)
      parameter (ddegtorad=dtwopi/360.d0)
      dsind=dsin(darg*ddegtorad)
      return
      end


c***********************************************************************
      subroutine read_de06(drcty)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c---- open and read Q model file
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      character*(*) drcty
      character*20 binfilenames(8)
      data binfilenames/'clm_dq_050_final.bin','clm_dq_075_final.bin',
     &                  'clm_dq_100_final.bin','clm_dq_125_final.bin',
     &                  'clm_dq_150_final.bin','clm_dq_175_final.bin',
     &                  'clm_dq_200_final.bin','clm_dq_250_final.bin'/
      real qinv(360,180,8),periods(8)
      integer ilon,ilat
      logical exists
      data periods/50,75,100,125,159,175,200,250/
      common/dbe06/qinv,periods
      do ifl=1,8
         inquire(file=drcty//binfilenames(ifl),exist=exists)
         if(.not.exists) then
            write(*,*) 'DE06 Q model file '//drcty//
     &binfilenames(ifl)//' does not exist'
	   stop
         endif
         open(1,file=drcty//binfilenames(ifl),status='OLD',
     &             form='UNFORMATTED')
         do ilat =1,180
            read(1) (qinv(ilon,ilat,ifl),ilon=1,360)
         enddo
         close(1)
      enddo
      write(*,*) 'Read DE06 Q model files from directory ',drcty
      return
      end
c***********************************************************************
      subroutine point_de06(lon,lat,f,rqinv,nf)
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c---- Calculate frequency-dependent Q at a point (lon,lat)
c
cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
      integer nf
      real f(nf),rqinv(nf),lon,lat    
Cf2py intent(in) lon,lat,f
Cf2py intent(out) rqinv
Cf2py depend(nf) f
      real xlon,tqinv(8),tcube(3,8),work(3,8),tfrq(8)
      real qinv(360,180,8),periods(8)
      integer ilon,ilat,ifr
      data periods/50,75,100,125,159,175,200,250/
      common/dbe06/qinv,periods
      xlon = lon
      if (lon.gt.180.) then
         xlon = lon-360.
      endif
      ilat = max(1,nint(lat+90.))
      ilon = max(1,nint(xlon+180.))
      if (ilat.gt.180..or.ilon.gt.360.) then
         write(*,*) 'point_de06: Invlaid lon,lat=(',lon,lat,')'
         call flush()
         stop
      endif
      do ifr=1,8
         tfrq(ifr)  = 1./periods(9-ifr)
         tqinv(ifr) = qinv(ilon,ilat,9-ifr)
      enddo
      call rspln(1,8,tfrq,tqinv,tcube,work)
      do ifr=1,nf
        rqinv(ifr) = rsple(1,8,tfrq,tqinv,tcube,f(ifr))
      enddo
      return
      end
