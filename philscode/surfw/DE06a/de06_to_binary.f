      program de06tobinary
      character*16 txtfilenames(8) 
      character*20 binfilename
      real lat,lon,qinv(360)
      integer ilat,ilon
      data txtfilenames/'clm_dq_050_final','clm_dq_075_final',
     &                  'clm_dq_100_final','clm_dq_125_final',
     &                  'clm_dq_150_final','clm_dq_175_final',
     &                  'clm_dq_200_final','clm_dq_250_final'/
      do ifl=1,8
         open(unit=1,file=txtfilenames(ifl),status='OLD')
         binfilename = txtfilenames(ifl)//'.bin'
         open(unit=2,file=binfilename,status='NEW',form='UNFORMATTED')
         write(*,*) 'Converting ',txtfilenames(ifl),' to binary ',
     &               binfilename
         do ilat=1,180
            do ilon=1,360
               read(1,*) lat,lon,qinv(ilon)
            enddo
            write(2) (qinv(ilon),ilon=1,360)
         enddo
         close(1)
         close(2)
      enddo
      end
