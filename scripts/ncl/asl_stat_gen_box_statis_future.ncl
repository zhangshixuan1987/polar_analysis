;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;This script used to calculate and plot the zonal mean variables from CAM;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;The terms zonal and meridional are used to describe directions on a globe;;;;;;;
;;;;;;;Zonal means "along a latitudinal circle" or "in the west–east direction";;;;;;;;
;;;;;;;while meridional means "along a longitudinal circle" (a.k.a. meridian) ;;;;;;;;;
;;;;;;;or "in the north–south direction" [2];;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"

begin

 scenario  = "future"
 diro      = "../../diag_data/"
 datdir    = "/lcrc/group/acme/ac.szhang/acme_scratch/data/asl_analysis/"
 Explist   = (/"v2_1.SORRM.historical", "v2_1.SORRM.1950Control","v2_1.SORRM.ssp370"/)
 Labels    = (/"E3SM-SORRM (HIST)",     "E3SM-SORRM (Control)",  "E3SM-SORRM (SSP370)"/)
;Timseg    = (/"1951-2000",             "801-1000",              "2051-2100"/)
 Timseg    = (/"1951-2010",             "801-1000",              "2041-2100"/)

 ngrps     = dimsizes(Explist)

 fnum      = (/"(a)","(b)","(c)","(d)","(e)","(f)","(g)","(h)"/)
 rstr      = ""

 varList   = (/"ASL_CENLAT","ASL_CENLON","ASL_CENPSL"/)
 vkyList   = (/"ASL",       "ASL",       "ASL"/)
 nvars     = dimsizes(varList)
 seas      = (/"ANN","DJF","MAM","JJA","SON"/)
 nseas     = dimsizes(seas)

 ;;;;;;;;;;select the regions for plot;;;;;;;;;;;;;
 Regions   = (/"Polar"/)
 MaxLat    = -55.

;ASL domain region
 plot_box_line = True 
 rlats  = -80.0 ;-75.0 
 rlatn  = -60.0 ;-60.0
 rlonw  = 170   ;180.0 
 rlone  = 298   ;310.0 

 do i = 0,ngrps-1,1

   do j = 0,nvars-1,1

    varnam    = varList(j)
    varkey    = vkyList(j)

    expnam   = Explist(i)
    if(expnam.eq."CMIP6") then 
      flnm0  = systemfunc("ls " + datdir +expnam+"/*/*."+varkey+".*.nc")
      f0     = addfiles (flnm0, "r") 
      print(num(flnm0))
     ;ListSetType (f0, "join")      
    else 
      flnm0  = systemfunc("ls " + datdir +expnam+"/"+expnam +"."+varkey+".*.nc")
      f0     = addfiles (flnm0, "r")
    end if 

    time      = f0[0]->time 
    xtmp      = f0[:]->$varnam$
    xtmp!1    = "time"
    xtmp&time = time 

    dimx          = dimsizes(xtmp(time|:,ens|:))
    ytmp          = new((/dimx(0),dimx(1),1,1/),typeof(xtmp))
    ytmp(:,:,0,0) = xtmp(time|:, ens|:)
    ytmp!0        = "time"
    ytmp&time     = time 
    ytmp!1        = "ens"
    ytmp!2        = "lat"
    ytmp!3        = "lon"

    utc_date  = cd_calendar(time,0)
    yyyy      = ispan(toint(min(utc_date(:,0))),toint(max(utc_date(:,0))),1)
    yyyy!0    = "year"
    yyyy&year = yyyy
   ;print(utc_date)

    tmpstrs   = str_split(Timseg(i), "-")
    syymm     = toint(tmpstrs(0))
    eyymm     = toint(tmpstrs(1))
    indx      = ind(yyyy.ge.syymm.and.yyyy.le.eyymm)
    
    do k = 0,nseas-1,1

     if(seas(k).eq."ANN") then 
       xsea   = month_to_annual(ytmp, 1)
     else
       printVarSummary(ytmp)
       xsea   = month_to_season(ytmp, seas(k))
     end if 
   
     x1d        = ndtooned(xsea)
     opt        = True ; Habilita personalizaÃ§Ã£o da estatÃ­stica.
     opt@PrintStat = False ; Mostra na tela o resultado da estÃ¡tistica da funÃ§Ã£o stat_dispersion. 
     g          = stat_dispersion(x1d,opt)
     printVarSummary(g)
     g!0        = "nstat"
     g&nstat    = ispan(0,dimsizes(g)-1,1)
     stat       = new(dimsizes(g),"string")
     stat!0     = "nstat"
     stat&nstat = ispan(0,dimsizes(g)-1,1)
     stat       = (/"Mean","StdDev","Min","LowDec","LowOct","LowSex","LowQuartile","LowTri",\
                    "Median","HighTri","HighQuartile","HighSex","HighOct",\
                    "HighDec","Max","Range","Dispersion","RMS","Ntotal","Nused","Nmiss","Pctmiss",\
                    "Lower0.1p","Lower1p","Lower5p","Upper5p","Upper1p","Upper0.1p",\
                    "Skewness","Kurtosis"/)

     outdir = diro+expnam+"/"
     system("if ! test -d " + outdir +" ; then mkdir -p " + outdir + " ; fi")
     fout = outdir+expnam+"."+varkey+"."+scenario+".nc"
     if (.not.isfilepresent(fout)) then 
       system("rm -rvf " + fout)
       setfileoption("nc","Format","Netcdf4") ;"LargeFile")
       z = addfile(fout,"cw")
     else
       z = addfile(fout,"rw")
     end if 
     vout = varnam+"_"+seas(k)
     z->stat   = stat
     z->$vout$ = g 
     delete([/xsea,x1d,g/])
   end do 
  delete([/time,xtmp,flnm0,f0,dimx,ytmp/])
  delete([/utc_date,yyyy,tmpstrs,syymm,eyymm,indx/])

 end do 
end do 

end

