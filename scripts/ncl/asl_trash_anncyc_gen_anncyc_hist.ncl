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

 scenario  = "hist"
 diro      = "../../diag_data/"
 datdir    = "/lcrc/group/acme/ac.szhang/acme_scratch/data/polar_analysis/"
 Explist   = (/"NOAA_20C",  "CMIP6",       "v2_1.SORRM.historical", "v2_1.LR.historical"/)
 Labels    = (/"NOAA-20C",  "CMIP6(MME)",  "v2_1.SORRM",            "v2_1"/)
 Timseg    = (/"1950-2014", "1950-2014",   "1950-2014",             "1950-2014"/)
 ngrps     = dimsizes(Explist)

 varList   = (/"ASL_INDEX","ASL_CENLAT","ASL_CENLON","ASL_CENPSL"/)
 vkyList   = (/"ASL",      "ASL",       "ASL",       "ASL"/)
 nvars     = dimsizes(varList)

 do i = 0,ngrps-1,1

   do j = 0,nvars-1,1

    varnam    = varList(j)
    varkey    = vkyList(j)

    expnam   = Explist(i)
    if(expnam.eq."CMIP6") then 
      flnm0  = systemfunc("ls " + datdir +expnam+"/*/*."+varkey+".*"+Timseg(i)+"*.nc")
      f0     = addfiles (flnm0, "r") 
     ;ListSetType (f0, "join")      
    else 
      flnm0  = systemfunc("ls " + datdir +expnam+"/"+expnam +"."+varkey+".*"+Timseg(i)+"*.nc")
      f0     = addfiles (flnm0, "r")
    end if 

    time        = f0[0]->time 
    xtmp        = f0[:]->$varnam$
    xtmp!1      = "time"
    xtmp&time   = time 

    utc_date    = cd_calendar(time,0)
    yyyy        = ispan(toint(min(utc_date(:,0))),toint(max(utc_date(:,0))),1)
    yyyy!0      = "year"
    yyyy&year   = yyyy
    tmpstrs     = str_split(Timseg(i), "-")
    syymm       = toint(tmpstrs(0))
    eyymm       = toint(tmpstrs(1))
    indx        = ind(yyyy.ge.syymm.and.yyyy.le.eyymm)
     
    dimx        = dimsizes(xtmp(time|:,ens|:))
    ytmp        = reshape(xtmp,(/dimx(0)/12,12,dimx(1)/)) 
    ytmp!0      = "year"
    ytmp!1      = "month"
    ytmp!2      = "ens"
    ytmp&year   = yyyy
    ytmp&month  = ispan(1,12,1)
    
    ztmp        = ytmp(month|:,year|indx,ens|:)
    x2d         = reshape(ztmp,(/12,dimsizes(indx)*dimx(1)/))
    gmn         = dim_avg_n_Wrap(ytmp(indx,:,:),0)  
    gsd         = dim_stddev_n_Wrap(ytmp(indx,:,:),0)

    outdir = diro+expnam+"/"
    system("if ! test -d " + outdir +" ; then mkdir -p " + outdir + " ; fi")
    fout = outdir+expnam+"."+varkey+".anacyc."+scenario+".nc"
    if (.not.isfilepresent(fout)) then 
      system("rm -rvf " + fout)
      setfileoption("nc","Format","Netcdf4") ;"LargeFile")
      z = addfile(fout,"cw")
    else
      z = addfile(fout,"rw")
    end if 
    vout = varnam+"_mn"
    z->$vout$ = gmn
    vout = varnam+"_sd"
    z->$vout$ = gsd

  delete([/x2d,gmn,gsd,time,flnm0,f0,dimx,xtmp,ytmp,ztmp/])
  delete([/utc_date,yyyy,tmpstrs,syymm,eyymm,indx/])

 end do 
end do 

end

