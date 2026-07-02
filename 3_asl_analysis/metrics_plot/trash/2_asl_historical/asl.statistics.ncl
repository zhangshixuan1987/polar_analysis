; Calculates Westerly asl intensity and location statistics 
;
; Variables used: asl_loc asl_ind 
;
load "/lcrc/group/acme/ac.szhang/acme_scratch/e3sm_cvdp/ncl_code_sig/functions.ncl"

begin
  print("Starting: polar.asl_statis.ncl")
  SCALE_TIMESERIES = "False"  ;getenv("SCALE_TIMESERIES")
  OUTPUT_DATA      = "True"   ;getenv("OUTPUT_DATA")
  PNG_SCALE        = 1.5      ;tofloat(getenv("PNG_SCALE"))
  OPT_CLIMO        = "Full"   ;"Custom" ;getenv("OPT_CLIMO")
  CLIMO_SYEAR      = 1915     ;toint(getenv("CLIMO_SYEAR"))
  CLIMO_EYEAR      = 2014     ;toint(getenv("CLIMO_EYEAR"))
  OUTPUT_TYPE      = "png"    ;getenv("OUTPUT_TYPE")
  COLORMAP         = 0        ;getenv("COLORMAP")
  OUTDIR           = "./data/"
  GENERATE_DATA    = True 
  CREATE_GRAPHICS  = True 

  wks_type = OUTPUT_TYPE
  if (wks_type.eq."png") then
     wks_type@wkWidth = 1500*PNG_SCALE
     wks_type@wkHeight = 1500*PNG_SCALE
  end if

  variname            = (/"DJF","JFM","MAM","JJA","JAS","SON","NDJFMA","MJJASO","ANN"/) 
  variname_title      = variname
  variname_title      = "ASL index"
  variname_title_seas = (/"DJF","JFM","MAM","JJA","JAS","SON","NDJFMA","MJJASO","ANN"/)
  si                  = (/ 1,1,3,6,7, 9,10, 5, 1 /)
  se                  = (/ 2,3,5,8,9,11, 5,10,12 /)

  mmename             = (/"NOAA", "CMIP6",  "v3.LR-historical",  "v2.1.LR-historical",\
                          "v3.LR-piControl", "v2.1.LR-piControl",  "v2.SORRM.CRYO1950"/)
  titname             = (/"NOAA_20C", "CMIP6 (MME)", "E3SMv3","E3SMv2.1","E3SMv3","E3SMv2.1","E3SMv3"/)
  nmme                = dimsizes(mmename)

  nsim = numAsciiRow("namelist_psl")
  na = asciiread("namelist_psl",(/nsim/),"string")
  names = new(nsim,"string")
  paths = new(nsim,"string")
  syear = new(nsim,"integer",-999)
  eyear = new(nsim,"integer",-999)
  names_EM = new(nsim,"string")
  EM_num = new(nsim,"integer",-999)
  delim = "|"

  do gg = 0,nsim-1
     names(gg) = str_strip(str_get_field(na(gg),1,delim))
     paths(gg) = str_strip(str_get_field(na(gg),2,delim))
     syear(gg) = stringtointeger(str_strip(str_get_field(na(gg),3,delim)))
     eyear(gg) = stringtointeger(str_strip(str_get_field(na(gg),4,delim)))
     temp      = str_split(str_strip(str_get_field(na(gg),5,delim)),"-")
     EM_num(gg)    = toint(temp(0))
     names_EM(gg)  = str_join(temp(1:),"-")
     delete(temp)
  end do
  modname = str_sub_str(names," ","_")

  bc = (/"/","'","(",")"/)
  do ff = 0,dimsizes(modname)-1
     do gg = 0,dimsizes(bc)-1
        modname(ff) = str_sub_str(modname(ff),bc(gg),"_")
     end do
  end do
  nyr = eyear-syear+1
  nyr_max = max(nyr)

  do gg = 0,dimsizes(variname)-1

    arr_EM  = new((/2,nmme/),double)
    parr    = new((/2,nmme,100/),double)
    do ff = 0,nmme-1
      modnm     = str_sub_str(modname,"v2_1","v2.1")
      modnm     = str_sub_str(modnm,"CMIP6-","CMIP6_")
      subStrs  := str_get_field(modnm,1,"_")
      cntr_ens := ind(subStrs.eq.mmename(ff)) 
      nensm     = dimsizes(cntr_ens)
      do hh = 0,nensm-1
        modname_mod  = modname(cntr_ens(hh))
        syear_mod    = syear(cntr_ens(hh))
        eyear_mod    = eyear(cntr_ens(hh))
        names_mod    = names(cntr_ens(hh))
        names_EM_mod = names_EM(cntr_ens(hh))

        fnt      = OUTDIR+modname_mod+".cvdp_data.psl.indices."+syear(cntr_ens(hh))+"-"+eyear(cntr_ens(hh))+".nc"
        fnt2     = OUTDIR+modname_mod+".cvdp_data."+syear(cntr_ens(hh))+"-"+eyear(cntr_ens(hh))+".nc"
        print(fnt)
        arr1    := read_cvdp_le_data(fnt,fnt2,"asl_clon")
        arr2    := read_cvdp_le_data(fnt,fnt2,"asl_rcpsl")
        nyyyy  = dimsizes(arr1&time)/12
        tmsk   = arr1&time
        do iyy = 0,nyyyy-1
          tmsk((iyy*12):((iyy+1)*12-1)) = ispan(1,12,1)
        end do

        if(hh.eq.0) then 
          arr_loc       = new((/nsim,dimsizes(arr1&time)/),float)
          arr_loc!0     = "ensmem"
          arr_loc!1     = "time"
          arr_ins       = arr_loc
          arr_loc@nens  = 0
          arr_ins@nens  = 0
        end if 

        if(gg.eq.0) then
          indxx = ind((tmsk.ge.si(gg).and.tmsk.le.se(gg)).or.tmsk.eq.12)
        else if(gg.eq.6) then
          indxx = ind(tmsk.lt.5.or.tmsk.gt.10)
        else
          indxx = ind(tmsk.ge.si(gg).and.tmsk.le.se(gg))
        end if
        end if
        arr_loc(hh,indxx)  = (/arr1(indxx)/)
        arr_ins(hh,indxx)  = (/avg(arr2(indxx))/)
        arr_loc@_FillValue  = -9999.0
        arr_ins@_FillValue  = -9999.0
        arr_loc@missing_value  = -9999.0
        arr_ins@missing_value  = -9999.0
        arr_loc@nens       = arr_loc@nens + 1
        arr_ins@nens       = arr_ins@nens + 1
        arr_loc@long_name  = "location"
        arr_ins@long_name  = "insensity"
        delete([/indxx,tmsk,arr1,arr2,modname_mod,syear_mod,eyear_mod,names_mod,names_EM_mod/])
      end do
      arr_EM(0,ff) = avg(arr_loc) 
      arr_EM(1,ff) = avg(arr_ins)
      printVarSummary(arr_loc)

      parr(0,ff,0:dimsizes(arr_loc(:,0))-1)  = (/dim_avg_n(arr_loc,1)/)
      parr(1,ff,0:dimsizes(arr_ins(:,0))-1)  = (/dim_avg_n(arr_ins,1)/)
      delete([/arr_loc,arr_ins/])
    end do 

  if (CREATE_GRAPHICS.eq."False") then
     print("Finished: polar.asl_statis.ncl")
     exit
  end if

;======================================================================
; Create ensemble summary graphics
;
  cmap = read_colormap_file("t2m_29lev")
  LineColors      = cmap((/0,25,7,16,28,7,16,28,7/),:)
  LineThicknesses = (/20.,20.,20.,20.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0/)*2.5      ; line thicknesses
  DashPatterns    = (/0.,0.,0.,0.,0.,0,0,0,0/)      ; make all lines solid
  delete(cmap)

  yAxis = ispan(0,nmme+1,1)
  labarr = new(nmme+2,"string")         ;Prepare actual Y-axis labels...
  labarr(0) = ""
  labarr(nmme+1) = ""

  pres = True
  polyres = True    ; individual obs/model line plot resource list
  if(wks_type.eq."png") then
   lthick = 4.0
  else
   lthick = 1.0
  end if

  FontHeightF = 0.022
  polyres@tiMainFontThicknessF       = 3.0
  polyres@tiMainFontHeightF          = FontHeightF*1.2
  polyres@tmYLLabelFontHeightF       = FontHeightF
  polyres@tmXBLabelFontHeightF       = FontHeightF
  polyres@tiXAxisFontHeightF         = FontHeightF
  polyres@tiYAxisFontHeightF         = FontHeightF
  
  polyres@vpHeightF       = 0.5
  polyres@vpWidthF        = 0.8

  polyres@gsLineThicknessF  = 3.5                          ; thickness of lines
  polyres@gsLineDashPattern = 0

  polyres2 = polyres   ; for ensemble mean bars
  polyres2@gsLineThicknessF  = 7.5
  polyres2@gsLineDashPattern = 0
  polyres2@gsLineOpacityF = 0.5
  polyres3 = polyres2   ; for 10/90% bars
  polyres3@gsLineThicknessF  = 5.5

  panres = True
  panres@gsnPanelYWhiteSpacePercent = 4.0
  Fscale = 0.92

  wks_type = "png"    ; output png
  wks_type@wkWidth  = 2500
  wks_type@wkHeight = 2500

  ofile = (/"ensemble_metrics_asl_"/)+variname_title_seas(gg)
  wks   = gsn_open_wks(wks_type,OUTDIR+"table")
  plot := new(2,graphic)

    do ii = 0,1

     if (ii.eq.0) then
        vxmin = 200.0 
        vxmax = 260.0 
        ncol_labels2 = "ASL longitude ("+variname_title_seas(gg)+")"
        polyres@tiXAxisString   = "degree"    
     end if
     if (ii.eq.1) then
        if (variname_title_seas(gg).eq."DJF") then 
          vxmin = -10.0 
          vxmax =  4.0 
        else 
          vxmin = -10.0
          vxmax =  4.0
        end if 
        ncol_labels2 = "ASL relative center pressure ("+variname_title_seas(gg)+")"
        polyres@tiXAxisString   = "hPa"
     end if

     ph = .65
     plot(ii) = create "plot" logLinPlotClass wks
        "vpHeightF": ph
        "vpWidthF": 0.8
        "trYMinF" : min(yAxis)+0.25
        "trYMaxF" : max(yAxis)-0.25
        "trXMinF" : vxmin  ; or 0 for rms
        "trXMaxF" : vxmax   ; or 1.0 for pc
        "pmTickMarkDisplayMode" : "Always"
        "tmYLMode"   : "Explicit"
        "tmYLValues" : yAxis
        "tmYROn"  : False
        "tmXTOn"  : False
        "tmBorderLineColor" : "black"
        "tmXBMajorLineColor" : "black"
        "tmXBMinorLineColor" : "black"
        "tmYLMajorLineColor" : "black"
        "tmYLMinorLineColor" : "black"
        "tmBorderThicknessF" : lthick
        "tmXBMajorThicknessF" : lthick
        "tmXBMinorThicknessF" : lthick
        "tmYLMajorThicknessF" : lthick
        "tmYLMinorThicknessF" : lthick
        "tmYRMajorThicknessF" : lthick
        "tmYRMinorThicknessF" : lthick
        "tmXBLabelFontHeightF" : 0.028 * Fscale
        "tmYLLabelFontHeightF" : 0.028 * Fscale
        "pmTitleDisplayMode": "Always"                   ; allow titles
        "tiMainOn"     : True

        "tiXAxisString"  : polyres@tiXAxisString
        "tiMainString"   : ncol_labels2
        "tiMainFontHeightF" : 0.038 * Fscale
     end create

     dum   := new((/10000/),graphic)
     dumEM := new((/10000,3/),graphic)
     cntr = 0
     cntr_em = 0
     ycoord = ispan(1,nmme,1)
     ycoord = ycoord(::-1)
     sti = 0

     do nn = 0,nmme-1,1
 
        labarr(nmme-nn)      = titname(nn) 
        polyres@gsLineColor  = LineColors(nn,: )
        polyres2@gsLineColor = LineColors(nn,:) 
        polyres3@gsLineColor = LineColors(nn,:) 
       do oo = 0,dimsizes(parr(ii,nn,:))-1
          val = parr(ii,nn,oo)
          if (.not.ismissing(val)) then
            dum(cntr) = gsn_add_polyline(wks,plot(ii),(/val,val/),(/ycoord(nn)-.25,ycoord(nn)+.25/),polyres)
          end if
          cntr = cntr+1
       end do
       dumEM(cntr_em,1) =  gsn_add_polyline(wks,plot(ii),(/arr_EM(ii,nn),arr_EM(ii,nn)/),(/ycoord(nn)-.45,ycoord(nn)+.45/),polyres2)    ; draw mean value
       cntr_em = cntr_em+1
     end do

     pres@tmYLLabels := labarr
     attsetvalues(plot(ii),pres)
     dumname = unique_string("dum")
     plot@$dumname$ = dum
     dumname = unique_string("dum")
     plot@$dumname$ = dumEM
     end do
     gsn_panel(wks,plot,(/1,2/),panres)
     delete(wks)
     if (wks_type.eq."png") then
       system("mv "+OUTDIR+"table."+wks_type+" "+OUTDIR+ofile+".png")
     else
       system("convert -density 2500 -trim +repage -border 40 -bordercolor white -transparent white "+OUTDIR+"table."+wks_type+" "+OUTDIR+ofile+".png")
       system("rm "+OUTDIR+"table.png")
     end if 
  end do
  delete([/pres,panres,polyres,polyres2/])

  print("Finished: polar.asl_statis.ncl")
end

