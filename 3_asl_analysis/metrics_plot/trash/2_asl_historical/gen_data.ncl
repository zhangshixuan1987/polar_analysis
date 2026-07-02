; Calculates Westerly asl intensity and location statistics 
;
; Variables used: asl_loc asl_ind 
;
load "/lcrc/group/acme/ac.szhang/acme_scratch/e3sm_cvdp/ncl_code_sig/functions.ncl"

begin

  GENERATE_DATA       = True
  variname            = (/"DJF","JFM","MAM","JJA","JAS","SON","NDJFMA","MJJASO","ANN"/) 
  variname_title      = variname
  variname_title      = "ASL index"
  variname_title_seas = (/"DJF","JFM","MAM","JJA","JAS","SON","NDJFMA","MJJASO","ANN"/)
  si                  = (/ 1,1,3,6,7, 9,10,  5,  1 /)
  se                  = (/ 2,3,5,8,9,11, 5, 10, 12 /)

  mmename             = (/"NOAA_20Cen00","CMIP6","v2_1.LR","v3.LR"/)
  titname             = (/"Obs.","CMIP6 (MME)","v3.LR","v2_1.LR"/)
  nmme                = dimsizes(mmename)

  if(GENERATE_DATA) then 

    fout = getenv("OUTDIR")+"Model_vs_obs_MME.cvdp_data.asl_statis.nc"
    if (isfilepresent2(fout)) then
       z = addfile(fout,"w")
    else
       z = addfile(fout,"c")
    end if

    do gg = 0,dimsizes(variname)-1

      do ff = 0,nmme-1

        if(ff.eq.0) then 
          subStrs := str_get_field(modname,1,"-")
        else 
          subStrs := str_get_field(modname,1,"_")
        end if 
        cntr_ens  := ind(subStrs.eq.mmename(ff))
        nensm      = dimsizes(cntr_ens)

        do hh = 0,nensm-1

          fnt      = getenv("OUTDIR")+modname_mod+".cvdp_data.polar.asl.indices."+syear(hh)+"-"+eyear(hh)+".nc"
          fnt2     = getenv("OUTDIR")+modname_mod+".cvdp_data."+syear(hh)+"-"+eyear(hh)+".nc"
          arr1    := read_cvdp_le_data(fnt,fnt2,"asl_center_lon")
          arr2    := read_cvdp_le_data(fnt,fnt2,"asl_center_psl")
          arr3    := read_cvdp_le_data(fnt,fnt2,"asl_regmn_psl")
          arr2     = arr2 - arr3
          delete(arr3)
          nyyyy  = dimsizes(arr1&time)/12
          tmsk   = arr1&time
          do iyy = 0,nyyyy-1
            tmsk((iyy*12):((iyy+1)*12-1)) = ispan(1,12,1)
          end do

          if(hh.eq.0) then 
            arr_loc         = new((/nsim,dimsizes(arr1&time)/),double)
            arr_loc!0       = "ensmem"
            arr_loc!1       = "time"
            arr_loc&time    = arr1&time
            arr_ins         = arr_loc
            arr_loc@nens    = 0
            arr_ins@nens    = 0
          end if 

          if(gg.eq.0) then
            indxx = ind((tmsk.ge.si(gg).and.tmsk.le.se(gg)).or.tmsk.eq.12)
          else if(gg.eq.6) then
            indxx = ind(tmsk.lt.5.or.tmsk.gt.10)
          else
            indxx = ind(tmsk.ge.si(gg).and.tmsk.le.se(gg))
          end if
          end if
          arr_loc(hh,indxx)  = arr1(indxx)
          arr_ins(hh,indxx)  = avg(arr2(indxx))
          arr_loc@nens       = arr_loc@nens + 1
          arr_ins@nens       = arr_ins@nens + 1
          delete([/indxx/])
          delete([/tmsk,arr1,arr2,modname_mod,syear_mod,eyear_mod,names_mod,names_EM_mod/])

         end do

         z->$(/mmename(ff)+"_asl_loc_"+variname(gg)+"_em"/)$  = set_varAtts(arr_loc,arr_loc@long_name+" Ensemble Mean","degree","")
         z->$(/mmename(ff)+"_asl_ins_"+variname(gg)+"_em"/)$  = set_varAtts(arr_ins,arr_loc@long_name+" Ensemble Mean","degree","")

      end do

     end do 

  end if 

  if (CREATE_GRAPHICS.eq."False") then
     print("Finished: polar.asl_statis.ncl")
     exit
  end if

;======================================================================
; Create ensemble summary graphics
;
  cmap = read_colormap_file("t2m_29lev")
  LineColors      = cmap((/0,25,7,16,28/),:)
  LineThicknesses = (/20.,20.,20.,20.0,20.0/)*2.5      ; line thicknesses
  DashPatterns    = (/0.,0.,0.,0.,0./)      ; make all lines solid
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

  do dd = 0,dimsizes(variname)-1,1

     ofile = (/"ensemble_metrics_asl_"/)+variname_title_seas(dd)
     wks   = gsn_open_wks(wks_type,OUTDIR+"table")
     plot := new(2,graphic)
     fout  = getenv("OUTDIR")+"Model_vs_obs_MME.cvdp_data.asl_statis.nc"
     z     = addfile(fout,"r")

    do ii = 0,1

     if (ii.eq.0) then
        vxmin = 200.0 
        vxmax = 260.0 
        ncol_labels2 = "ASL longitude ("+variname_title_seas(dd)+")"
        polyres@tiXAxisString   = "degree"    
     end if
     if (ii.eq.1) then
        if (variname_title_seas(dd).eq."DJF") then 
          vxmin = -10.0 
          vxmax =  4.0 
        else 
          vxmin = -10.0
          vxmax =  4.0
        end if 
        ncol_labels2 = "ASL relative center pressure ("+variname_title_seas(dd)+")"
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
 
        labarr(nmme-nn)    = titname(nn) 
        polyres@gsLineColor  = LineColors(nn,: )
        polyres2@gsLineColor = LineColors(nn,:) 
        polyres3@gsLineColor = LineColors(nn,:) 

       if (ii.eq.0) then
          arr    = z->$(/mmename(nn)+"_asl_loc_"+variname(dd)+"_em"/)$ 
          stats := boxplot_stat(dim_avg_n(arr,0))
          finEM_10 = stats(0)
          finEM_90 = stats(4)
          arr_EM   = avg(arr)
          parr     = dim_avg_n(arr,1)
       end if
       if (ii.eq.1) then
          arr    = z->$(/mmename(nn)+"_asl_ins_"+variname(dd)+"_em"/)$  
          stats := boxplot_stat(dim_avg_n(arr,0))
          finEM_10 = stats(0)
          finEM_90 = stats(4)
          arr_EM   = avg(arr)
          parr     = dim_avg_n(arr,1)
       end if
       do oo = 0,dimsizes(parr)-1
          val = parr(oo)
          if (.not.ismissing(val)) then
            dum(cntr) = gsn_add_polyline(wks,plot(ii),(/val,val/),(/ycoord(nn)-.25,ycoord(nn)+.25/),polyres)
          end if
          cntr = cntr+1
       end do
      ;dumEM(cntr_em,0) =  gsn_add_polyline(wks,plot(ii),(/finEM_10,finEM_10/),(/ycoord(nn)-.45,ycoord(nn)+.45/),polyres2)    ; draw mean value
       dumEM(cntr_em,1) =  gsn_add_polyline(wks,plot(ii),(/arr_EM,arr_EM/),(/ycoord(nn)-.45,ycoord(nn)+.45/),polyres2)    ; draw mean value
      ;dumEM(cntr_em,2) =  gsn_add_polyline(wks,plot(ii),(/finEM_90,finEM_90/),(/ycoord(nn)-.45,ycoord(nn)+.45/),polyres2)    ; draw mean value
       cntr_em = cntr_em+1
       delete([/parr,arr,arr_EM/]) 
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

