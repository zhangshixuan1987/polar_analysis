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
load "./add_color_bar.ncl"

begin

 scenario  = "future"
 datdir    = "../../diag_data/"
 Explist   = (/"NOAA_20C", "v2_1.SORRM.historical", "v2_1.SORRM.ssp370"/)
 Labels    = (/"NOAA-20C", "Historical",            "SSP370"/)
 LabList   = (/"NOAA-20C", "Historical",            "SSP370"/)
 Timseg    = (/"1951-2010","1951-2010",             "2015-2100"/)
 ngrps     = dimsizes(Explist)

 fnum      = (/"(d)","(e)","(f)","(g)"/)
 rstr      = ""

 vkyList   = (/"ASL", "ASL", "ASL"/)
 varList   = (/"ASL_CENLAT","ASL_CENLON","ASL_CENPSL"/)
 varUnit   = (/"degree","degree","hPa"/)
 varDecs   = (/ "ASL Central Latitude", "ASL Central Longitude", "ASL Central Pressure"/)
 varFacs   = (/1,1,1/)
 varMin    = (/-80,160,956/)
 varMax    = (/-60,310,1004/)
 varSpc    = (/5.0,10.0,10.0/)
 nvars     = dimsizes(varList)

 seas      = (/"J","F","M","A","M","J",\
               "J","A","S","O","N","D","J"/)
 nseas     = dimsizes(seas)

 ;----------------------------------------------------------------------
 ; Plotting section
 ;----------------------------------------------------------------------
 wtype = "pdf"
 outdir = "./"
 system("if ! test -d " + outdir +" ; then mkdir " + outdir + " ; fi")
 plotname = outdir+"/fig_anacyc_ASL_"+scenario
 wks      = gsn_open_wks(wtype,plotname)
 plot     = new((/nvars/),graphic)
 gsn_define_colormap(wks,"cb_9step")
 colors   = (/1,2,6,53,29,38,58/)
 bopc     = (/1.0,1.0,1.0,1.0/)
 pattn    = (/0,0,0,0,0/)
 lnthick  = (/1.5,1.5,1.5,1.5,1.5,1.5/)*1.0
 dashind  = (/0,0,0,0,0/)
 markers  = (/16,6,7,8,4,0/)
 mkszf    = (/0.006,0.008,0.008,0.008,0.08/)
 mkthk    = (/1.5,1.5,1.5,1.5,1.5,1.5/)*1.0
 
 do j = 0,nvars-1,1

  varnam    = varList(j)
  varkey    = vkyList(j)
  varunt    = varUnit(j)
  varfac    = varFacs(j)
  vardec    = varDecs(j)
 
  do i = 0,ngrps-1,1

    expnam = Explist(i)
    flnm   = datdir +expnam+"/"+expnam+"."+varkey+".anacyc."+scenario+".nc"
    f      = addfile (flnm, "r")

    vnm    = varnam+"_mn"
    var    = f->$vnm$
    vnm    = varnam+"_sd"
    vsd    = f->$vnm$
    if(varnam.eq."ASL_CENLON") then
      if (var(0).lt.0) then
         var = var + 360.0
      end if
    end if

    if (i.eq.0) then 
      x  = ispan(0,nseas-1,1) 
      y1 = new((/ngrps,dimsizes(x)/),float)
      y2 = new((/ngrps,dimsizes(x)/),float)
      y3 = new((/ngrps,dimsizes(x)/),float)
    end if 
    printVarSummary(var)
    print(var)

    y1(i,0:dimsizes(x)-2) = tofloat(var)
    y2(i,0:dimsizes(x)-2) = tofloat(var - vsd )
    y3(i,0:dimsizes(x)-2) = tofloat(var + vsd) 
    y1(i,dimsizes(x)-1)   = y1(i,0)
    y2(i,dimsizes(x)-1)   = y2(i,0)
    y3(i,dimsizes(x)-1)   = y3(i,0)

    printMinMax(y1,False)
    printMinMax(y2,False)
    printMinMax(y3,False)
  end do 

  ;---Resources to share between both plots
   res                  = True
   res@tmXTOn           = False
   res@tmYROn           = False
   res@gsnFrame         = False
   res@gsnDraw          = False

   res@vpXF             =  0.093
   res@vpYF             =  0.70
   res@vpWidthF         =  0.85
   res@vpHeightF        =  0.80

   FontHeightF              = 0.035
   res@tiMainFontHeightF    = FontHeightF*1.1
   res@tmYLLabelFontHeightF = FontHeightF*1.1
   res@tmXBLabelFontHeightF = FontHeightF*1.1
   res@tiXAxisFontHeightF   = FontHeightF*1.2
   res@tiYAxisFontHeightF   = FontHeightF*1.2

   res@tmXBMajorOutwardLengthF = -0.024 ; set tickmarks
   res@tmXBMinorOutwardLengthF = -0.018
   res@tmXBMajorLengthF        = -0.024
   res@tmXBMinorLengthF        = -0.018

   res@tmYLMajorOutwardLengthF = -0.024 ; set tickmarks
   res@tmYLMinorOutwardLengthF = -0.018
   res@tmYLMajorLengthF        = -0.024
   res@tmYLMinorLengthF        = -0.018

   res@tmXMajorGrid                =  False ;True
   res@tmXMajorGridThicknessF      =  1.0
   res@tmXMajorGridLineDashPattern =  2
   res@tmYMajorGrid                =  False ;True
   res@tmYMajorGridThicknessF      =  1.0
   res@tmYMajorGridLineDashPattern =  2

   bthick = 2.0
   res@tmBorderThicknessF    = bthick
   res@tmXBMajorThicknessF   = bthick
   res@tmXBMinorThicknessF   = res@tmXBMajorThicknessF/2.0
   res@tmYLMajorThicknessF   = bthick
   res@tmYLMinorThicknessF   = res@tmYLMajorThicknessF/2.0

   res@tiMainString        = "" 
   res@tmXBMode            = "Explicit"     ; Eixo y esquerdo (YL) de forma manual.
   res@trXMinF             = min(x) ;- 1   ; MÃ­nimo valor do eixo y.
   res@trXMaxF             = max(x) ;+ 1   ; MÃ¡ximo valor do eixo y.
   res@tmXBValues          = x ; fspan(res@trXMinF,res@trXMaxF-1.0,dimsizes(x)+1)
   res@tmXBLabels          = seas 

   res@trYMinF             = varMin(j)    ; MÃ­nimo valor do eixo y.      
   res@trYMaxF             = varMax(j)    ; MÃ¡ximo valor do eixo y.
   res@tmYLMode            = "Manual"     ; Eixo y esquerdo (YL) de forma manual.
   res@tmYLTickStartF      = res@trYMinF  ; Onde comeÃ§am os valores do eixo y.
   res@tmYLTickEndF        = res@trYMaxF  ; Onde terminam os valores do eixo y.
   res@tmYLTickSpacingF    = varSpc(j)     ; EspaÃ§amento entre os rÃ³tulos.
   res@tmYLMinorOn         = False        ; Desliga o minortick do eixo y esquerdo.
   res@tmYRBorderOn        = True         ; Habilita a borda do eixo y direito (YR). 
   res@tmXTBorderOn        = True         ; Habilita a borda do eixo x superior (XB). 
   res@tiYAxisString       = " (" + varunt+ ")"

   res@gsnLeftString       = fnum(j)+" " +varDecs(j)
   res@gsnRightString      = ""

 
;   res@tmYLOn                 = False
;   res@tmYROn                 = True
;   res@tmXBOn                 = False
;   res@tmXMajorGrid           = False
;   res@tmYMajorGrid           = False
;   res@tmXMinorGrid           = False
;   res@tmYMinorGrid           = False
;   res@gsnLeftString          = ""
;   res@gsnRightString         = ""
;   res@gsnCenterString        = ""
;   res@tiXAxisString          = ""
   res@gsnXYBarChart          = False
   res@xyMarkLineMode        := "Lines";"MarkLines"
   res@xyMarkers             := markers(0:ngrps-1)
   res@xyMarkerSizes         := mkszf(0:ngrps-1)
   res@xyMarkerThicknesses   := lnthick(0:ngrps-1)*0.5
   res@xyMarkerColors        := colors(0:ngrps-1)
   res@xyDashPatterns        := dashind(0:ngrps-1)
   res@xyLineThicknesses     := lnthick(0:ngrps-1)
   res@xyLineColors          := colors(0:ngrps-1)
   res@xyExplicitLegendLabels = "   " + LabList(0:ngrps-1)
   res@pmLegendDisplayMode    = "Always"            ; turn on legend
   res@pmLegendSide           = "Top"               ; Change location of
   res@pmLegendWidthF         = 0.12
   res@pmLegendHeightF        = 0.12
   res@pmLegendParallelPosF   = 0.235 ;+ xpos               ; move units right
   res@pmLegendOrthogonalPosF = -0.65 ;+ ypos              ; move units down
   res@lgPerimOn              = False               ; turn off box around
   res@lgLabelFontHeightF     = FontHeightF*0.9       ; label font height
   print(y1)
   plot(j) = gsn_csm_xy(wks,x,y1,res)

;   if(i.ne.0) then
;     dumx = gsn_csm_xy(wks,x,y1(1:ngrps-1,:),resR)
;     overlay(plot(j),dumx)
;   end if 

end do

 ;---Specify y positions of each row of plots.
 pres                          = True
 pres@gsnFrame                 = False
 pres@gsnMaximize              = False
 pres@gsnPanelMainFontHeightF  = 0.015
 pres@gsnPanelFigureStrings    = ""
 pres@gsnPanelFigureStringsFontHeightF  = 0.01
 pres@gsnPanelXWhiteSpacePercent = 4
 gsn_panel(wks,plot,(/1,nvars/),pres)

;***********************************************
; legend resources
;***********************************************
 lgres                    = True
 lgres@vpWidthF           = 0.18 *0.8                 ; width of legend (NDC)
 lgres@vpHeightF          = 0.08 *0.8                   ; height of legend (NDC) 
 lgres@lgRightMarginF     = 0.01
 lgres@lgBoxMajorExtentF  = 0.15
 lgres@lgBoxMinorExtentF  = 0.2
 lgres@lgPerimOn          = False 
 lgres@lgTitleString      = "" 
 lgres@lgAutoManage       = False                 ; Necessary to set font hgt
 lgres@lgMonoLineColor    = True
 lgres@lgLineColor        = "black"
 lgres@lgMonoItemType     = True                 ; more than one type
 lgres@lgItemType         = "Markers" 
 lgres@lgMarkerIndexes    = markers(2:ngrps+1)         ; marker indexes
 lgres@lgMarkerColors     = colors(2:ngrps+1) 
 lgres@lgMarkerThicknessF = mkthk(2:ngrps+1)
 lgres@lgLineLabelStrings = Labels 
 lgres@lgLineLabelFontHeights = FontHeightF * 1.5
 lgres@lgLabelFontHeightF     = FontHeightF * 1.5

 gsn_legend_ndc(wks,dimsizes(Labels),Labels,0.75,0.455,lgres)

 frame(wks)

end

