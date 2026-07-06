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
 datdir    = "../../diag_data/"
 Explist   = (/"CMIP6",      "NOAA_20C",   "ERA5", "v2_1.SORRM.historical", "v2_1.LR.historical"/)
 Labels    = (/"CMIP6(MME)", "NOAA-20C",   "ERA5", "v2_1.SORRM",            "v2_1"/)
 Timseg    = (/"1951-2014",  "1951-2014",  "ERA5", "1951-2014",             "1951-2014"/)
 ngrps     = dimsizes(Explist)

 fnum      = (/"(a)","(b)","(c)"/)
 rstr      = ""

 vkyList   = (/"ASL", "ASL", "ASL"/)
 varList   = (/"ASL_CENLAT","ASL_CENLON","ASL_CENPSL"/)
 varUnit   = (/"degree","degree","hPa"/)
 varDecs   = (/"ASL Central Latitude", "ASL Central Longitude", "ASL Central Pressure"/)
 varFacs   = (/1,1,1/)
 varMin    = (/-80,160,940/)
 varMax    = (/-60,310,1020/)
 varSpc    = (/4, 20, 10/) 
 nvars     = dimsizes(varList)

 seas      = (/"Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"/)
 sear      = (/"D","J","F","M","A","M","J","J","A","S","O","N","D"/)
 nseas     = dimsizes(seas)

 do j = 0,nvars-1,1

  varnam    = varList(j)
  varkey    = vkyList(j)
  varunt    = varUnit(j)
  varfac    = varFacs(j)
  vardec    = varDecs(j)
 
  do i = 0,ngrps-1,1

    expnam = Explist(i)
    flnm   = datdir +expnam+"/"+expnam+"."+varkey+"."+scenario+".nc"
    f      = addfile (flnm, "r")

    x = ispan(0,nseas-1,1) 
    y = new((/nseas,6/),float)
    do k = 0,nseas-1,1
      vnm1   = varnam+"_"+seas(k)
      g      = f->$vnm1$
      y(k,0) = g(2)  ;Min 
      y(k,1) = g(6)  ;25th percentile 
      y(k,2) = g(8)  ;Median 
      y(k,3) = g(10) ;75th percentile 
      y(k,4) = g(14) ;Max
      y(k,5) = g(0)  ;mean
    end do 
    printMinMax(y,False)

    if(varnam.eq."ASL_CENLON") then 
      if (y(0,2).lt.0) then 
         y = y + 360.0 
      end if 
    end if 

   if (i.eq.0) then ; .and. k.eq.0 ) then 
    ;----------------------------------------------------------------------
    ; Plotting section
    ;----------------------------------------------------------------------
    wtype = "pdf"
    outdir = "./"
    system("if ! test -d " + outdir +" ; then mkdir " + outdir + " ; fi")
    plotname = outdir+"/fig_pdf_ASL_"+scenario +"_" + varnam
    wks      = gsn_open_wks(wtype,plotname)
    plot     = new((/nvars/),graphic)
    gsn_define_colormap(wks,"cb_9step")
    colors   = (/1,2,6,29,38,58/)
    markers  = (/16,6,7,8,4,0/)
    mkszf    = (/0.006,0.008,0.008,0.008,0.008/)*1.2
    mkthk    = (/1.5,1.5,1.5,1.5,1.5,1.5/)*1.2

  end if 
  ;---Resources to share between both plots
   res                  = True
   res@tmXTOn           = False
   res@tmYROn           = False

   res@vpXF             =  0.093
   res@vpYF             =  0.70
   res@vpWidthF         = 0.9 ;0.85
   res@vpHeightF        = 0.5 ;0.80

   FontHeightF              = 0.035
   res@tiMainFontHeightF    = FontHeightF*1.1
   res@tmYLLabelFontHeightF = FontHeightF*1.1
   res@tmXBLabelFontHeightF = FontHeightF*1.1
   res@tiXAxisFontHeightF   = FontHeightF*1.2
   res@tiYAxisFontHeightF   = FontHeightF*1.2

   bthick = 2.0
   res@tmBorderThicknessF    = bthick
   res@tmXBMajorThicknessF   = bthick
   res@tmXBMinorThicknessF   = res@tmXBMajorThicknessF/2.0
   res@tmYLMajorThicknessF   = bthick
   res@tmYLMinorThicknessF   = res@tmYLMajorThicknessF/2.0


   res@tmXMajorGrid                =  False ;True
   res@tmXMajorGridThicknessF      =  1.0
   res@tmXMajorGridLineDashPattern =  2
   res@tmYMajorGrid                =  False ;True
   res@tmYMajorGridThicknessF      =  1.0
   res@tmYMajorGridLineDashPattern =  2

   res@tmXBMajorOutwardLengthF = -0.024	; set tickmarks 
   res@tmXBMinorOutwardLengthF = -0.018
   res@tmXBMajorLengthF        = -0.024
   res@tmXBMinorLengthF        = -0.018

   res@tmYLMajorOutwardLengthF = -0.024 ; set tickmarks
   res@tmYLMinorOutwardLengthF = -0.018
   res@tmYLMajorLengthF        = -0.024
   res@tmYLMinorLengthF        = -0.018

   res1 = res
   res1@tiMainString        = fnum(j) + " "+varDecs(j)
   res1@tmXBLabels          = sear 
   res1@trYMinF             = varMin(j)    ; MÃ­nimo valor do eixo y.      
   res1@trYMaxF             = varMax(j)    ; MÃ¡ximo valor do eixo y.
   res1@tmYLMode            = "Manual"     ; Eixo y esquerdo (YL) de forma manual.
   res1@tmYLTickStartF      = res1@trYMinF  ; Onde comeÃ§am os valores do eixo y.
   res1@tmYLTickEndF        = res1@trYMaxF  ; Onde terminam os valores do eixo y.
   res1@tmYLTickSpacingF    = varSpc(j)     ; EspaÃ§amento entre os rÃ³tulos.
   res1@tmYLMinorOn         = False        ; Desliga o minortick do eixo y esquerdo.
   res1@tmYRBorderOn        = True         ; Habilita a borda do eixo y direito (YR). 
   res1@tmXTBorderOn        = True         ; Habilita a borda do eixo x superior (XB). 
   res1@tiYAxisString       = " (" + varunt+ ")"

   llres                    = True     ; PersonalizaÃ§Ã£o das linhas do boxplot.
   llres@gsLineThicknessF   = 1.8      ; Espessura da linha.

   opti           = True ; Habilita personalizaÃ§Ã£o da caixa.
   opti@boxWidth  = 0.45 ; Largura da caixa. Quanto maior o valor, mais larga serÃ¡ a caixa.
  ;opti@boxColors = (/"blue","red","green","orange"/) ; Cores de cada caixa.
   if(i.eq.0) then   
     plot(j)  = boxplot(wks,x,y(:,0:4),opti,res1,llres) ; GeraÃ§Ã£o do boxplot.
   end if 

   if(.not.isvar("dum")) then 
      dum = new ((/100,100,100/),graphic)
   end if 

   do ff = 0,dimsizes(x)-1,1
    res2 = True 
    res2@gsMarkerThicknessF = 2.0
    res2@gsMarkerColor      = colors(i)   ; It's okay to list more than you need here
    res2@gsMarkerIndex      = markers(i)                        ; polymarker style
    res2@gsMarkerSizeF      = mkszf(i)                      ; polymarker size
    dum(ff,i,j) = gsn_add_polymarker(wks,plot(j),x(ff),y(ff,5),res2) ; GeraÃ§Ã£o do boxplot.
   end do 
  ;overlay(plot(j),dum(i,j))  

 end do 

 ;---Specify y positions of each row of plots.
 pres                          = True
 pres@gsnFrame                 = False
 pres@gsnMaximize              = False
 pres@gsnPanelMainFontHeightF  = 0.015
 pres@gsnPanelFigureStrings    = ""
 pres@gsnPanelFigureStringsFontHeightF  = 0.01
 pres@gsnPanelXWhiteSpacePercent = 4
 gsn_panel(wks,plot(j),(/1,2/),pres)

;***********************************************
; legend resources
;***********************************************
 lgres                    = True
 lgres@vpWidthF           = 0.18                   ; width of legend (NDC)
 lgres@vpHeightF          = 0.10                   ; height of legend (NDC) 
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
 lgres@lgMarkerIndexes    = markers(0:ngrps-1)         ; marker indexes
 lgres@lgMarkerColors     = colors(0:ngrps-1) 
 lgres@lgMarkerThicknessF = mkthk(0:ngrps-1)
 lgres@lgLineLabelStrings = Labels 
 lgres@lgLineLabelFontHeights = FontHeightF * 1.8
 lgres@lgLabelFontHeightF     = FontHeightF * 1.8

 gsn_legend_ndc(wks,dimsizes(Labels),Labels,0.76,0.476,lgres)

 frame(wks)

end do

end

