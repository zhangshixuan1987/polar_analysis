;-----------------------------------------------------------------
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

begin
  runnam   = "v2_1.LR.historical_0101"
  expstr   = "LR.HIST_0101"

  runnam   = "v2_1.LR.historical_0151"
  expstr   = "LR.HIST_0151"

  runnam   = "v2_1.LR.historical_0201"
  expstr   = "LR.HIST_0201"

  runnam   = "v2_1.LR.historical_0251"
  expstr   = "LR.HIST_0251"

  runnam   = "v2_1.LR.historical_0301"
  expstr   = "LR.HIST_0301"

  runtim   = "1985-2014"
  expnam   = "v2_1.SORRM.histssp370"

  dat_dir  = "/lcrc/group/e3sm/ac.szhang/acme_scratch/e3sm_project/E3SMv21_testings/"
  top_dir  = "/lcrc/group/e3sm/public_html/diagnostic_output/ac.golaz/E3SMv2_1/"
  run_dir  = top_dir+runnam+"/e3sm_diags/atm_monthly_180x360_aave/model_vs_obs_"+runtim+"/viewer/table-data/"
  diro     = dat_dir +"paper_material/fig_data/"

  strs     = asciiread("var_list.txt",-1,"string")
  varnam   = str_squeeze(str_get_field(strs, 1, ","))
  varunt   = str_squeeze(str_get_field(strs, 2, ","))
  varmod   = str_squeeze(str_get_field(strs, 3, ","))
  varreg   = str_squeeze(str_get_field(strs, 4, ","))
  varobs   = str_squeeze(str_get_field(strs, 5, ","))
  nvars    = dimsizes(varnam)
  scenario = "Historical"

  seas     = (/"ANN","DJF","MAM","JJA","SON"/)
  nseas    = dimsizes(seas)

  nens     = 1
  vrmse    = new((/nvars,nseas,nens/),float)

  do i = 0,nseas-1,1
 
   ;read diag data
   filname = run_dir + seas(i) + "_metrics_table.csv"
   lines    = asciiread(filname,-1,"string")
   nlines   = dimsizes(lines)  ; First 3 lines are a header
   delim    = ","

   str1      = str_split(lines(0),delim)
   diag_nam  = str_split(lines(0),delim) 
   indx      = ind(diag_nam.eq."RMSE")
   rmse      = str_get_field(lines(1:),indx+1,delim)
   diag_unt  = str_get_field(lines(2:),2,delim)
   tmp_var   = str_get_field(lines(1:),1,delim)
   diag_var  = new(dimsizes(tmp_var),string)
   diag_reg  = new(dimsizes(tmp_var),string)
   diag_obs  = new(dimsizes(tmp_var),string)
   do kk = 0,dimsizes(tmp_var)-1,1
    tmpstr = str_split(tmp_var(kk)," ")
    diag_var(kk)  = tmpstr(0)
    diag_reg(kk)  = tmpstr(1)
    if(dimsizes(tmpstr).ge.3) then 
      diag_obs(kk)  = tmpstr(2)
    end if 
   end do 
   delete([/tmp_var/])
   do j = 0,nvars-1,1
      indv = ind(diag_var.eq.varmod(j).and.diag_obs.eq.varobs(j).and.diag_reg.eq.varreg(j))
     ;print(varmod(j) + " " + varnam(j))
     ;print(diag_var(indv)+ " " +varnam(j)+ " " + varmod(j))
      print(diag_obs(indv)+" " +diag_var(indv)+ " " +varnam(j)+ " " + varmod(j) + " " + rmse(indv))
      vrmse(j,i,:)= tofloat(rmse(indv(0)))
      delete([/indv/])
   end do 
   delete([/indx,rmse/])
  end do 

  vrmse!0    = "var"
  vrmse!1    = "sea"
  vrmse!2    = "exp"
  expstr!0  = "exp"
  seas!0     = "sea"
  varnam!0   = "var"
  
  outdir = diro
  system("if ! test -d " + outdir +" ; then mkdir -p " + outdir + " ; fi")
  fout1 = outdir+expstr+".rmse.seas.clim."+scenario+"."+runtim+".nc"
  system("rm -rvf " + fout1)
  setfileoption("nc","Format","netcdf4")
  z1 = addfile(fout1,"cw")
  z1->varnam = varnam
  z1->expnam = expstr
  z1->season = seas
  z1->rmse   = vrmse(:,:,:)

end 

