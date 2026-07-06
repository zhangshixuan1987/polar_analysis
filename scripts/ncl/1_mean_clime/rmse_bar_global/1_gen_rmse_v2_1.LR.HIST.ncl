;-----------------------------------------------------------------
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

begin

  dat_dir  = "/lcrc/group/e3sm/ac.szhang/acme_scratch/e3sm_project/E3SMv21_testings/"
  diro     = dat_dir +"paper_material/fig_data/rmse_global/"

  strs     = asciiread("var_list.txt",-1,"string")
  varnam   = str_squeeze(str_get_field(strs, 1, ","))
  varunt   = str_squeeze(str_get_field(strs, 2, ","))
  varmod   = str_squeeze(str_get_field(strs, 3, ","))
  varreg   = str_squeeze(str_get_field(strs, 4, ","))
  varobs   = str_squeeze(str_get_field(strs, 5, ","))
  nvars    = dimsizes(varnam)
  scenario = "historical"

  seas     = (/"ANN","DJF","MAM","JJA","SON"/)
  nseas    = dimsizes(seas)

  expstr   = "v2_1.LR.historical"
  enstr    = (/"0101","0151","0201","0251","0301"/)
  runtim   = "1985-2014"
  nens     = dimsizes(enstr)
  vrmse    = new((/nvars,nseas,nens/),float)

  do k = 0,nens-1,1
    runnam   = expstr+"_"+enstr(k)
    run_dir  = dat_dir +"paper_material/fig_data/rmse_global/"+runnam+"/"
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
      do j = 0,nvars-1,1
        indv = ind(diag_var.eq.varmod(j).and.diag_obs.eq.varobs(j).and.diag_reg.eq.varreg(j))
       ;print(varmod(j) + " " + varnam(j))
       ;print(diag_var(indv)+ " " +varnam(j)+ " " + varmod(j))
        print(diag_obs(indv)+" " +diag_var(indv)+ " " +varnam(j)+ " " + varmod(j) + " " + rmse(indv))
        vrmse(j,i,k)= tofloat(rmse(indv(0)))
        delete([/indv/])
      end do 
      delete([/indx,tmp_var,rmse,filname,lines,delim,diag_nam,str1,diag_unt,tmp_var,diag_var,diag_reg,diag_obs/])
    end do 
  end do 

  vrmse!0    = "var"
  vrmse!1    = "sea"
  vrmse!2    = "exp"
  expstr!0   = "exp"
  seas!0     = "sea"
  varnam!0   = "var"
  
  outdir = diro
  system("if ! test -d " + outdir +" ; then mkdir -p " + outdir + " ; fi")
  fout1 = outdir+expstr+".rmse.seas.clim."+scenario+"."+runtim+".nc"
  system("rm -rvf " + fout1)
  setfileoption("nc","Format","netcdf4")
  z1 = addfile(fout1,"cw")
  z1->varnam = varnam
  z1->expnam = expstr+"_"+enstr
  z1->season = seas
  z1->rmse   = vrmse(:,:,:)

end 

