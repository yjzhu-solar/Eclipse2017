pro gen_chianti_table

n_temps = 100
temp = 4+0.05*findgen(n_temps)
dens=5+findgen(9)
n_heights = 21
height = findgen(n_heights)/20d + 1d

FeXIV_emiss_array = dblarr(100, 9, n_heights)
FeX_emiss_array = dblarr(100, 9, n_heights)
; FeXIII_emiss_array = dblarr(100, 9, n_heights)
; FeXI_emiss_array = dblarr(100, 9, n_heights)
; FeXI_emiss_array = dblarr(100, 9, n_heights)
; FeXV_emiss_array = dblarr(100, 9, n_heights)
; SVIII_emiss_array = dblarr(100, 9, n_heights)
; SVIII_emiss_array = dblarr(100, 9, n_heights)

read_ioneq,!ioneq_file,temp_grid,ioneq_grid
ion_frac_FeXIV = interpol(ioneq_grid[*,25,13],temp_grid,temp,/spline)
ion_frac_FeX = interpol(ioneq_grid[*,25,9],temp_grid,temp,/spline)

abund_file = "/usr/local/ssw/packages/chianti/dbase/abundance/sun_coronal_1992_feldman_ext.abund"
read_abund,abund_file,abundances,ref
Fe_abund = abundances[25]

for ii = 0, n_heights - 1 do begin
    FeXIV_data = emiss_calc('fe_14',temp=temp,dens=dens,radtemp=5770d,rphot=height[ii])
    FeXIV_id = where(FeXIV_data.lambda eq 5304.477)
    FeXIV_emiss_array[*,*,ii] = (FeXIV_data.em)[*,*,FeXIV_id]*0.83*Fe_abund/4/!PI

    for jj = 0, n_temps - 1 do begin
        FeXIV_emiss_array[jj,*,ii] = FeXIV_emiss_array[jj,*,ii]*ion_frac_FeXIV[jj]
    endfor 

    FeX_data = emiss_calc('fe_10',temp=temp,dens=dens,radtemp=5770d,rphot=height[ii])
    FeX_id = where(FeX_data.lambda eq 6376.29)
    FeX_emiss_array[*,*,ii] = (FeX_data.em)[*,*,FeX_id]*0.83*Fe_abund/4/!PI
    
    for jj = 0, n_temps - 1 do begin
        FeX_emiss_array[jj,*,ii] = FeX_emiss_array[jj,*,ii]*ion_frac_FeX[jj]
    endfor 
endfor

save,filename = "../sav/AWSoM/chianti_table/FeXIV_FeX_emiss.sav",FeXIV_emiss_array,FeX_emiss_array,temp,dens,height

end