pro gen_chianti_table

n_temps = 100
n_dens = 20
temp = 4+0.05*findgen(n_temps)
dens=5+findgen(n_dens)/2d
n_heights = 21
height = findgen(n_heights)/20d + 1d

FeXIV_emiss_array = dblarr(n_temps, n_dens, n_heights)
FeX_emiss_array = dblarr(n_temps, n_dens, n_heights)
FeXIII_10747_emiss_array = dblarr(n_temps, n_dens, n_heights)
FeXIII_10798_emiss_array = dblarr(n_temps, n_dens, n_heights)
FeXI_emiss_array = dblarr(n_temps, n_dens, n_heights)
FeXV_emiss_array = dblarr(n_temps, n_dens, n_heights)
SVIII_emiss_array = dblarr(n_temps, n_dens, n_heights)
SXII_emiss_array = dblarr(n_temps, n_dens, n_heights)
FeXIII_10747_emiss_array = dblarr(n_temps, n_dens, n_heights)
FeXIII_10798_emiss_array = dblarr(n_temps, n_dens, n_heights)
NiXV_6703_emiss_array = dblarr(n_temps, n_dens, n_heights)
NiXV_8026_emiss_array = dblarr(n_temps, n_dens, n_heights)

read_ioneq,!ioneq_file,temp_grid,ioneq_grid
ion_frac_FeXIV = interpol(ioneq_grid[*,25,13],temp_grid,temp,/spline)
ion_frac_FeX = interpol(ioneq_grid[*,25,9],temp_grid,temp,/spline)
ion_frac_FeXI = interpol(ioneq_grid[*,25,10],temp_grid,temp,/spline)
ion_frac_FeXIII = interpol(ioneq_grid[*,25,12],temp_grid,temp,/spline)
ion_frac_FeXV = interpol(ioneq_grid[*,25,14],temp_grid,temp,/spline)
ion_frac_SVIII = interpol(ioneq_grid[*,15,7],temp_grid,temp,/spline)
ion_frac_SXII = interpol(ioneq_grid[*,15,11],temp_grid,temp,/spline)
ion_frac_NiXV = interpol(ioneq_grid[*,27,14],temp_grid,temp,/spline)

abund_file = "/usr/local/ssw/packages/chianti/dbase/abundance/sun_coronal_1992_feldman_ext.abund"
read_abund,abund_file,abundances,ref
Fe_abund = abundances[25]
S_abund = abundances[15]
Ni_abund = abundances[27]

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

    FeXI_data = emiss_calc('fe_11',temp=temp,dens=dens,radtemp=5770d,rphot=height[ii])
    FeXI_id = where(FeXI_data.lambda eq 7894.031)
    FeXI_emiss_array[*,*,ii] = (FeXI_data.em)[*,*,FeXI_id]*0.83*Fe_abund/4/!PI

    for jj = 0, n_temps - 1 do begin
        FeXI_emiss_array[jj,*,ii] = FeXI_emiss_array[jj,*,ii]*ion_frac_FeXI[jj]
    endfor 

    FeXIII_data = emiss_calc('fe_13',temp=temp,dens=dens,radtemp=5770d,rphot=height[ii])
    FeXIII_10747_id = where(FeXIII_data.lambda eq 10749.105)
    FeXIII_10798_id = where(FeXIII_data.lambda eq 10800.770)
    FeXIII_10747_emiss_array[*,*,ii] = (FeXIII_data.em)[*,*,FeXIII_10747_id]*0.83*Fe_abund/4/!PI
    FeXIII_10798_emiss_array[*,*,ii] = (FeXIII_data.em)[*,*,FeXIII_10798_id]*0.83*Fe_abund/4/!PI

    for jj = 0, n_temps - 1 do begin
        FeXIII_10747_emiss_array[jj,*,ii] = FeXIII_10747_emiss_array[jj,*,ii]*ion_frac_FeXIII[jj]
        FeXIII_10798_emiss_array[jj,*,ii] = FeXIII_10798_emiss_array[jj,*,ii]*ion_frac_FeXIII[jj]
    endfor 

    FeXV_data = emiss_calc('fe_15',temp=temp,dens=dens,radtemp=5770d,rphot=height[ii])
    FeXV_id = where(FeXV_data.lambda eq 7062.147)
    FeXV_emiss_array[*,*,ii] = (FeXV_data.em)[*,*,FeXV_id]*0.83*Fe_abund/4/!PI

    for jj = 0, n_temps - 1 do begin
        FeXV_emiss_array[jj,*,ii] = FeXV_emiss_array[jj,*,ii]*ion_frac_FeXV[jj]
    endfor 

    SVIII_data = emiss_calc('s_8',temp=temp,dens=dens,radtemp=5770d,rphot=height[ii])
    SVIII_id = where(SVIII_data.lambda eq 9916.700)
    SVIII_emiss_array[*,*,ii] = (SVIII_data.em)[*,*,SVIII_id]*0.83*S_abund/4/!PI

    for jj = 0, n_temps - 1 do begin
        SVIII_emiss_array[jj,*,ii] = SVIII_emiss_array[jj,*,ii]*ion_frac_SVIII[jj]
    endfor 

    SXII_data = emiss_calc('s_12',temp=temp,dens=dens,radtemp=5770d,rphot=height[ii])
    SXII_id = where(SXII_data.lambda eq 7613.073)
    SXII_emiss_array[*,*,ii] = (SXII_data.em)[*,*,SXII_id]*0.83*S_abund/4/!PI

    for jj = 0, n_temps - 1 do begin
        SXII_emiss_array[jj,*,ii] = SXII_emiss_array[jj,*,ii]*ion_frac_SXII[jj]
    endfor 

    NiXV_data = emiss_calc('ni_15',temp=temp,dens=dens,radtemp=5770d,rphot=height[ii])
    NiXV_6703_id = where(NiXV_data.lambda eq 6073.536)
    NiXV_8026_id = where(NiXV_data.lambda eq 8026.326)
    NiXV_6703_emiss_array[*,*,ii] = (NiXV_data.em)[*,*,NiXV_6703_id]*0.83*Ni_abund/4/!PI
    NiXV_8026_emiss_array[*,*,ii] = (NiXV_data.em)[*,*,NiXV_8026_id]*0.83*Ni_abund/4/!PI

    for jj = 0, n_temps - 1 do begin
        NiXV_6703_emiss_array[jj,*,ii] = NiXV_6703_emiss_array[jj,*,ii]*ion_frac_NiXV[jj]
        NiXV_8026_emiss_array[jj,*,ii] = NiXV_8026_emiss_array[jj,*,ii]*ion_frac_NiXV[jj]
    endfor 


endfor

; save,filename = "../sav/AWSoM/chianti_table/FeXIV_FeX_emiss.sav",FeXIV_emiss_array,FeX_emiss_array,temp,dens,height
    save,filename = "../sav/AWSoM/chianti_table/AWSoM_UCoMP_emiss.sav",FeXIV_emiss_array, FeX_emiss_array, $
    FeXI_emiss_array, FeXIII_10747_emiss_array, FeXIII_10798_emiss_array, FeXV_emiss_array, SVIII_emiss_array, $
    SXII_emiss_array, NiXV_6703_emiss_array, NiXV_8026_emiss_array temp, dens, height

end