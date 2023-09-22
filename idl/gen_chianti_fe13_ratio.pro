pro gen_chianti_fe13_ratio

temp = 6.25
n_dens = 21
dens=6+findgen(n_dens)*0.25d
n_heights = 21
height = findgen(n_heights)*0.02d + 1d

FeXIII_1074_1079_ratio = fltarr(n_dens,n_heights)


for ii = 0, n_heights - 1 do begin
    FeXIII_data = emiss_calc('fe_13',temp=temp,dens=dens,radtemp=5770d,rphot=height[ii])
    FeXIII_10747_id = where(FeXIII_data.lambda eq 10749.105)
    FeXIII_10798_id = where(FeXIII_data.lambda eq 10800.770)
    FeXIII_1074_1079_ratio[*,ii] = (FeXIII_data.em)[0,*,FeXIII_10747_id]/(FeXIII_data.em)[0,*,FeXIII_10798_id]
endfor

save,filename = "../sav/CoMP/FeXIII_1074_1079_ratio.sav",FeXIII_1074_1079_ratio,dens,height,temp

end