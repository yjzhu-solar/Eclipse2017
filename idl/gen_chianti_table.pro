pro gen_chianti_table

temp = 4+0.05*findgen(100)
dens=6+findgen(8)
n_heights = 21
height = findgen(n_heights)/20d + 1d

FeXIV_emiss_array = dblarr(100, 8, n_heights)
FeX_emiss_array = dblarr(100, 8, n_heights)

for ii = 0, n_heights - 1 do begin
    FeXIV_data = emiss_calc('fe_14',temp=temp,dens=dens,radtemp=5770d,rphot=height[ii])
    FeXIV_id = where(FeXIV_data.lambda eq 5304.477)
    FeXIV_emiss_array[*,*,ii] = (FeXIV_data.em)[*,*,FeXIV_id]
    FeX_data = emiss_calc('fe_10',temp=temp,dens=dens,radtemp=5770d,rphot=height[ii])
    FeX_id = where(FeX_data.lambda eq 6376.29)
    FeX_emiss_array[*,*,ii] = (FeX_data.em)[*,*,FeX_id]
end

save,filename = "../sav/AWSoM/chianti_table/FeXIV_FeX_emiss.sav",FeXIV_emiss_array,FeX_emiss_array,temp,dens,height

end