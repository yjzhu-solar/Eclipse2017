pro eis_merg_raster_170821_getwindata

n_window = 23
n_raster = 30

filenames = file_search('../src/EIS/level1/SPCH/eis_l1*')
for ii = 0, sizeof(filenames) - 1 do begin
    for jj = 0, n_window - 1 do begin
        wd = eis_getwindata(filenames[ii],jj)
        wd_shift = eis_shift_spec(wd)  
        save,filename = "../sav/EIS/SPCH/" + string(jj + 1,format='(I02)') + $ 
            "_" + wd_shift.line_id + "_tilt_corr.sav",wd_shift

    end
end

end