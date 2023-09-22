pro read_airspec_ir


fname = 'airspec_20170821_182352_obs03_IR3.fits' ; 3 micron file
;fname = 'airspec_20170821_182352_obs03_IR4.fits' ; 4 micron file

; read the IR data (3 micron or 4 micron)
fits_read, fname, data, header ; intensity (DN)
fits_read, fname, x, x_header, EXTEN = 1 ; solar x coordinate (arcsec)
fits_read, fname, y, y_header, EXTEN = 2 ; solar y coordinate (arcsec)
fits_read, fname, wave, wave_header, EXTEN = 3 ; wavelength (Angstrom)
fits_read, fname, table_data, table_header, EXTEN = 4 ; table of metadata and time

; decode the table data
unix_time = tbget(table_header, table_data, 1)
exp_time  = tbget(table_header, table_data, 2)
fpa_temp  = tbget(table_header, table_data, 3)
start_row = tbget(table_header, table_data, 4)

; average the data, solar position, and wavelength in time
x_mean = mean(x,DIMENSION=1)
y_mean = mean(y,DIMENSION=1)
wave_mean = mean(wave,DIMENSION=1)
data_mean = mean(data,DIMENSION=3)

; pull out the mean spectrum at the lunar limb
limb_row = sxpar(header, 'LIMB_ROW')
spec = mean(data_mean(*,indgen(15,START=limb_row-7)),DIMENSION=2,/NAN)

; plot the temporal mean image
fig1 = image(data_mean,wave_mean/10000,x_mean,AXIS_STYLE=1,ASPECT_RATIO=0,XTITLE='First order wavelength (micron)',YTITLE='Solar x (arcsec)')
plot,wave_mean/10000, spec,XTITLE='First order wavelength (micron)',YTITLE='Intensity (DN)'

end
