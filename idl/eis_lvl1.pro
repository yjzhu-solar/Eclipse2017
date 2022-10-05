pro eis_lvl1

;filenames = file_search('../raw_data/north_pole/20071116/eis_l0_*.fits')
;output_dir = "../raw_data/north_pole/20071116/level1/"

; filenames = file_search('../src/EIS/level0/SPCH/eis_l0*')
; output_dir = "../src/EIS/level1/SPCH/"

filenames = file_search('../src/EIS/level0/EastOffLimbFullSpectra/eis_l0*')
output_dir = "../src/EIS/level1/EastOffLimbFullSpectra/"

for i = 0, sizeof(filenames) - 1 do begin
    eis_prep, filenames[i],outdir=output_dir,/hkwavecorr, /default, /quiet, /save, /retain
end
print,"end"
end