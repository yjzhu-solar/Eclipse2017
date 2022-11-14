pro cal_eis_radcorr

wvl = [184.537, (257.259+257.261)/2.,(188.216+188.299)/2.,(257.547+257.554)/2., $
(186.854+186.887)/2., 192.394, 195.119, 258.374, 261.056,264.788]

for ii = 0, n_elements((wvl)) - 1 do begin
    print,"HPW",wvl[ii],eis_recalibrate_intensity('21-Aug-2017',wvl[ii],1)
    print,"GDZ",wvl[ii],eis_recalibrate_intensity('21-Aug-2017',wvl[ii],1,/gdz)
    print,"---------------------------------"
endfor

end
