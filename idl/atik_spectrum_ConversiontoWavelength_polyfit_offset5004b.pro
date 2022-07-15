pro atik_spectrum_ConversiontoWavelength_polyfit_offset5004b

    print, ' atik_spectrum_ConversionToWavelength_polyfit_________________________'

    fp = fltarr(200)
    yp = fltarr(200)
    xu = fltarr(200)
    yu = fltarr(200)
    xd = fltarr(200)
    yd = fltarr(200)
    yd0 = fltarr(200)
    xo = intarr(200)
    xe = strarr(200)


    device, decomposed = 0
    erase
    !p.background = 255 
    !p.color = 0

    linehoriz = fltarr(1391)
    
    filename = dialog_pickfile(xfilter=['*.asc'],/mult)
    if filename[0] eq '' then return 

    cd, file_dirname(filename)
    fnamesub = filename
    subname = strarr(100)
    len = strlen(fnamesub)
    print, 'file:', fnamesub, ' length:',len
    subname = strsplit(fnamesub, '\', count=subcount, /extract)
    print, 'subcount:', subcount
    for i = 0, subcount - 1 do print, subname[i]
    fnamesb = subname[subcount - 1]
    print,'fnamesb:',fnamesb

    string1 = '                                '
    string2 = '                                '
    string3 = '                                '
    string4 = '                                '
    string5 = '                                '
    element = '                                '

    help,element

    close, 1
    openr, 1, filename
    readf, 1, string1
    print, string1
    readf, 1, string2
    print,string2
    readf, 1, string3
    print,string3
    readf, 1, string4
    print,string4 
    readf, 1, string5
    print,string5

    loadct, 38
    window, 1, xpos = 1365, ypos = 0, xsize = 1270, ysize = 1000, colors =black, title='redueced wavelength against pixel number'

    lammin = 5000
    lammax = 5600
    xcal = [0,1400]
    ycal = [lammin, lammax]
    plot, xcal, ycal, /xst,/yst, yrange = [lammin, lammax], xrange=[0,1400], $
    /nodata, title='wavelength calibration green channel: ' + fnamesb

    baseord = 60
    n = 0
    k = 0
    l = 1
    m = 1
    colr = 140

    readf, 1, format = '(I6,I6,I6,F10.2,I4,A8)',lfds, colb, hpos, xl, ord, element
    print, format = '(I6,I6,I6,F10.2,I4,A8)',lfds, colb, hpos, xl, ord, element
    lfd = abs(lfds)

    if lfd gt 999 then goto, cont1

    if ord lt 100 then begin
        if lfd eq 3 then begin
            xu[l] = hpos
            prinit,'++',l,xu[l]
            l = l + 1
        endif
    endif
    



end