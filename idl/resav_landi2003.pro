pro resav_landi2003
    temp1=6.0+0.01*[15,15,14,14,14,13,13,13,14,13,13,14,13,14,14,13,14,13,13,14,13,13,13,14,13,13,14,12,12,13]

    restore,'../sav/Landi2003/vnth_north.save',/verb
    res=get_sun('23-Feb-1999 23:59',sd=sd)      ;  Raggio solare a terra per tale data
    print,soho_fac('23-Feb-1999 23:59')         ;  Correzione per tenere conto che siamo alla distanza di SOHO
    solrad=sd*soho_fac('23-Feb-1999 23:59')     ;  Raggio solare visto da SOHO, in arcsec dal centro del sole

    sunrad=697000.                              ;  Km, Di suo
    arc2km=sunrad/solrad                        ;  Km
    slitcen=1148.                               ;  Arcsec, altezza del centro della slit sul lembo
    incl=1.0                                    ;  seno (o coseno) per tener conto dell'inclinazione della slit sulla radiale
    slitmin=slitcen-150.*incl-solrad            ;  Reale altezza minima raggiunta dalla slit
    slitmax=slitcen+150.*incl-solrad            ;  Reale altezza massima raggiunta dalla slit
    rnorth=findgen(30)*arc2km*10/sunrad+slitmin*arc2km/sunrad+1

    north=fltarr(9,30)
    dnorth=fltarr(9,30)
    north(0,*)=vo6 & north(1,*)=vne8 & north(2,*)=vna9 & north(3,*)=vmg9 & north(4,*)=vmg10
    north(5,*)=vsi11 & north(6,*)=vs10 & north(7,*)=vfe11 & north(8,*)=vfe12
    dnorth(0,*)=dvo6 & dnorth(1,*)=dvne8 & dnorth(2,*)=dvna9 & dnorth(3,*)=dvmg9 & dnorth(4,*)=dvmg10
    dnorth(5,*)=dvsi11 & dnorth(6,*)=dvs10 & dnorth(7,*)=dvfe11 & dnorth(8,*)=dvfe12
    save, filename="../sav/Landi2003/vnth_north_withr.save", vo6, vne8, vna9, vmg9, vmg10, vsi11, vs10, vfe11, vfe12, $
            dvo6, dvne8, dvna9, dvmg9, dvmg10, dvsi11, dvs10, dvfe11, dvfe12, rnorth, north, dnorth, temp1

end