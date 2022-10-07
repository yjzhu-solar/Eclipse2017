import matplotlib.pyplot as plt
import astropy.units as u
import eispac

if __name__ == '__main__':
    # input data and template files
    data_filepath = '../../src/EIS/level1/EastOffLimbFullSpectra/eis_20170821_205401.data.h5'
    fe_10_184_template_filepath = '../../sav/EIS/fit_template/fe_10_184_536.1c.template.h5'
    fe_14_264_template_filepath = '../../sav/EIS/fit_template/fe_14_264_787.1c.template.h5'

    fe_10_184_tmplt = eispac.read_template(fe_10_184_template_filepath)
    data_cube = eispac.read_cube(data_filepath, fe_10_184_tmplt.central_wave)
    fe_10_184_fit_res = eispac.fit_spectra(data_cube, fe_10_184_tmplt, ncpu='max')

    fe_14_264_tmplt = eispac.read_template(fe_14_264_template_filepath)
    data_cube = eispac.read_cube(data_filepath, fe_14_264_tmplt.central_wave)
    fe_14_264_fit_res = eispac.fit_spectra(data_cube, fe_14_264_tmplt, ncpu='max')

    for fit_res in (fe_10_184_fit_res,fe_14_264_fit_res):
        eispac.save_fit(fit_res, save_dir='../../sav/EIS/EQSPY/eispac_fit/')
