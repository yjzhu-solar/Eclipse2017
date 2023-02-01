import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib import ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, 
                                AutoLocator, MaxNLocator)
from matplotlib import rcParams
from matplotlib import patches
import scipy.io
import astropy.constants as const
import juanfit
from juanfit import SpectrumFitSingle,SpectrumFitRow, gaussian
import copy
from scipy import interpolate
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate import simps
from scipy.signal import find_peaks
rcParams['axes.linewidth'] = 1.5
import cmcrameri.cm as cmcm
import h5py
from astropy.visualization import ZScaleInterval, ImageNormalize, LogStretch, AsymmetricPercentileInterval,\
         ManualInterval, SqrtStretch
import os

np.seterr(all="ignore")


m_p = const.m_p.cgs.value
k_b = const.k_B.cgs.value
c = const.c.cgs.value
rs = const.R_sun.cgs.value
amu = const.u.cgs.value

def plot_colorbar(im, ax, width="3%", height="100%",loc="lower left",fontsize=14):
    clb_ax = inset_axes(ax,width=width,height=height,loc=loc,
                bbox_to_anchor=(1.02, 0., 1, 1),
                 bbox_transform=ax.transAxes,
                 borderpad=0)
    clb = plt.colorbar(im,pad = 0.05,orientation='vertical',ax=ax,cax=clb_ax)
    clb_ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    clb_ax.yaxis.get_offset_text().set_fontsize(fontsize)
    clb_ax.tick_params(labelsize=fontsize)
    return clb, clb_ax

def create_limb_circle(rsun):
    return patches.Circle((0,0),rsun,linewidth=2,edgecolor="grey",
                        facecolor="none",alpha=0.6,ls="--")

def calc_profile(emiss_table,dens,temp,height,height_grid,dens_grid,te_grid,tp_grid,
                    vlos_grid, vnt2_grid, wvl_grid, rest_wvl,ion_mass_amu,
                    off_limb_yz_index, on_disk_yz_index,pos_xindex=150,method="linear",return_emiss_veff=False,
                    p_to_e=0.83):
    emiss_func = interpolate.RegularGridInterpolator(points=(height,dens,temp),
                    values=emiss_table,
                    bounds_error=False,method=method)
    emiss_box = emiss_func((height_grid[:,:,:], np.log10(dens_grid[:,:,:])/p_to_e, np.log10(te_grid[:,:,:])))* \
        (dens_grid/p_to_e)*0.01*const.R_sun.cgs.value

    veff_grid = np.sqrt(2*k_b*tp_grid/ion_mass_amu/amu + vnt2_grid)
    line_wvl_grid = rest_wvl*(1 - vlos_grid/c)
    fwhm_grid = np.sqrt(4*np.log(2))*rest_wvl*veff_grid/c

    line_profile_grid = gaussian(wvl=wvl_grid[np.newaxis,np.newaxis,np.newaxis,:], line_wvl=line_wvl_grid[:,:,:,np.newaxis],
                int_total=emiss_box[:,:,:,np.newaxis], fwhm=fwhm_grid[:,:,:,np.newaxis])

    profile_integrated = np.nansum(line_profile_grid[:,:,:,:],axis=2)
    profile_integrated[on_disk_yz_index] = profile_integrated[on_disk_yz_index] - \
        np.nansum(line_profile_grid[:,:,:pos_xindex,:],axis=2)[on_disk_yz_index]

    if return_emiss_veff:
        return profile_integrated, emiss_box, veff_grid
    else:
        return profile_integrated

def fit_all_spec(syn_profiles, wvl_grid,line_number,line_wvl_init,int_max_init,fwhm_init):
    fit_matrix = np.full_like(syn_profiles[:,:,:4],np.nan)
    err_matrix = np.full_like(syn_profiles[:,:,:4],np.nan)
    for ii in range(syn_profiles.shape[1]):
        fit_row_model = SpectrumFitRow(syn_profiles[:,ii,:],wvl_grid,line_number=line_number,line_wvl_init=line_wvl_init,
                                        int_max_init=int_max_init,fwhm_init=fwhm_init)

        try:
            fit_row_model.run_lse()
        except RuntimeError:
            pass

        fit_matrix[:,ii,0] = fit_row_model.line_wvl_fit[:,0]
        err_matrix[:,ii,0] = fit_row_model.line_wvl_err[:,0]
        fit_matrix[:,ii,1] = fit_row_model.int_total_fit[:,0]
        err_matrix[:,ii,1] = fit_row_model.int_total_err[:,0]
        fit_matrix[:,ii,2] = fit_row_model.fwhm_fit[:,0]
        err_matrix[:,ii,2] = fit_row_model.fwhm_err[:,0]
        fit_matrix[:,ii,3] = fit_row_model.int_cont_fit[:]
        err_matrix[:,ii,3] = fit_row_model.int_cont_err[:]

    return fit_matrix, err_matrix


# awsom_data_set = scipy.io.readsav(r'../../sav/AWSoM/syn_fit/box_run0017_run03_75/box_run0017_run03_75.sav',verbose = False,python_dict=True)
# awsom_data_set = scipy.io.readsav(r'../../sav/AWSoM/syn_fit/box_run0019_run03_75/box_run0019_run03_75.sav',verbose = False,python_dict=True)
awsom_data_set = scipy.io.readsav(r'../../sav/AWSoM/syn_fit/box_run0021_run01_75_5th/box_run0021_run01_75_5th.sav',verbose = False,python_dict=True)

p_e_ratio = 0.83
awsom_x = awsom_data_set['x'][0,0,0,:]
awsom_y = awsom_data_set['x'][1,0,:,0]
awsom_z = awsom_data_set['x'][2,:,0,0]
awsom_x_grid = awsom_data_set['x'][0,:,:,:]
awsom_y_grid = awsom_data_set['x'][1,:,:,:]
awsom_z_grid = awsom_data_set['x'][2,:,:,:]
rho = awsom_data_set['w'][0,:,:,:]
n = rho/m_p
#n = np.nan_to_num(n,nan=0)
# ux = awsom_data_set['w'][1,:,:,:]
# uy = awsom_data_set['w'][2,:,:,:]
# uz = awsom_data_set['w'][3,:,:,:]
# bx = awsom_data_set['w'][4,:,:,:]
# by = awsom_data_set['w'][5,:,:,:]
# bz = awsom_data_set['w'][6,:,:,:]
I01 = awsom_data_set['w'][7,:,:,:]
I02 = awsom_data_set['w'][8,:,:,:]
p = awsom_data_set['w'][9,:,:,:]
t_p = p/n/k_b
#t = np.nan_to_num(t,nan=1e3)
p_e = awsom_data_set['w'][10,:,:,:]
t_e = p_e/n/k_b
# t_e = np.nan_to_num(t_e,nan=1e3)
# n = np.nan_to_num(n,nan=1)
# n = n + 1
p_par = awsom_data_set['w'][11,:,:,:]
t_par = p_par/n/k_b
t_perp = (3*t_p - t_par)/2.
# bx_rot = np.zeros_like(bx)
# by_rot = np.zeros_like(by)
# bz_rot = np.zeros_like(bz)
# ux_rot = np.zeros_like(ux)
# uy_rot = np.zeros_like(uy)
# uz_rot = np.zeros_like(uz)

rot_DD = np.resize(awsom_data_set["param"],(3,3))
b_vec = awsom_data_set['w'][4:7,:,:,:]
b_tot = np.sqrt(np.sum(np.square(b_vec),axis=0))
u_vec = awsom_data_set['w'][1:4,:,:,:]
u_tot = np.sqrt(np.sum(np.square(u_vec),axis=0))
bx_rot, by_rot, bz_rot = np.einsum('ij,jklm->iklm',rot_DD,b_vec)
ux_rot, uy_rot, uz_rot = np.einsum('ij,jklm->iklm',rot_DD,u_vec)
# for ii in range(ux.shape[0]):
#     for jj in range(ux.shape[1]):
#         for kk in range(ux.shape[2]):
#             bx_rot[ii,jj,kk],by_rot[ii,jj,kk],bz_rot[ii,jj,kk] = np.matmul([bx[ii,jj,kk],by[ii,jj,kk],bz[ii,jj,kk]],rot_DD.T)
#             ux_rot[ii,jj,kk],uy_rot[ii,jj,kk],uz_rot[ii,jj,kk] = np.matmul([ux[ii,jj,kk],uy[ii,jj,kk],uz[ii,jj,kk]],rot_DD.T)


chianti_emiss_tbl = scipy.io.readsav("../../sav/AWSoM/chianti_table/AWSoM_UCoMP_emiss_4Rsun.sav",verbose=True,python_dict=True)
awsom_on_disk_yz_index = np.where(np.sqrt(awsom_y_grid[:,:,150]**2 + awsom_z_grid[:,:,150]**2) <= 1) 
awsom_off_limb_yz_index = np.where(np.sqrt(awsom_y_grid[:,:,150]**2 + awsom_z_grid[:,:,150]**2) > 1) 

height_grid = np.sqrt(awsom_x_grid**2 + awsom_y_grid**2 + awsom_z_grid**2)
cos_alpha = bx_rot/b_tot
sin_alpha = np.sqrt(1 - cos_alpha**2)

vnt2 = 0.5*(I01 + I02)/m_p/n*sin_alpha**2 
tlos = t_perp*sin_alpha**2 + t_par*cos_alpha**2

FeXIV_wvl_grid = np.linspace(530.,530.6,26)
FeXIV_rest_wvl = 530.286
Fe_amu = 55.85
FeXIV_int_max_init = 100.
FeXIV_fwhm_init = 0.1
FeXIV_emiss_array_name = "fexiv_emiss_array"

FeX_wvl_grid = np.linspace(637.15,637.75,20)
FeX_rest_wvl = 637.451
FeX_int_max_init = 100.
FeX_fwhm_init = 0.1
FeX_emiss_array_name = "fex_emiss_array"

FeXIII_10747_wvl_grid = np.linspace(1074.1,1075.3,40)
FeXIII_10747_rest_wvl = 1074.68
FeXIII_10747_int_max_init = 100.
FeXIII_10747_fwhm_init = 0.2
FeXIII_10747_emiss_array_name = "fexiii_10747_emiss_array"

# save_path = "../../sav/AWSoM/syn_fit/box_run0017_run03_75/"
# save_path = "../../sav/AWSoM/syn_fit/box_run0019_run03_75/"
save_path = "../../sav/AWSoM/syn_fit/box_run0021_run01_75_5th/"

emiss_array_names = [FeXIV_emiss_array_name, FeX_emiss_array_name,
                    FeXIII_10747_emiss_array_name]
wvl_grids = [FeXIV_wvl_grid, FeX_wvl_grid, FeXIII_10747_wvl_grid]
rest_wvls = [FeXIV_rest_wvl, FeX_rest_wvl, FeXIII_10747_rest_wvl]
ion_amus = [Fe_amu, Fe_amu, Fe_amu]
int_max_inits = [FeXIV_int_max_init, FeX_int_max_init, 
            FeXIII_10747_int_max_init]
fwhm_inits = [FeXIV_fwhm_init, FeX_fwhm_init, 
            FeXIII_10747_fwhm_init]
filenames = ["FeXIV_530_synspec_emiss.h5", "FeX_637_synspec_emiss.h5",
            "FeXIII_1074_synspec_emiss.h5"]


for ii, (emiss_array_, wvl_grid_, rest_wvl_, ion_amu_, int_max_init_, fwhm_init_, filename_) in \
    enumerate(zip(emiss_array_names, wvl_grids, rest_wvls, ion_amus, int_max_inits,
                fwhm_inits, filenames)):
    syn_profiles, emiss_box, veff_box = calc_profile(chianti_emiss_tbl[emiss_array_], chianti_emiss_tbl["dens"],
                        chianti_emiss_tbl["temp"],chianti_emiss_tbl["height"],height_grid, n, t_e,
                        tlos,ux_rot*1e5,vnt2,wvl_grid_,rest_wvl_,ion_amu_,awsom_off_limb_yz_index,
                         awsom_on_disk_yz_index,return_emiss_veff=True)

    fit_matrix, err_matrix = fit_all_spec(syn_profiles, wvl_grid_,line_number=1,line_wvl_init=rest_wvl_,
                                int_max_init=int_max_init_,fwhm_init=fwhm_init_)

    
    with h5py.File(os.path.join(save_path,filename_), 'w') as hf:
        df_syn_profiles = hf.create_dataset("syn_profiles",  data=syn_profiles)
        df_wvl_grid = hf.create_dataset("wvl_grid",  data=wvl_grid_)
        df_emiss_box = hf.create_dataset("emiss_box",  data=emiss_box)
        df_veff_box = hf.create_dataset("veff_box",  data=veff_box/1e5)
        df_vlos_box = hf.create_dataset("vlos_box",  data=ux_rot)
        df_fit_matrix = hf.create_dataset("fit_matrix",  data=fit_matrix)
        df_err_matrix = hf.create_dataset("err_matrix",  data=err_matrix)
        df_awsom_x = hf.create_dataset("awsom_x",  data=awsom_x)
        df_awsom_y = hf.create_dataset("awsom_y",  data=awsom_y)
        df_awsom_z = hf.create_dataset("awsom_z",  data=awsom_z)
        df_syn_profiles.attrs["rest_wvl"] = rest_wvl_
        df_fit_matrix.attrs["description"] = "wvl;int;fwhm;cont"


    








