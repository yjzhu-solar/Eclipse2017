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
import cmasher as cmr
import h5py
from astropy.visualization import ZScaleInterval, ImageNormalize, LogStretch, AsymmetricPercentileInterval,\
         ManualInterval, SqrtStretch
import argparse
import sys
import os
from num2tex import num2tex
from glob import glob

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

def show_fit(line_name, path, veff_min, veff_max, int_scale, int_min, int_max, save_path):
    with h5py.File(os.path.join(path,line_name+"_synspec_emiss.h5"), 'r') as hf:
        syn_profiles = hf['syn_profiles'][:]
        wvl_grid = hf['wvl_grid'][:]
        emiss_box = hf['emiss_box'][:]
        vlos_box = hf['vlos_box'][:]
        veff_box = hf['veff_box'][:]
        fit_matrix =  hf['fit_matrix'][:]
        err_matrix =  hf['err_matrix'][:]
        awsom_x = hf['awsom_x'][:]
        awsom_y = hf['awsom_y'][:]
        awsom_z =  hf['awsom_z'][:]
        rest_wvl = hf['syn_profiles'].attrs['rest_wvl']
    ion_name, wvl_name = line_name.split("_")

    awsom_filename = glob(os.path.join(path,"box*.sav"))
    awsom_data_set = scipy.io.readsav(awsom_filename[0],verbose = False,python_dict=True)
    rho = awsom_data_set['w'][0,:,:,:]
    n = rho/m_p
    p_e = awsom_data_set['w'][10,:,:,:]
    te_box = p_e/n/k_b
    p_e_ratio = 0.83
    ne_box = n/p_e_ratio
    

    if ion_name in ["FeXIV", "FeXIII"]:
        solarx_slice = slice(0,None)
        solary_slice = slice(0,None)
        syn_profiles = syn_profiles[solary_slice,solarx_slice,:]
        emiss_box = emiss_box[solary_slice,solarx_slice,:]
        vlos_box = vlos_box[solary_slice,solarx_slice,:]
        veff_box = veff_box[solary_slice,solarx_slice,:]
        fit_matrix = fit_matrix[solary_slice,solarx_slice,:]
        err_matrix = err_matrix[solary_slice,solarx_slice,:]
        awsom_y = awsom_y[solarx_slice]
        awsom_z = awsom_z[solary_slice]
        te_box = te_box[solary_slice,solarx_slice,:]
        ne_box = ne_box[solary_slice,solarx_slice,:]

    awsom_z_grid, awsom_y_grid, awsom_x_grid = np.meshgrid(awsom_z,awsom_y,awsom_x,indexing="ij")
    
    fig, axes = plt.subplots(1,3,figsize=(11/200*len(awsom_y),3.8/250*len(awsom_z)),constrained_layout=True)
    matrix_mask_TR = np.copy(fit_matrix)
    mask_transition_region_fit = np.where(np.sqrt(awsom_y_grid[:,:,150]**2 + awsom_z_grid[:,:,150]**2)[:,:,np.newaxis]* \
                            np.ones_like(matrix_mask_TR) <= 1.03) 
    matrix_mask_TR[mask_transition_region_fit] = np.nan

    if int_scale == "sqrt":
        strech = SqrtStretch()
    elif int_scale == "log":
        strech = LogStretch()
    
    if (int_min is not None) and (int_max is not None):
        interval = ManualInterval(int_min,int_max)
    else:
        interval = None
    norm_matrix_int_mask_TR = ImageNormalize(matrix_mask_TR[:,:,1],interval=interval,stretch=strech)

    if ion_name == "FeXIV":
        int_cmap = cmr.jungle_r
    else:
        int_cmap = cmcm.lajolla
    
    title = r"{:} \textsc{{{:}}} {:} nm".format(ion_name[:2],ion_name[2:].lower(),wvl_name)
    im_int = axes[0].pcolormesh(awsom_y,awsom_z, matrix_mask_TR[:,:,1],norm=norm_matrix_int_mask_TR,
    cmap=int_cmap, shading="auto",rasterized=True)
    axes[0].set_title(title + "\n" + \
                r"$I_{\rm tot}\ \mathrm{[erg\,s^{-1}\,cm^{-2}\,sr^{-1}]}$",fontsize=14)
    plot_colorbar(im_int, axes[0],width="8%")

    im_vlos = axes[1].pcolormesh(awsom_y,awsom_z, (1 - matrix_mask_TR[:,:,0]/rest_wvl)*c/1e5,vmin=-10,vmax=10,
    cmap=cmcm.vik_r, shading="auto",rasterized=True)
    axes[1].set_title(title + "\n" + \
                r"$v_{\rm LOS}\ \mathrm{[km\,s^{-1}]}$",fontsize=14)
    plot_colorbar(im_vlos, axes[1],width="8%")

    norm_veff = ImageNormalize(matrix_mask_TR[:,:,2]/rest_wvl/np.sqrt(4*np.log(2))*c/1e5, interval=ManualInterval(vmin=veff_min, vmax=veff_max))
    im_veff = axes[2].pcolormesh(awsom_y,awsom_z, matrix_mask_TR[:,:,2]/rest_wvl/np.sqrt(4*np.log(2))*c/1e5,
    cmap=cmcm.batlowK, shading="auto",rasterized=True,norm=norm_veff)
    axes[2].set_title(title + "\n" + \
                r"$v_{\rm eff}\ \mathrm{[km\,s^{-1}]}$",fontsize=14)
    plot_colorbar(im_veff, axes[2],width="8%")

    axes[0].set_ylabel(r"Solar-Y $[R_\odot]$",fontsize=14)
    for ax_ in (axes[1:]):
        ax_.tick_params(labelleft=False)

    for ax_ in axes:
        ax_xlim = ax_.get_xlim()
        ax_ylim = ax_.get_ylim()
        ax_.set_aspect(1)
        ax_.add_patch(create_limb_circle(1))
        ax_.set_xlim(ax_xlim)
        ax_.set_ylim(ax_ylim)
        ax_.tick_params(labelsize=14)
        ax_.set_xlabel(r"Solar-X $[R_\odot]$",fontsize=14)

    GetFitProfile(fig,axes,awsom_y_grid[:,:,0],awsom_z_grid[:,:,0],awsom_x,syn_profiles,emiss_box,title,
                    wvl_grid,rest_wvl,fit_matrix,err_matrix,te_box=te_box,ne_box=ne_box,veff_box=veff_box,vlos_box=vlos_box)
    
    if save_path is not None:
        plt.savefig(fname=save_path, format="pdf", dpi=300)
    else:
        plt.show()

def give_ax_index(target_ax, axes):
    for ii, ax_ in enumerate(axes):
        if target_ax is ax_:
            return ii


def find_nearest_pixel(x,y,x_grid,y_grid):
    distance = np.sqrt((x - x_grid)**2 + (y - y_grid)**2)
    index = np.unravel_index(np.argmin(distance),distance.shape)
    return index

class GetFitProfile:
    def __init__(self,fig,axes,x_grid,y_grid,los_grid,syn_profiles,
                emiss_box,title,wvl_grid,rest_wvl,fit_matrix,fit_matrix_err,
                te_box=None,ne_box=None,veff_box=None,vlos_box=None,
                params_prec = {"int":2,"wvl":1,"fwhm":1}):
        
        self.fig = fig
        self.axes = axes
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.los_grid = los_grid
        self.syn_profiles = syn_profiles
        self.emiss_box = emiss_box
        self.title = title
        self.wvl_grid = wvl_grid
        self.rest_wvl = rest_wvl
        self.fit_matrix = fit_matrix
        self.fit_matrix_err = fit_matrix_err
        self.te_box = te_box
        self.ne_box = ne_box
        self.veff_box = veff_box
        self.vlos_box = vlos_box
        self.params_prec = params_prec

        self.cid = fig.canvas.mpl_connect('button_press_event',self)

    def __call__(self, event):
        x_select_loc, y_select_loc = event.xdata,event.ydata
        ax_index = give_ax_index(event.inaxes,self.axes)

        y_select_pixel,x_select_pixel = find_nearest_pixel(x_select_loc, y_select_loc, 
        self.x_grid,self.y_grid)
        solarx_select, solary_select = self.x_grid[y_select_pixel,x_select_pixel], self.y_grid[y_select_pixel,x_select_pixel]

        profile_to_fit = self.syn_profiles[y_select_pixel, x_select_pixel, :]
        wvl_to_plot = np.linspace(self.wvl_grid[0],self.wvl_grid[-1],301)
        profile_fit_to_plot = gaussian(wvl_to_plot, *self.fit_matrix[y_select_pixel,x_select_pixel,:3]) + \
                self.fit_matrix[y_select_pixel,x_select_pixel,3]


        # fig, (ax1,ax2) = plt.subplots(2,1,figsize=(7,7),gridspec_kw={"height_ratios":[4,2]},constrained_layout=True)
        fig = plt.figure(figsize=(14,6),constrained_layout=True)
        subfigs = fig.subfigures(1,2,width_ratios=[5,7])
        ax1 = subfigs[0].subplots()
        ax2, ax3, ax4 = subfigs[1].subplots(3,1)

        ax2_te = ax2.twinx()
        ax2_ne = ax2.twinx()
        ax2_ne.spines['right'].set_position(('outward',60))

        ax3_vlos = ax3.twinx()
        ax3_veff = ax3.twinx()
        ax3_veff.spines['right'].set_position(('outward',60))
        # gs0 = fig.add_gridspec(1,2)
        # gs1 = gs0[1].subgridspec(2,1)
        # ax1 = fig.add_subplot(gs0[0])
        # ax2 = fig.add_subplot(gs1[0])
        # ax3 = fig.add_subplot(gs1[1])

        ax1.step(self.wvl_grid,profile_to_fit,where="mid",
                    color="#E87A90",label = r"$I_{\rm obs}$",lw=2,zorder=15)
        ax1.fill_between(self.wvl_grid,
        np.ones_like(self.wvl_grid)*np.min(profile_to_fit),profile_to_fit,
                    step='mid',color="#FEDFE1",alpha=0.6)

        ax1.plot(wvl_to_plot,profile_fit_to_plot,color="black",ls="-",label = r"$I_{\rm fit}$",lw=2,
                            zorder=16,alpha=0.7)

        ln_emiss = ax2.plot(self.los_grid, self.emiss_box[y_select_pixel,x_select_pixel],color="#BDC0BA",zorder=0,alpha=0.8,
                label=r"$\epsilon_{ij}\Delta s/4\pi$",lw=2)

        ln_te = ax2_te.plot(self.los_grid, self.te_box[y_select_pixel,x_select_pixel]/1e6,color="#8A6BBE",zorder=0,alpha=0.8,
                label=r"$T_e$",ls="-.",lw=2)
        ax2_te.set_ylabel(r"$T_e$ [MK]",fontsize=16)
        ln_ne = ax2_ne.plot(self.los_grid, self.ne_box[y_select_pixel,x_select_pixel],color="#90B44B",zorder=0,alpha=0.8,
                label=r"$n_e$",ls="--",lw=2)

        ax2_ne.yaxis.get_offset_text().set_position((1.21,1.05))
        ax2_ne.yaxis.get_offset_text().set_fontsize(16)
        # ax2_ne.set_yscale("log")
        ax2_ne.set_ylabel(r"$n_e\ \mathrm{[cm^{-3}]}$",fontsize=16)

        ax2_leg_list = [*ln_emiss, *ln_te, *ln_ne]
        ax2.legend(ax2_leg_list,[leg_.get_label() for leg_ in ax2_leg_list],
                bbox_to_anchor=(0,1.02,1,0.102),fontsize = 16,frameon=False,ncol=3,
                borderaxespad=0.)

        ax3.plot(self.los_grid, self.emiss_box[y_select_pixel,x_select_pixel],color="#BDC0BA",zorder=0,alpha=0.8,
                label=r"$\epsilon_{ij}\Delta s/4\pi$",lw=2)
        ln_veff = ax3_veff.plot(self.los_grid, self.veff_box[y_select_pixel,x_select_pixel],color="#F596AA",zorder=0,alpha=0.8,
                label=r"$v_{\rm eff}$",ls="-.",lw=2)
        ax3_veff.set_ylabel(r"$v_{\rm eff}\ \mathrm{[km\,s^{-1}]}$",fontsize=16)
        ax3_veff.set_ylim(top=50)
        ln_vlos = ax3_vlos.plot(self.los_grid, self.vlos_box[y_select_pixel,x_select_pixel],color="#FFB11B",zorder=0,alpha=0.8,
                label=r"$v_{\rm LOS}$",ls="--",lw=2)
        ax3_vlos.set_ylabel(r"$v_{\rm LOS}\ \mathrm{[km\,s^{-1}]}$",fontsize=16)
        ax3_vlos.set_ylim(top=25)

        ax4.plot(self.los_grid, self.emiss_box[y_select_pixel,x_select_pixel],color="#BDC0BA",zorder=0,alpha=0.8,
                label=r"$\epsilon_{ij}\Delta s/4\pi$",lw=2)
        ln_vth = ax3_veff.plot(self.los_grid, self.[y_select_pixel,x_select_pixel],color="#F596AA",zorder=0,alpha=0.8,
                label=r"$v_{\rm eff}$",ls="-.",lw=2)        

        ax1.set_ylabel(r"$I_\lambda\ [\mathrm{erg\,s^{-1}\,cm^{-2}\,nm^{-1}\,sr^{-1}}]$",fontsize=16)
        ax1.set_xlabel(r"$\textrm{Wavelength}\ \lambda\ [\mathrm{nm}]$",fontsize=16)
        subfigs[1].supylabel(r"$\epsilon_{ij}\Delta s/4\pi$ " + r"$[\mathrm{erg\,s^{-1}\,cm^{-2}\,sr^{-1}}]$",fontsize=16)

        ax4.set_xlabel(r"$s_{\rm LOS} [R_\odot]$",fontsize=16)

        ax3_leg_list = [*ln_veff, *ln_vlos]
        ax3.legend(ax3_leg_list,[leg_.get_label() for leg_ in ax3_leg_list],
                bbox_to_anchor=(0,1.2,0.4,0.102),fontsize = 16,frameon=False,ncol=2,
                borderaxespad=0.,mode="expand")

        int_total_text_fmt = r'$I_0 = {:.3g}$'
        line_wvl_text_fmt = r'$\lambda_0 = {:.2f}$'
        fwhm_text_fmt = r'$\Delta \lambda = {:.3f}$'
        vlos_text_fmt = r'$v_{{\rm LOS}} = {:.1f}$'
        veff_text_fmt = r'$v_{{\rm eff}} = {:.1f}$'

        line_wvl_plot, int_total_plot, fwhm_plot = self.fit_matrix[y_select_pixel,x_select_pixel,:3]
        vlos_plot = (1 - line_wvl_plot/self.rest_wvl)*c/1e5
        veff_plot = fwhm_plot/self.rest_wvl*c*1e-5/np.sqrt(4*np.log(2))

        ax1.text(0.03,0.92,line_wvl_text_fmt.format(num2tex(line_wvl_plot)),ha = 'left',va = 'center', 
        color = 'black',fontsize = 16,linespacing=1.5,transform=ax1.transAxes)

        ax1.text(0.03,0.82,int_total_text_fmt.format(num2tex(int_total_plot)),ha = 'left',va = 'center', 
        color = 'black',fontsize = 16,linespacing=1.5,transform=ax1.transAxes)

        ax1.text(0.03,0.72,fwhm_text_fmt.format(num2tex(fwhm_plot)),ha = 'left',va = 'center', 
        color = 'black',fontsize = 16,linespacing=1.5,transform=ax1.transAxes)

        ax1.text(0.03,0.62,vlos_text_fmt.format(num2tex(vlos_plot)),ha = 'left',va = 'center', 
        color = 'black',fontsize = 16,linespacing=1.5,transform=ax1.transAxes)

        ax1.text(0.03,0.52,veff_text_fmt.format(num2tex(veff_plot)),ha = 'left',va = 'center', 
        color = 'black',fontsize = 16,linespacing=1.5,transform=ax1.transAxes)

        
        for ax_ in (ax1, ax2, ax3,ax4):
            ax_.tick_params(labelsize=16,direction="in",top=True,which="both")
        ax1.tick_params(right=True)
        ax2.tick_params(labelbottom=False)

        for ax_ in (ax2,ax3,ax2_ne, ax2_te,ax3_veff,ax3_vlos):
            ax_.tick_params(labelsize=16,direction="in",which="both")
            ax_.yaxis.set_minor_locator(AutoMinorLocator())

        plt.show()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l","--line", help="the spectral line to plot, e.g., FeXIV_530, FeX_637")
    parser.add_argument("-fp","--filepath", help="path of the fitting file")
    parser.add_argument("-mi","--veffmin", help="minimum value of the veff colormap")
    parser.add_argument("-ma","--veffmax", help="maximum value of the veff colormap")
    parser.add_argument("-i1","--intmax", help="maximum of the int colormap")
    parser.add_argument("-i0","--intmin", help="minimum of the int colormap")
    parser.add_argument("-is","--intscale",help="scale of the intensity, log or sqrt")
    parser.add_argument("-sp","--savepath",help="path to save the plot")

    args = parser.parse_args()
    if args.line is None:
        args.line = "FeXIV_530"
    if args.filepath is None:
        args.filepath = "../../sav/AWSoM/syn_fit/box_run0034_run01_75_5th/"
    if args.intscale is None:
        args.intscale = "sqrt"

    show_fit(args.line, args.filepath, args.veffmin, args.veffmax, args.intscale,
    args.intmin, args.intmax,  args.savepath)
