"""Plot relative power spectra from simulations with Nbodykit"""
import os
import os.path
import numpy as np
import scipy.interpolate

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from fake_spectra import spectra
matplotlib.use("PDF")

simnames = {"L1000-baronlyglass": "BIGGLASS", "L300-adaptive": "ADAPTIVE", "L300-baronlyglass": "HALFGLASS",
            "L300-oversample": "UNDERSAMP", "L300" : "TWOGRID", "L120-total" : "LYATOTAL", "L120-baronlyglass" : "LYAGLASS",
            "L60-total" : "LYATOTAL", "L60-baronlyglass" : "LYAGLASS",
            "L300-hydro": "HYDROGLASS",
           }

lss = {"L1000-baronlyglass": "-", "L300-adaptive": "-.", "L300-baronlyglass": "--",
        "L300-oversample": "-", "L300" : "-", "L120-total" : "-", "L120-baronlyglass" : "--",
        "L60-total" : "-", "L60-baronlyglass" : "--",
        "L300-hydro": "-.",
      }
colors = {"L1000-baronlyglass": "blue", "L300-adaptive": '#1f77b4', "L300-baronlyglass": '#d62728',
          "L300-oversample": '#2ca02c', "L300" : '#bcbd22', "L120-total" : "brown", "L120-baronlyglass" : "blue",
          "L60-total" : "brown", "L60-baronlyglass" : "blue",
            "L300-hydro": "brown",
         }
colorsbar = {"L1000-baronlyglass": '#bcbdff', "L300-adaptive": "#7f7f7f", "L300-baronlyglass": '#d627ff',
             "L300-oversample": 'yellowgreen', "L300" : 'yellowgreen', "L120-total" : "yellowgreen", "L120-baronlyglass" : "#bcbdff",
             "L60-total" : "yellowgreen", "L60-baronlyglass" : "#bcbdff",
            "L300-hydro": "orange",
            }
datadir = "powers"

plotdir = "plots"

def modecount_rebin(kk, pk, modes, pkc, minmodes=10, ndesired=100):
    """Rebins a power spectrum so that there are sufficient modes in each bin"""
    assert np.all(kk) > 0
    logkk=np.log10(kk)
    mdlogk = (np.max(logkk) - np.min(logkk))/ndesired
    istart=iend=1
    count=0
    pk_div = pk /pkc(kk)
    k_list=[kk[0]]
    pk_list=[pk_div[0]]
    targetlogk=mdlogk+logkk[istart]
    while iend < np.size(logkk)-1:
        count+=modes[iend]
        iend+=1
        if count >= minmodes and logkk[iend-1] >= targetlogk:
            pk1 = np.sum(modes[istart:iend]*pk_div[istart:iend])/count
            kk1 = np.sum(modes[istart:iend]*kk[istart:iend])/count
            k_list.append(kk1)
            pk_list.append(pk1)
            istart=iend
            targetlogk=mdlogk+logkk[istart]
            count=0
    k_list = np.array(k_list)
    pk_list = np.array(pk_list) * pkc(k_list)
    return (k_list, pk_list)

def get_saved_power(sdir, redshift, sp, kpc=True):
    """Load a saved power spectrum from a file."""
    scale = 1./(1+redshift)
    fname = os.path.join(sdir, "output/%s-%.4f.txt" % (sp, scale))
    pkk = np.loadtxt(fname)
    ii = np.where((pkk[:,2] >= 1)*(pkk[:,0] > 0))[0]
    #Convert to Mpc/h units
    if kpc:
        pkk[ii, 0]*= 1e3
        pkk[ii, 1]/=1e9
    return pkk[ii,0], pkk[ii,1], pkk[ii, 2]

def get_class_power(z, camb_transfer):
    """Find the class baryon and DM power spectrum. Only works for snapshots at integer redshift."""
    try:
        camb_trans = np.loadtxt(os.path.join(camb_transfer, "transfer.dat-"+str(int(np.round(z)))))
        camb_mat = np.loadtxt(os.path.join(camb_transfer, "matterpow.dat-"+str(int(np.round(z)))))
    except IOError:
        camb_trans = np.loadtxt(os.path.join(camb_transfer, "transfer.dat-%.1f" % z))
        camb_mat = np.loadtxt(os.path.join(camb_transfer, "matterpow.dat-%.1f" % z))
    intptot = scipy.interpolate.interp1d(camb_mat[:,0], camb_mat[:,1])
    intpdm = scipy.interpolate.interp1d(camb_trans[:,0], camb_trans[:,3]/camb_trans[:,5])
    intpbar = scipy.interpolate.interp1d(camb_trans[:,0], camb_trans[:,2]/camb_trans[:,5])
    intpdmpk = scipy.interpolate.interp1d(camb_mat[:,0], intpdm(camb_mat[:,0])**2 * camb_mat[:,1])
    intpbarpk = scipy.interpolate.interp1d(camb_mat[:,0], intpbar(camb_mat[:,0])**2 * camb_mat[:,1])
    intpratpk = scipy.interpolate.interp1d(camb_trans[:,0], camb_trans[:,2]/camb_trans[:,3])
    return camb_mat[:,0], intpbarpk, intpdmpk, intptot, intpratpk

def plot_power(zz, sims, plottitle, total=False):
    """Check the initial power against linear theory and a linearly grown IC power"""

    #Check types have the same power
    fig = Figure()
    canvas = FigureCanvasPdf(fig)
    ax = fig.add_subplot(111)

    xmin = 0.1
    ymin = 0.92
    fig2 = Figure()
    canvas2 = FigureCanvasPdf(fig2)
    ax2 = fig2.add_subplot(111)
    for ss in sims:
        sdir = os.path.join(datadir, ss)
        classkk, classbarpk, classdmpk, classtot, classrat = get_class_power(zz, os.path.join(datadir, ss))
        if ss == sims[0]:
            ax.plot(classkk, classbarpk(classkk)/ classdmpk(classkk), ls=":", label='CLASS', color="grey")
        kkcdm, pkcdm, modecdm = get_saved_power(sdir, zz, "power-DM")
        kkbar, pkbar, modebar = get_saved_power(sdir, zz, "power-bar")
        #Note k in kpc/h
        ax2.axhline(1, ls=":", color="grey")
        kkrat, pkrat = modecount_rebin(kkcdm, pkbar/pkcdm, modecdm, classrat, minmodes = 1, ndesired=500)
        ax.plot(kkrat, pkrat, ls=lss[ss], color=colors[ss], label=simnames[ss])
        ii = np.where(kkcdm < 0.2)
        ymin = np.min([np.min(pkbar[ii]/pkcdm[ii])-0.02, ymin])
        kkcdm, pkcdm = modecount_rebin(kkcdm, pkcdm, modecdm, classdmpk)
        kkbar, pkbar = modecount_rebin(kkbar, pkbar, modebar, classbarpk)
        ax2.plot(kkbar, pkbar/classbarpk(kkbar) , ls=lss[ss], color=colorsbar[ss], label=simnames[ss]+" baryon")
        ax2.plot(kkcdm, pkcdm/classdmpk(kkcdm) , ls=lss[ss], color=colors[ss], label=simnames[ss]+" DM")
        xmin = np.min([xmin, kkcdm[0]])
        if total:
            kktot, pktot, modetot = get_saved_power(sdir, zz, "powerspectrum", kpc=False)
            kktot, pktot = modecount_rebin(kktot, pktot, modetot, classtot)
            ax2.plot(kktot, pktot/classtot(kktot), label=simnames[ss]+" Total", ls=lss[ss], color="black")

    ax.set_xlabel("k (h/Mpc)")
    ax.set_ylabel(r"$P_\mathrm{b} / P_\mathrm{CDM} (k, z=%d)$" % zz )
    ax.set_xscale('log')
    ax.set_xlim(xmin/2., 2)
    ax.set_ylim(ymin, 1.04)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(plotdir, plottitle + '_%d_relpower.pdf' % zz))
    fig.clf()
    ax2.set_xlabel("k (h/Mpc)")
    ax2.set_ylabel(r"$P_\mathrm{SIM} / P_\mathrm{CLASS} (k, z=%d)$" % zz )
    ax2.set_xscale('log')
    ax2.set_xlim(xmin/2., 2)
    ax2.set_ylim(0.9, 1.15)
    ax2.legend(loc="upper left")
    fig2.tight_layout()
    fig2.savefig(os.path.join(plotdir, plottitle + '_%d_class.pdf' % zz))
    fig2.clf()

def plot_lyman_alpha_spectra(nums, sim1, sim2, plottitle):
    """Plot the effect of this on the Lyman alpha forest mean flux."""
    fig = Figure()
    canvas = FigureCanvasPdf(fig)
    ax = fig.add_subplot(111)
    fig2 = Figure()
    canvas2 = FigureCanvasPdf(fig2)
    ax2 = fig2.add_subplot(111)
    for nn in nums:
        sdir1 = os.path.join(os.path.join(datadir, sim1), "output")
        sdir2 = os.path.join(os.path.join(datadir, sim2), "output")
        first = spectra.Spectra(nn, sdir1, None, None, savefile="lya_forest_spectra.hdf5")
        second = spectra.Spectra(nn, sdir2, None, None, savefile="lya_forest_spectra.hdf5")
        #Get flux power without mean flux rescaling
        kf1, pkf1 = first.get_flux_power_1D()
        kf2, pkf2 = second.get_flux_power_1D()
        ax.semilogx(kf1, pkf1/pkf2, label="z=%.1f" % first.red)
        #Get flux power with mean flux rescaling
        mf = 0.0023 * (1 + first.red)**3.65
        kf1, pkf1 = first.get_flux_power_1D(mean_flux_desired=mf)
        kf2, pkf2 = second.get_flux_power_1D(mean_flux_desired=mf)
        ax2.semilogx(kf1, pkf1/pkf2, label="z= %.1f" %first.red)

    ax.set_xscale('log')
    ax.set_xlim(1e-3,2e-2)
    ax.set_xlabel(r"$k_F$ (s/km)")
    ax.set_ylabel(r'$P_\mathrm{F}(k)$ ratio')
    ax.set_ylim(0.8, 1.15)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(plotdir, plottitle + '_relflux_nomf.pdf'))
    fig.clf()
    ax2.set_xlim(1e-3,2e-2)
    ax2.set_xlabel(r"$k_F$ (s/km)")
    ax2.set_ylabel(r'$P_\mathrm{F}(k)$ ratio')
    ax2.set_ylim(0.8, 1.1)
    ax2.legend(loc="lower left")
    fig2.tight_layout()
    fig2.savefig(os.path.join(plotdir, plottitle + '_relflux_mf.pdf'))
    fig2.clf()

if __name__ == "__main__":
    for red in (49, 2, 4, 9):
        plot_power(red, ["L300"], "literature", total=True)
        plot_power(red, ["L300-baronlyglass", "L1000-baronlyglass", "L300-hydro"], "halfglass")
        plot_power(red, ["L300-baronlyglass", "L300-oversample","L300-adaptive"], "oversample")
#     for red in (2.2, 3.2,4.2, 9):
#         plot_power(red, ["L60-total", "L60-baronlyglass"], "lya60")
    for red in (2.2, 3, 4, 9, 49):
        plot_power(red, ["L120-total", "L120-baronlyglass" ], "lya120")
    plot_lyman_alpha_spectra([12, 8, 3], "L120-total", "L120-baronlyglass", "lya120")
