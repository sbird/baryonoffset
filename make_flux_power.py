"""Script to make a text file with the mean flux for each snapshot."""

import os.path
import numpy as np
from fake_spectra.spectra import Spectra

def obs_mean_tau(redshift, amp=0, slope=0):
    """The mean flux from 0711.1862: is (0.0023±0.0007) (1+z)^(3.65±0.21)
    Note we constrain this much better from the SDSS data itself:
    this is a weak prior"""
    return (2.3+amp)*1e-3*(1.0+redshift)**(3.65+slope)

def make_flux_power(simdir):
    """Plot the relative flux power spectrum."""
    #Without mean flux rescaling
    for snap in range(0,30):
        try:
            spec = Spectra(snap, simdir, None, None, savefile="lya_forest_spectra.hdf5")
            (kf1, pkf1) = spec.get_flux_power_1D()
            nomf = os.path.join(os.path.join(simdir,"SPECTRA_"+str(snap).rjust(3,'0')), "flux_power_no_mf_%d.txt" % spec.red)
            np.savetxt(nomf, (kf1, pkf1))
            (kf1, spkf1) = spec.get_flux_power_1D(mean_flux_desired=obs_mean_tau(spec.red))
            mf = os.path.join(os.path.join(simdir,"SPECTRA_"+str(snap).rjust(3,'0')), "flux_power_mf_%d.txt" % spec.red)
            np.savetxt(mf, (kf1, spkf1))
        except IOError:
            pass
