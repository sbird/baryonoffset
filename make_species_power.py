"""Make the per-species power spectrum using NBodyKit"""
import os.path as path
import sys
import glob
import numpy
from nbodykit.lab import BigFileCatalog,FFTPower,MultipleSpeciesCatalog
import timeit

def sptostr(sp):
    """Get a string from a species"""
    if sp == 0:
        return "bar"
    elif sp == 1:
        return "DM"
    elif sp == 2:
        return "nu"
    return ""

def compute_power(output, Nmesh=1024, species=1, spec2 = None):
    """Compute the compensated power spectrum from a catalogue."""
    #If there are stars present, treat them as baryons
    if species == 0:
        try:
            catnu = MultipleSpeciesCatalog(["gas", "star"],
                BigFileCatalog(output, dataset='0/', header='Header'),
                BigFileCatalog(output, dataset='4/', header='Header'))
            time = catnu.attrs["gas.Time"][0]
        except TypeError:
            catnu = BigFileCatalog(output, dataset=str(species)+'/', header='Header')
            time = catnu.attrs["Time"][0]
    else:
        catnu = BigFileCatalog(output, dataset=str(species)+'/', header='Header')
        time = catnu.attrs["Time"][0]
    sp = sptostr(species)
    sp2 = sptostr(spec2)
    outfile = path.join(output,"../power-"+sp+sp2+"-%.4f.txt" % time)
    if path.isfile(outfile):
        return
    catnu.to_mesh(Nmesh=Nmesh, resampler='cic', compensated=True, interlaced=True, )
    if spec2 is not None:
        catcdm = BigFileCatalog(output, dataset=str(spec2)+'/', header='Header')
        catcdm.to_mesh(Nmesh=Nmesh, window='cic', compensated=True, interlaced=True)
        pkcross = FFTPower(catnu, mode='1d', Nmesh=1024,second = catcdm, dk=1.0e-7)
        power = pkcross.power
    else:
        pknu = FFTPower(catnu, mode='1d', Nmesh=Nmesh, dk=1.0e-7)
        power = pknu.power
    numpy.savetxt(outfile,numpy.array([power['k'], power['power'].real,power['modes']]).T)
    return power

def all_compute(directory):
    """Do computation for all snapshots in a directory"""
    snaps = glob.glob(path.join(directory, "output/PART_[0-9][0-9][0-9]"))
    for ss in snaps:
        try:
            compute_power(ss)
            compute_power(ss,species=0)
        except IOError:
            print("Could not read snapshot", ss)
            pass

if __name__ == "__main__":
    all_compute(sys.argv[1])
