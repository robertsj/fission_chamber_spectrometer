import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from master_data import umg_path, data_path, plot_path


class Unfolding(object):
    '''Used to create functions and classes useful for the spectral unfolding.'''

    def __init__(self):
        ids = [['bare    ', 'bareSphere      '],
               ['2in     ', '2inSphere       '],
               ['3in     ', '3inSphere       '],
               ['5in     ', '5inSphere       '],
               ['8in     ', '8inSphere       '],
               ['10in    ', '10inSphere      '],
               ['12in    ', '12inSphere      ']]
        self.setSphereIDs(ids)
        self.setSphereSizes([0.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0])
        self.setCorrectionFactor(0)
        self.setDefEnergyUnits('MeV', 'MeV')
        self.setMode(2)
        self.set_names('generic')
        # self.setChiSqr(len(self.sphereSizes))
        self.setChiSqr(1)
        self.setTemp([1.0, 0.85])
        self.setIter([2000, 25])
        self.setSolnStructure(2)
        self.setSolnRepresentation(1)
        self.setSolnScaling([0, 1, 1])
        self.set_routine('maxed')

    def setSphereIDs(self, ids):
        # sphereIDs - a list of lists that contains an 8 character short ID and 16 char long ID for
        #             naming each sphere
        self.sphereIDs = ids

    def setSphereSizes(self, sizes):
        # sphereSizes - the sphere diameters in inches
        self.sphereSizes = sizes

    def setCorrectionFactor(self, f):
        # correctionFactor - a factor by which to multiply measured data and absolute uncertainties
        #                    entering a value of zero means data remains unchanged
        self.correctionFactor = f

    def setDefEnergyUnits(self, rfu, dsu):
        # reErgUnits (or dsErgUnits) - the units of the response functions, either 'MeV', 'keV', or 'eV'
        self.rfErgUnits = rfu
        self.dsErgUnits = dsu

    def setMode(self, m):
        # mode - the default spectrum input format
        #   1 - (fluence rate per bin) / (width of the bin in E)     ~ dPhi / dE
        #   2 - fluence rate per bin
        #   3 - (fluence rate per bin) / (width of the bin in ln(E))     ~ E * dPhi / dE
        self.mode = m

    def set_names(self, name):
        self.setIbuName(name)
        self.setFmtName(name)
        self.setFluName(name)
        self.setOutName(name)
        self.setInpName(name)

    def setIbuName(self, name):
        self.ibuName = name

    def setFmtName(self, name):
        self.fmtName = name

    def setFluName(self, name):
        self.fluName = name

    def setOutName(self, name):
        self.outName = name[:-1] + 'o'

    def setInpName(self, name):
        self.inpName = name

    def setChiSqr(self, chi):
        self.finalChiSqr = chi

    def setTemp(self, temp):
        self.temp = temp

    def setIter(self, it):
        self.iter = it

    def setSolnStructure(self, s):
        self.solnStructure = s

    def setSolnRepresentation(self, r):
        self.solnRepresentation = r

    def setSolnScaling(self, s):
        self.scaling = s

    def set_routine(self, r):
        self.routine = r

    def set_responses(self, data):
        # responseData - the measured response from each bonner sphere
        # responseError - the uncertainty associated with responseData due to statistics (as a percentage)
        # extraError - uncertainty in responseData due to causes other than statistics (as a percentage)
        self.responseData = data
        self.responseError = np.sqrt(data) / data
        self.extraError = np.full(len(data), 0.5)

    def set_rf(self, edges, responses):
        # rfErgEdges = the energy bin edges of the response functions
        # responses - the 2D matrix containing the responses for each sphere
        self.rfErgEdges = edges
        self.responses = responses

    def set_ds(self, ds):
        # set the default spectrum
        self.ds = ds

    def makeStep(self, x, y):
        assert len(x) - 1 == len(y), '{} - 1 != {}'.format(len(x), len(y))
        Y = np.array([[yy, yy] for yy in np.array(y)]).flatten()
        X = np.array([[xx, xx] for xx in np.array(x)]).flatten()[1:-1]
        return X, Y

    def getExe(self):
        if self.routine == 'maxed':
            return umg_path + 'MXD_FC33.exe'
        elif self.routine == 'gravel':
                return umg_path + 'GRV_FC33.exe'
        else:
            print('NOT A VALID ROUTINE')

    def writeMeasuredData(self):
        ibuString = 'Measured Responses for Bonner Spheres\n'
        ibuString += '   {}   {}\n'.format(len(self.sphereSizes), self.correctionFactor)
        for i, names in enumerate(self.sphereIDs):
            ibuString += '{}  {:4.1f}      {:4.3E}      {:4.3E}    {:4.2f}    {:4.2f}{:6d}\n'.format(
                    self.sphereIDs[i][0], self.sphereSizes[i], self.responseData[i],
                    self.responseData[i] * self.responseError[i], self.responseError[i], self.extraError[i], i + 1)
        ibuString += '\n12341234----.-123456789.12345---------.12345-----.12-----.12I23456'
        with open('inp/{}.ibu'.format(self.ibuName), 'w+') as F:
            F.write(ibuString)
        return

    def writeResponseFunctions(self):
        if self.rfErgUnits == 'eV':
            self.rfIEU = 0
        elif self.rfErgUnits == 'MeV':
            self.rfIEU = 1
        elif self.rfErgUnits == 'keV':
            self.rfIEU = 2
        fmtString = 'Response Function file for Bonner Spheres\n'
        fmtString += 'Contains {} spheres and {} energy groups using units of {}\n'.format(
                len(self.sphereIDs), len(self.rfErgEdges) - 1, self.rfErgUnits)
        fmtString += '        {}   {}\n '.format(len(self.rfErgEdges), self.rfIEU)

        for i, erg in enumerate(self.rfErgEdges):
            fmtString += '{:4.3E} '.format(erg)
            if (i + 1) % 8 == 0:
                fmtString += '\n '
                eolFlag = 1
            else:
                eolFlag = 0
        if eolFlag == 0:
            fmtString += '\n'

        fmtString += '         0   \n'
        fmtString += '         {}   \n'.format(len(self.sphereSizes))

        for i, names in enumerate(self.sphereIDs):
            fmtString += '{}  {}\n'.format(names[0], names[1])
            fmtString += ' 1.000E+00      cm^2         0         0    3    1    1    0\n'

            for j, resp in enumerate(self.responses[i]):
                fmtString += ' {:4.3E}'.format(resp)
                if (j + 1) % 8 == 0:
                    fmtString += '\n'
                    eolFlag = 1
                else:
                    eolFlag = 0
            if eolFlag == 0:
                fmtString += '\n'
        with open('inp/{}.fmt'.format(self.fmtName), 'w+') as F:
            F.write(fmtString)
        return

    def writeDefaultSpectrum(self):
        if self.dsErgUnits == 'eV':
            self.dsIEU = 0
        elif self.dsErgUnits == 'MeV':
            self.dsIEU = 1
        elif self.dsErgUnits == 'keV':
            self.dsIEU = 2
        fluString = 'Default Spectrum for Bonner Sphere Unfolding\n'
        fluString += '       {}          {}\n'.format(self.mode, self.dsIEU)
        fluString += '       2        {}        {}   {:4.3E}\n'.format(
                self.ds.num_edges, self.ds.num_edges, self.ds.edges[-1])
        for i in range(self.ds.num_bins):
            fluString += '{:4.3E}  {:4.3E}  {:4.3E}\n'.format(self.ds.edges[i], self.ds.values[i], self.ds.error[i])
        with open('inp/{}.flu'.format(self.fluName), 'w+') as F:
            F.write(fluString)
        return

    def writeControlFile(self):
        inpString = '{}.ibu   \n'.format(self.ibuName)
        inpString += '{}.fmt   \n'.format(self.fmtName)
        inpString += '{}   \n'.format(self.outName)
        inpString += '{}.flu   \n'.format(self.fluName)
        inpString += '{}   \n'.format(max(self.rfErgEdges))
        inpString += '{}   \n'.format(self.finalChiSqr)
        if self.routine == 'maxed':
            inpString += '{}, {}   \n'.format(self.temp[0], self.temp[1])
        if self.routine == 'gravel':
            inpString += '{}, {}   \n'.format(self.iter[0], self.iter[1])
        inpString += '{}, {}   \n'.format(self.solnStructure, self.solnRepresentation)
        inpString += '{}   \n{}   \n{}\n'.format(self.scaling[0], self.scaling[1], self.scaling[2])
        with open('inp/{}.inp'.format(self.inpName), 'w+') as F:
            F.write(inpString)
        return

    def writeInputFiles(self):
        try:
            os.mkdir('inp')
        except:
            pass
        self.writeMeasuredData()
        self.writeResponseFunctions()
        self.writeDefaultSpectrum()
        self.writeControlFile()

    def unfold(self):
        self.exe = self.getExe()
        os.chdir('inp')
        with open('dump.txt', 'a+') as dump:
            subprocess.call(['wine', '{}'.format(self.exe), '{}.inp'.format(self.inpName)], stdout=dump)
        os.chdir('..')

    def storeResult(self, label):
        sol = np.loadtxt('inp/{}.flu'.format(self.outName), skiprows=3)
        s = ''
        for i in sol:
            s += '{:10.6e}  {:10.6e}  {:10.6e}\n'.format(*i)
        with open(data_path + '{}_unfolded.txt'.format(label), 'w+') as F:
            F.write(s)

    def run(self, label):
        self.writeInputFiles()
        self.unfold()
        self.storeResult(label)

    def plotSpectra(self, name='generic', clear=True, ds=True):
        plt.figure(0)
        for label, s in self.solutions:
            plt.plot(s.step_x, s.step_y, label='{}'.format(label))
        if ds:
            plt.plot(self.ds.step_x, self.ds.step_y, label='Default Spectrum')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(1E-8, 20)
        plt.ylim(1E1, 1E13)
        plt.xlabel('Energy ${}$'.format(self.dsErgUnits))
        plt.ylabel('Fluence')
        plt.legend()
        plt.savefig(plot_path + '{}_plot.png'.format(name))
        if clear:
            self.solutions = []
        plt.close()

    def clearRepo(self):
        pass
