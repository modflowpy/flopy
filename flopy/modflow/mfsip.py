from flopy.mbase import Package

class ModflowSip(Package):
    'Strongly Implicit Procedure package class\n'
    def __init__(self, model, mxiter=200, nparm=5, \
                 accl=1, hclose=1e-5, ipcalc=1, wseed=0, iprsip=0, extension='sip', unitnumber=25):
        Package.__init__(self, model, extension, 'SIP', unitnumber) # Call ancestor's init to set self.parent, extension, name and unit number
        self.url = 'sip.htm'
        self.mxiter = mxiter
        self.nparm = nparm
        self.accl = accl
        self.hclose = hclose
        self.ipcalc = ipcalc
        self.wseed = wseed
        self.iprsip = iprsip
        self.parent.add_package(self)
    def write_file(self):
        # Open file for writing
        f_sip = open(self.fn_path, 'w')
        f_sip.write('%10i%10i\n' % (self.mxiter,self.nparm))
        f_sip.write('%10f%10f%10i%10f%10i\n' % (self.accl, self.hclose, self.ipcalc, self.wseed, self.iprsip))
        f_sip.close()

