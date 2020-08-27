Tutorial 1: Confined Steady-State Flow Model
============================================

This tutorial demonstrates use of FloPy to develop a simple MODFLOW model.
Note that you can access the latest version this tutorial python script from
`here <https://github.com/modflowpy/flopy/blob/develop/.docs/pysrc/tutorial01.py>`_.

Getting Started
---------------
If FloPy has been properly installed, then it can be imported as follows::


  import flopy


Now that we can import flopy, we begin creating our simple MODFLOW model.

Creating the MODFLOW Model
--------------------------
One of the nice things about creating models in python is that it is very easy
to change one or two things and completely change the grid resolution for your
model.  So in this example, we will design our python script so that the number
of layers, columns, and rows can be easily changed.

We can create a very simple MODFLOW model that has a basic package (BAS),
discretization input file (DIS), layer-property flow (LPF) package, output
control (OC), and preconditioned conjugate gradient (PCG) solver.  Each one of
these has its own input file, which will be created automatically by flopy,
provided that we pass flopy the correct information.

Discretization
^^^^^^^^^^^^^^

We start by creating our flopy model object as follows::


    # Assign name and create modflow model object
    modelname = 'tutorial1'
    mf = flopy.modflow.Modflow(modelname, exe_name='mf2005')

Next, let's proceed by defining our model domain and creating a MODFLOW grid
to span the domain::

    # Model domain and grid definition
    Lx = 1000.
    Ly = 1000.
    ztop = 0.
    zbot = -50.
    nlay = 1
    nrow = 10
    ncol = 10
    delr = Lx / ncol
    delc = Ly / nrow
    delv = (ztop - zbot) / nlay
    botm = np.linspace(ztop, zbot, nlay + 1)

With this information, we can now create the flopy discretization object by
entering the following::

    # Create the discretization object
    dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                                   top=ztop, botm=botm[1:])

The obvious question at this point is, how do I know which arguments are
required by this strange thing called flopy.modflow.ModflowDis?  Fortunately,
there is an online help page for each one of the model objects.  The page for
the DIS input file is located at `flopy.modflow.mfdis <mfdis.html>`__.

Basic Package
^^^^^^^^^^^^^

Next we can create a flopy object that represents the MODFLOW Basic Package.
Details on the flopy BAS class are at: `flopy.modflow.mfbas <mfbas.html>`__.
For this simple model, we will assign constant head values of 10. and 0. to the
first and last model columns (in all layers), respectively.  The python code
for doing this is::

    # Variables for the BAS package
    ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
    ibound[:, :, 0] = -1
    ibound[:, :, -1] = -1
    strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
    strt[:, :, 0] = 10.
    strt[:, :, -1] = 0.
    bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt)

Layer-Property Flow Package
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Details on the flopy LPF class are at: `flopy.modflow.mflpf <mflpf.html>`__.
Values of 10. are assigned for the horizontal and vertical hydraulic
conductivity::

    # Add LPF package to the MODFLOW model
    lpf = flopy.modflow.ModflowLpf(mf, hk=10., vka=10., ipakcb=53)

Because we did not specify a value for laytyp, Flopy will use the default value
of 0, which means that this model will be confined.

Output Control
^^^^^^^^^^^^^^

Details on the flopy OC class are at: `flopy.modflow.mfoc <mfoc.html>`__.  Here
we can use the default OC settings by specifying the following::

    # Add OC package to the MODFLOW model
    spd = {(0, 0): ['print head', 'print budget', 'save head', 'save budget']}
    oc = flopy.modflow.ModflowOc(mf, stress_period_data=spd, compact=True)

The stress period dictionary is used to set what output is saved for the
corresponding stress period and time step.  In this case, the tuple (0, 0)
means that stress period 1 and time step 1 for MODFLOW will have output saved.
Head and budgets will be printed and head and budget information will be saved.

Preconditioned Conjugate Gradient Package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Details on the flopy PCG class are at: `flopy.modflow.mfpcg <mfpcg.html>`__.
The default settings used by flopy will be used by specifying the following
commands::

    # Add PCG package to the MODFLOW model
    pcg = flopy.modflow.ModflowPcg(mf)

Writing the MODFLOW Data Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MODFLOW input data files are written by simply issuing the following::

    # Write the MODFLOW model input files
    mf.write_input()

Running the Modeling
--------------------

Flopy can also be used to run the model.  The model object (mf in this example)
has an attached method that will run the model.  For this to work, the MODFLOW
program must be located somewhere within the system path, or within the working
directory.  In this example, we have specified that the name of the executable
program is 'mf2005'.  Issue the following to run the model::

    # Run the MODFLOW model
    success, buff = mf.run_model()

Here we have used run_model, and we could also have specified values for the
optional keywords silent, pause, and report.

Post-Processing the Results
---------------------------

Now that we have successfully built and run our MODFLOW model, we can look at
the results.  MODFLOW writes the simulated heads to a binary data output file.
We cannot look at these heads with a text editor, but flopy has a binary
utility that can be used to read the heads.  The following statements will
read the binary head file and create a plot of simulated heads for layer 1::

    import matplotlib.pyplot as plt
    import flopy.utils.binaryfile as bf
    plt.subplot(1, 1, 1, aspect='equal')
    hds = bf.HeadFile(modelname + '.hds')
    head = hds.get_data(totim=1.0)
    levels = np.arange(1, 10, 1)
    extent = (delr / 2., Lx - delr / 2., Ly - delc / 2., delc / 2.)
    plt.contour(head[0, :, :], levels=levels, extent=extent)
    plt.savefig('tutorial1a.png')

If everything has worked properly, you should see the following head contours.

.. figure:: _static/tutorial1a.png
   :alt: head contours in first layer
   :scale: 100 %
   :align: left

Flopy also has some pre-canned plotting capabilities can can be accessed using
the PlotMapView class.  The following code shows how to use the plotmapview
class to plot boundary conditions (IBOUND), plot the grid, plot head contours,
and plot vectors::

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1, aspect='equal')

    hds = bf.HeadFile(modelname + '.hds')
    times = hds.get_times()
    head = hds.get_data(totim=times[-1])
    levels = np.linspace(0, 10, 11)

    cbb = bf.CellBudgetFile(modelname + '.cbc')
    kstpkper_list = cbb.get_kstpkper()
    frf = cbb.get_data(text='FLOW RIGHT FACE', totim=times[-1])[0]
    fff = cbb.get_data(text='FLOW FRONT FACE', totim=times[-1])[0]

    pmv = flopy.plot.PlotMapView(model=mf, layer=0)
    qm = pmv.plot_ibound()
    lc = pmv.plot_grid()
    cs = pmv.contour_array(head, levels=levels)
    quiver = pmv.plot_discharge(frf, fff, head=head)
    plt.savefig('tutorial1b.png')

.. figure:: _static/tutorial1b.png
   :alt: head contours in first layer
   :scale: 100 %
   :align: left
