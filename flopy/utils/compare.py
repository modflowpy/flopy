import os
import textwrap
from typing import List, Optional, Union

import numpy as np

from flopy.modflow import ModflowOc
from flopy.utils import FormattedHeadFile, HeadFile, HeadUFile
from flopy.utils.mfreadnam import get_entries_from_namefile


def _diffmax(v1, v2):
    """Calculate the maximum difference between two vectors.

    Parameters
    ----------
    v1 : numpy.ndarray
        array of base model results
    v2 : numpy.ndarray
        array of comparison model results

    Returns
    -------
    diffmax : float
        absolute value of the maximum difference in v1 and v2 array values
    indices : numpy.ndarry
        indices where the absolute value of the difference is equal to the
        absolute value of the maximum difference.

    """
    if v1.ndim > 1 or v2.ndim > 1:
        v1 = v1.flatten()
        v2 = v2.flatten()
    if v1.size != v2.size:
        err = (
            f"Error: calculate_difference v1 size ({v1.size}) "
            + f"is not equal to v2 size ({v2.size})"
        )
        raise Exception(err)

    diff = abs(v1 - v2)
    diffmax = diff.max()
    return diffmax, np.where(diff == diffmax)


def _difftol(v1, v2, tol):
    """Calculate the difference between two arrays relative to a tolerance.

    Parameters
    ----------
    v1 : numpy.ndarray
        array of base model results
    v2 : numpy.ndarray
        array of comparison model results
    tol : float
        tolerance used to evaluate base and comparison models

    Returns
    -------
    diffmax : float
        absolute value of the maximum difference in v1 and v2 array values
    indices : numpy.ndarry
        indices where the absolute value of the difference exceed the
        specified tolerance.

    """
    if v1.ndim > 1 or v2.ndim > 1:
        v1 = v1.flatten()
        v2 = v2.flatten()
    if v1.size != v2.size:
        err = (
            f"Error: calculate_difference v1 size ({v1.size}) "
            + f"is not equal to v2 size ({v2.size})"
        )
        raise Exception(err)

    diff = abs(v1 - v2)
    return diff.max(), np.where(diff > tol)


def compare_budget(
    namefile1: Optional[Union[str, os.PathLike]],
    namefile2: Optional[Union[str, os.PathLike]],
    max_cumpd=0.01,
    max_incpd=0.01,
    outfile: Optional[Union[str, os.PathLike]] = None,
    files1: Optional[
        Union[str, os.PathLike, List[Union[str, os.PathLike]]]
    ] = None,
    files2: Optional[
        Union[str, os.PathLike, List[Union[str, os.PathLike]]]
    ] = None,
):
    """Compare the budget results from two simulations.

    Parameters
    ----------
    namefile1 : str or PathLike, optional
        namefile path for base model
    namefile2 : str or PathLike, optional
        namefile path for comparison model
    max_cumpd : float
        maximum percent discrepancy allowed for cumulative budget terms
        (default is 0.01)
    max_incpd : float
        maximum percent discrepancy allowed for incremental budget terms
        (default is 0.01)
    outfile : str or PathLike, optional
        budget comparison output file name. If outfile is None, no
        comparison output is saved. (default is None)
    files1 : str, PathLike, or list, optional
        base model output file. If files1 is not None, results
        will be extracted from files1 and namefile1 will not be used.
        (default is None)
    files2 : str, PathLike, or list, optional
        comparison model output file. If files2 is not None, results
        will be extracted from files2 and namefile2 will not be used.
        (default is None)

    Returns
    -------
    success : bool
        boolean indicating if the difference between budgets are less
        than max_cumpd and max_incpd

    """
    try:
        import flopy
    except:
        msg = "flopy not available - cannot use compare_budget"
        raise ValueError(msg)

    # headers
    headers = ("INCREMENTAL", "CUMULATIVE")
    direction = ("IN", "OUT")

    # Get name of list files
    lst_file1 = None
    if files1 is None:
        lst_file = get_entries_from_namefile(namefile1, "list")
        lst_file1 = lst_file[0][0] if any(lst_file) else None
    else:
        if isinstance(files1, (str, os.PathLike)):
            files1 = [files1]
        for file in files1:
            if (
                "list" in os.path.basename(file).lower()
                or "lst" in os.path.basename(file).lower()
            ):
                lst_file1 = file
                break
    lst_file2 = None
    if files2 is None:
        lst_file = get_entries_from_namefile(namefile2, "list")
        lst_file2 = lst_file[0][0] if any(lst_file) else None
    else:
        if isinstance(files2, (str, os.PathLike)):
            files2 = [files2]
        for file in files2:
            if (
                "list" in os.path.basename(file).lower()
                or "lst" in os.path.basename(file).lower()
            ):
                lst_file2 = file
                break
    # Determine if there are two files to compare
    if lst_file1 is None or lst_file2 is None:
        print("lst_file1 or lst_file2 is None")
        print(f"lst_file1: {lst_file1}")
        print(f"lst_file2: {lst_file2}")
        return True

    # Open output file
    if outfile is not None:
        f = open(outfile, "w")

    # Initialize SWR budget objects
    lst1obj = flopy.utils.MfusgListBudget(lst_file1)
    lst2obj = flopy.utils.MfusgListBudget(lst_file2)

    # Determine if there any SWR entries in the budget file
    if not lst1obj.isvalid() or not lst2obj.isvalid():
        return True

    # Get numpy budget tables for lst_file1
    lst1 = []
    lst1.append(lst1obj.get_incremental())
    lst1.append(lst1obj.get_cumulative())

    # Get numpy budget tables for lst_file2
    lst2 = []
    lst2.append(lst2obj.get_incremental())
    lst2.append(lst2obj.get_cumulative())

    icnt = 0
    v0 = np.zeros(2, dtype=float)
    v1 = np.zeros(2, dtype=float)
    err = np.zeros(2, dtype=float)

    # Process cumulative and incremental
    for idx in range(2):
        if idx > 0:
            max_pd = max_cumpd
        else:
            max_pd = max_incpd
        kper = lst1[idx]["stress_period"]
        kstp = lst1[idx]["time_step"]

        # Process each time step
        for jdx in range(kper.shape[0]):
            err[:] = 0.0
            t0 = lst1[idx][jdx]
            t1 = lst2[idx][jdx]

            if outfile is not None:
                maxcolname = 0
                for colname in t0.dtype.names:
                    maxcolname = max(maxcolname, len(colname))

                s = 2 * "\n"
                s += (
                    f"STRESS PERIOD: {kper[jdx] + 1} "
                    + f"TIME STEP: {kstp[jdx] + 1}"
                )
                f.write(s)

                if idx == 0:
                    f.write("\nINCREMENTAL BUDGET\n")
                else:
                    f.write("\nCUMULATIVE BUDGET\n")

                for i, colname in enumerate(t0.dtype.names):
                    if i == 0:
                        s = (
                            f"{'Budget Entry':<21} {'Model 1':>21} "
                            + f"{'Model 2':>21} {'Difference':>21}\n"
                        )
                        f.write(s)
                        s = 87 * "-" + "\n"
                        f.write(s)
                    diff = t0[colname] - t1[colname]
                    s = (
                        f"{colname:<21} {t0[colname]:>21} "
                        + f"{t1[colname]:>21} {diff:>21}\n"
                    )
                    f.write(s)

            v0[0] = t0["TOTAL_IN"]
            v1[0] = t1["TOTAL_IN"]
            if v0[0] > 0.0:
                err[0] = 100.0 * (v1[0] - v0[0]) / v0[0]
            v0[1] = t0["TOTAL_OUT"]
            v1[1] = t1["TOTAL_OUT"]
            if v0[1] > 0.0:
                err[1] = 100.0 * (v1[1] - v0[1]) / v0[1]
            for kdx, t in enumerate(err):
                if abs(t) > max_pd:
                    icnt += 1
                    if outfile is not None:
                        e = (
                            f'"{headers[idx]} {direction[kdx]}" '
                            + f"percent difference ({t})"
                            + f" for stress period {kper[jdx] + 1} "
                            + f"and time step {kstp[jdx] + 1} > {max_pd}."
                            + f" Reference value = {v0[kdx]}. "
                            + f"Simulated value = {v1[kdx]}."
                        )
                        e = textwrap.fill(
                            e,
                            width=70,
                            initial_indent="    ",
                            subsequent_indent="    ",
                        )
                        f.write(f"{e}\n")
                        f.write("\n")

    # Close output file
    if outfile is not None:
        f.close()

    # test for failure
    success = True
    if icnt > 0:
        success = False
    return success


def compare_swrbudget(
    namefile1: Optional[Union[str, os.PathLike]],
    namefile2: Optional[Union[str, os.PathLike]],
    max_cumpd=0.01,
    max_incpd=0.01,
    outfile: Optional[Union[str, os.PathLike]] = None,
    files1: Optional[
        Union[str, os.PathLike, List[Union[str, os.PathLike]]]
    ] = None,
    files2: Optional[
        Union[str, os.PathLike, List[Union[str, os.PathLike]]]
    ] = None,
):
    """Compare the SWR budget results from two simulations.

    Parameters
    ----------
    namefile1 : str or PathLike, optional
        namefile path for base model
    namefile2 : str or PathLike, optional
        namefile path for comparison model
    max_cumpd : float
        maximum percent discrepancy allowed for cumulative budget terms
        (default is 0.01)
    max_incpd : float
        maximum percent discrepancy allowed for incremental budget terms
        (default is 0.01)
    outfile : str or PathLike, optional
        budget comparison output file name. If outfile is None, no
        comparison output is saved. (default is None)
    files1 : str, PathLike, or list, optional
        base model output file. If files1 is not None, results
        will be extracted from files1 and namefile1 will not be used.
        (default is None)
    files2 : str, PathLike, or list, optional
        comparison model output file. If files2 is not None, results
        will be extracted from files2 and namefile2 will not be used.
        (default is None)

    Returns
    -------
    success : bool
        boolean indicating if the difference between budgets are less
        than max_cumpd and max_incpd

    """
    try:
        import flopy
    except:
        msg = "flopy not available - cannot use compare_swrbudget"
        raise ValueError(msg)

    # headers
    headers = ("INCREMENTAL", "CUMULATIVE")
    direction = ("IN", "OUT")

    # Get name of list files
    list1 = None
    if files1 is None:
        lst = get_entries_from_namefile(namefile1, "list")
        list1 = lst[0][0] if any(lst) else None
    else:
        for file in files1:
            if (
                "list" in os.path.basename(file).lower()
                or "lst" in os.path.basename(file).lower()
            ):
                list1 = file
                break
    list2 = None
    if files2 is None:
        lst = get_entries_from_namefile(namefile2, "list")
        list2 = lst[0][0] if any(lst) else None
    else:
        for file in files2:
            if (
                "list" in os.path.basename(file).lower()
                or "lst" in os.path.basename(file).lower()
            ):
                list2 = file
                break
    # Determine if there are two files to compare
    if list1 is None or list2 is None:
        return True

    # Initialize SWR budget objects
    lst1obj = flopy.utils.SwrListBudget(list1)
    lst2obj = flopy.utils.SwrListBudget(list2)

    # Determine if there any SWR entries in the budget file
    if not lst1obj.isvalid() or not lst2obj.isvalid():
        return True

    # Get numpy budget tables for list1
    lst1 = []
    lst1.append(lst1obj.get_incremental())
    lst1.append(lst1obj.get_cumulative())

    # Get numpy budget tables for list2
    lst2 = []
    lst2.append(lst2obj.get_incremental())
    lst2.append(lst2obj.get_cumulative())

    icnt = 0
    v0 = np.zeros(2, dtype=float)
    v1 = np.zeros(2, dtype=float)
    err = np.zeros(2, dtype=float)

    # Open output file
    if outfile is not None:
        f = open(outfile, "w")

    # Process cumulative and incremental
    for idx in range(2):
        if idx > 0:
            max_pd = max_cumpd
        else:
            max_pd = max_incpd
        kper = lst1[idx]["stress_period"]
        kstp = lst1[idx]["time_step"]

        # Process each time step
        for jdx in range(kper.shape[0]):
            err[:] = 0.0
            t0 = lst1[idx][jdx]
            t1 = lst2[idx][jdx]

            if outfile is not None:
                maxcolname = 0
                for colname in t0.dtype.names:
                    maxcolname = max(maxcolname, len(colname))

                s = 2 * "\n"
                s += (
                    f"STRESS PERIOD: {kper[jdx] + 1} "
                    + f"TIME STEP: {kstp[jdx] + 1}"
                )
                f.write(s)

                if idx == 0:
                    f.write("\nINCREMENTAL BUDGET\n")
                else:
                    f.write("\nCUMULATIVE BUDGET\n")

                for i, colname in enumerate(t0.dtype.names):
                    if i == 0:
                        s = (
                            f"{'Budget Entry':<21} {'Model 1':>21} "
                            + f"{'Model 2':>21} {'Difference':>21}\n"
                        )
                        f.write(s)
                        s = 87 * "-" + "\n"
                        f.write(s)
                    diff = t0[colname] - t1[colname]
                    s = (
                        f"{colname:<21} {t0[colname]:>21} "
                        + f"{t1[colname]:>21} {diff:>21}\n"
                    )
                    f.write(s)

            v0[0] = t0["TOTAL_IN"]
            v1[0] = t1["TOTAL_IN"]
            if v0[0] > 0.0:
                err[0] = 100.0 * (v1[0] - v0[0]) / v0[0]
            v0[1] = t0["TOTAL_OUT"]
            v1[1] = t1["TOTAL_OUT"]
            if v0[1] > 0.0:
                err[1] = 100.0 * (v1[1] - v0[1]) / v0[1]
            for kdx, t in enumerate(err):
                if abs(t) > max_pd:
                    icnt += 1
                    e = (
                        f'"{headers[idx]} {direction[kdx]}" '
                        + f"percent difference ({t})"
                        + f" for stress period {kper[jdx] + 1} "
                        + f"and time step {kstp[jdx] + 1} > {max_pd}."
                        + f" Reference value = {v0[kdx]}. "
                        + f"Simulated value = {v1[kdx]}."
                    )
                    e = textwrap.fill(
                        e,
                        width=70,
                        initial_indent="    ",
                        subsequent_indent="    ",
                    )
                    f.write(f"{e}\n")
                    f.write("\n")

    # Close output file
    if outfile is not None:
        f.close()

    # test for failure
    success = True
    if icnt > 0:
        success = False
    return success


def compare_heads(
    namefile1: Optional[Union[str, os.PathLike]],
    namefile2: Optional[Union[str, os.PathLike]],
    precision="auto",
    text="head",
    text2=None,
    htol=0.001,
    outfile: Optional[Union[str, os.PathLike]] = None,
    files1: Optional[
        Union[str, os.PathLike, List[Union[str, os.PathLike]]]
    ] = None,
    files2: Optional[
        Union[str, os.PathLike, List[Union[str, os.PathLike]]]
    ] = None,
    difftol=False,
    verbose=False,
    exfile: Optional[Union[str, os.PathLike]] = None,
    exarr=None,
    maxerr=None,
):
    """Compare the head results from two simulations.

    Parameters
    ----------
    namefile1 : str or PathLike
        namefile path for base model
    namefile2 : str or PathLike
        namefile path for comparison model
    precision : str
        precision for binary head file ("auto", "single", or "double")
        default is "auto"
    htol : float
        maximum allowed head difference (default is 0.001)
    outfile : str or PathLike
        head comparison output file name. If outfile is None, no
        comparison output is saved. (default is None)
    files1 : str or PathLike, or List of str or PathLike
        base model output files. If files1 is not None, results
        will be extracted from files1 and namefile1 will not be used.
        (default is None)
    files2 : str or PathLike, or List of str or PathLike
        comparison model output files. If files2 is not None, results
        will be extracted from files2 and namefile2 will not be used.
        (default is None)
    difftol : bool
        boolean determining if the absolute value of the head
        difference greater than htol should be evaluated (default is False)
    verbose : bool
        boolean indicating if verbose output should be written to the
        terminal (default is False)
    exfile : str or PathLike, optional
        path to a file with exclusion array data. Head differences will not
        be evaluated where exclusion array values are greater than zero.
        (default is None)
    exarr : numpy.ndarry
        exclusion array. Head differences will not be evaluated where
        exclusion array values are greater than zero. (default is None).
    maxerr : int
        maximum number of head difference greater than htol that should be
        reported. If maxerr is None, all head difference greater than htol
        will be reported. (default is None)

    Returns
    -------
    success : bool
        boolean indicating if the head differences are less than htol.

    """

    if text2 is None:
        text2 = text

    dbs = "DATA(BINARY)"

    # Get head info for namefile1
    hfpth1 = None
    status1 = dbs
    if files1 is None:
        # Get oc info, and return if OC not included in models
        ocf1 = get_entries_from_namefile(namefile1, "OC")
        if not any(ocf1) is None:
            return True

        hu1, hfpth1, du1, _ = ModflowOc.get_ocoutput_units(ocf1[0][0])
        if text.lower() == "head":
            iut = hu1
        elif text.lower() == "drawdown":
            iut = du1
        if iut != 0:
            entries = get_entries_from_namefile(namefile1, unit=abs(iut))
            hfpth1 = entries[0][0] if any(entries) else None
            status1 = entries[0][1] if any(entries) else None

    else:
        if isinstance(files1, (str, os.PathLike)):
            files1 = [files1]
        for file in files1:
            if text.lower() == "head":
                if (
                    "hds" in os.path.basename(file).lower()
                    or "hed" in os.path.basename(file).lower()
                ):
                    hfpth1 = file
                    break
            elif text.lower() == "drawdown":
                if "ddn" in os.path.basename(file).lower():
                    hfpth1 = file
                    break
            elif text.lower() == "concentration":
                if "ucn" in os.path.basename(file).lower():
                    hfpth1 = file
                    break
            else:
                hfpth1 = file
                break

    # Get head info for namefile2
    hfpth2 = None
    status2 = dbs
    if files2 is None:
        # Get oc info, and return if OC not included in models
        ocf2 = get_entries_from_namefile(namefile2, "OC")
        if not any(ocf2):
            return True

        hu2, hfpth2, du2, dfpth2 = ModflowOc.get_ocoutput_units(ocf2[0][0])
        if text.lower() == "head":
            iut = hu2
        elif text.lower() == "drawdown":
            iut = du2
        if iut != 0:
            entries = get_entries_from_namefile(namefile2, unit=abs(iut))
            hfpth2 = entries[0][0] if any(entries) else None
            status2 = entries[0][1] if any(entries) else None
    else:
        if isinstance(files2, (str, os.PathLike)):
            files2 = [files2]
        for file in files2:
            if text2.lower() == "head":
                if (
                    "hds" in os.path.basename(file).lower()
                    or "hed" in os.path.basename(file).lower()
                ):
                    hfpth2 = file
                    break
            elif text2.lower() == "drawdown":
                if "ddn" in os.path.basename(file).lower():
                    hfpth2 = file
                    break
            elif text2.lower() == "concentration":
                if "ucn" in os.path.basename(file).lower():
                    hfpth2 = file
                    break
            else:
                hfpth2 = file
                break

    # confirm that there are two files to compare
    if hfpth1 is None or hfpth2 is None:
        print("hfpth1 or hfpth2 is None")
        print(f"hfpth1: {hfpth1}")
        print(f"hfpth2: {hfpth2}")
        return True

    # make sure the file paths exist
    if not os.path.isfile(hfpth1) or not os.path.isfile(hfpth2):
        print("hfpth1 or hfpth2 is not a file")
        print(f"hfpth1 isfile: {os.path.isfile(hfpth1)}")
        print(f"hfpth2 isfile: {os.path.isfile(hfpth2)}")
        return False

    # Open output file
    if outfile is not None:
        f = open(outfile, "w")
        f.write(f"Performing {text.upper()} to {text2.upper()} comparison\n")

        if exfile is not None:
            f.write(f"Using exclusion file {exfile}\n")
        if exarr is not None:
            f.write("Using exclusion array\n")

        msg = f"{hfpth1} is a "
        if status1 == dbs:
            msg += "binary file."
        else:
            msg += "ascii file."
        f.write(msg + "\n")
        msg = f"{hfpth2} is a "
        if status2 == dbs:
            msg += "binary file."
        else:
            msg += "ascii file."
        f.write(msg + "\n")

    # Process exclusion data
    exd = None
    # get data from exclusion file
    if exfile is not None:
        e = None
        if isinstance(exfile, (str, os.PathLike)):
            try:
                exd = np.genfromtxt(exfile).flatten()
            except:
                e = (
                    "Could not read exclusion "
                    + f"file {os.path.basename(exfile)}"
                )
                print(e)
                return False
        else:
            e = "exfile is not a valid file path"
            print(e)
            return False

    # process exclusion array
    if exarr is not None:
        e = None
        if isinstance(exarr, np.ndarray):
            if exd is None:
                exd = exarr.flatten()
            else:
                exd += exarr.flatten()
        else:
            e = "exarr is not a numpy array"
            print(e)
            return False

    # Get head objects
    status1 = status1.upper()
    unstructured1 = False
    if status1 == dbs:
        headobj1 = HeadFile(
            hfpth1, precision=precision, verbose=verbose, text=text
        )
        txt = headobj1.recordarray["text"][0]
        if isinstance(txt, bytes):
            txt = txt.decode("utf-8")
        if "HEADU" in txt:
            unstructured1 = True
            headobj1 = HeadUFile(hfpth1, precision=precision, verbose=verbose)
    else:
        headobj1 = FormattedHeadFile(hfpth1, verbose=verbose, text=text)

    status2 = status2.upper()
    unstructured2 = False
    if status2 == dbs:
        headobj2 = HeadFile(
            hfpth2, precision=precision, verbose=verbose, text=text2
        )
        txt = headobj2.recordarray["text"][0]
        if isinstance(txt, bytes):
            txt = txt.decode("utf-8")
        if "HEADU" in txt:
            unstructured2 = True
            headobj2 = HeadUFile(hfpth2, precision=precision, verbose=verbose)
    else:
        headobj2 = FormattedHeadFile(hfpth2, verbose=verbose, text=text2)

    # get times
    times1 = headobj1.get_times()
    times2 = headobj2.get_times()
    for t1, t2 in zip(times1, times2):
        if not np.allclose([t1], [t2]):
            msg = "times in two head files are not " + f"equal ({t1},{t2})"
            raise ValueError(msg)

    kstpkper = headobj1.get_kstpkper()

    line_separator = 15 * "-"
    header = (
        f"{' ':>15s} {' ':>15s} {'MAXIMUM':>15s} {'EXCEEDS':>15s}\n"
        + f"{'STRESS PERIOD':>15s} {'TIME STEP':>15s} "
        + f"{'HEAD DIFFERENCE':>15s} {'CRITERIA':>15s}\n"
        + f"{line_separator:>15s} {line_separator:>15s} "
        + f"{line_separator:>15s} {line_separator:>15s}\n"
    )

    if verbose:
        print(f"Comparing results for {len(times1)} times")

    icnt = 0
    # Process cumulative and incremental
    for idx, (t1, t2) in enumerate(zip(times1, times2)):
        h1 = headobj1.get_data(totim=t1)
        if unstructured1:
            temp = np.array([])
            for a in h1:
                temp = np.hstack((temp, a))
            h1 = temp
        h2 = headobj2.get_data(totim=t2)
        if unstructured2:
            temp = np.array([])
            for a in h2:
                temp = np.hstack((temp, a))
            h2 = temp

        if exd is not None:
            # reshape exd to the shape of the head arrays
            if idx == 0:
                e = (
                    f"shape of exclusion data ({exd.shape})"
                    + "can not be reshaped to the size of the "
                    + f"head arrays ({h1.shape})"
                )
                if h1.flatten().shape != exd.shape:
                    raise ValueError(e)
                exd = exd.reshape(h1.shape)
                iexd = exd > 0

            # reset h1 and h2 to the same value in the excluded area
            h1[iexd] = 0.0
            h2[iexd] = 0.0

        if difftol:
            diffmax, indices = _difftol(h1, h2, htol)
        else:
            diffmax, indices = _diffmax(h1, h2)

        if outfile is not None:
            if idx < 1:
                f.write(header)
            if diffmax > htol:
                sexceed = "*"
            else:
                sexceed = ""
            kk1 = kstpkper[idx][1] + 1
            kk0 = kstpkper[idx][0] + 1
            f.write(f"{kk1:15d} {kk0:15d} {diffmax:15.6g} {sexceed:15s}\n")

        if diffmax >= htol:
            icnt += 1
            if outfile is not None:
                if difftol:
                    ee = (
                        "Maximum absolute head difference "
                        + f"({diffmax}) -- "
                        + f"{htol} tolerance exceeded at "
                        + f"{indices[0].shape[0]} node location(s)"
                    )
                else:
                    ee = (
                        "Maximum absolute head difference "
                        + f"({diffmax}) exceeded "
                        + f"at {indices[0].shape[0]} node location(s)"
                    )
                e = textwrap.fill(
                    ee + ":",
                    width=70,
                    initial_indent="  ",
                    subsequent_indent="  ",
                )

                if verbose:
                    f.write(f"{ee}\n")
                    print(ee + f" at time {t1}")

                e = ""
                ncells = h1.flatten().shape[0]
                fmtn = "{:" + f"{len(str(ncells))}" + "d}"
                for itupe in indices:
                    for jdx, ind in enumerate(itupe):
                        iv = np.unravel_index(ind, h1.shape)
                        iv = tuple(i + 1 for i in iv)
                        v1 = h1.flatten()[ind]
                        v2 = h2.flatten()[ind]
                        d12 = v1 - v2
                        # e += '    ' + fmtn.format(jdx + 1) + ' node: '
                        # e += fmtn.format(ind + 1)  # convert to one-based
                        e += "    " + fmtn.format(jdx + 1)
                        e += f" {iv}"
                        e += " -- "
                        e += f"h1: {v1:20} "
                        e += f"h2: {v2:20} "
                        e += f"diff: {d12:20}\n"
                        if isinstance(maxerr, int):
                            if jdx + 1 >= maxerr:
                                break
                    if verbose:
                        f.write(f"{e}\n")
                # Write header again, unless it is the last record
                if verbose:
                    if idx + 1 < len(times1):
                        f.write(f"\n{header}")

    # Close output file
    if outfile is not None:
        f.close()

    # test for failure
    success = True
    if icnt > 0:
        success = False
    return success


def compare_concentrations(
    namefile1: Union[str, os.PathLike],
    namefile2: Union[str, os.PathLike],
    precision="auto",
    ctol=0.001,
    outfile: Optional[Union[str, os.PathLike]] = None,
    files1: Optional[
        Union[str, os.PathLike, List[Union[str, os.PathLike]]]
    ] = None,
    files2: Optional[
        Union[str, os.PathLike, List[Union[str, os.PathLike]]]
    ] = None,
    difftol=False,
    verbose=False,
):
    """Compare the mt3dms and mt3dusgs concentration results from two
    simulations.

    Parameters
    ----------
    namefile1 : str or PathLike
        namefile path for base model
    namefile2 : str or PathLike
        namefile path for comparison model
    precision : str
        precision for binary head file ("auto", "single", or "double")
        default is "auto"
    ctol : float
        maximum allowed concentration difference (default is 0.001)
    outfile : str or PathLike, optional
        concentration comparison output file name. If outfile is None, no
        comparison output is saved. (default is None)
    files1 : str, PathLike, or list, optional
        base model output file. If files1 is not None, results
        will be extracted from files1 and namefile1 will not be used.
        (default is None)
    files2 : str, PathLike, or list, optional
        comparison model output file. If files2 is not None, results
        will be extracted from files2 and namefile2 will not be used.
        (default is None)
    difftol : bool
        boolean determining if the absolute value of the concentration
        difference greater than ctol should be evaluated (default is False)
    verbose : bool
        boolean indicating if verbose output should be written to the
        terminal (default is False)

    Returns
    -------
    success : bool
        boolean indicating if the concentration differences are less than
        ctol.

    Returns
    -------

    """
    try:
        import flopy
    except:
        msg = "flopy not available - cannot use compare_concs"
        raise ValueError(msg)

    # list of valid extensions
    valid_ext = ["ucn"]

    # Get info for first ucn file
    ufpth1 = None
    if files1 is None:
        for ext in valid_ext:
            ucn = get_entries_from_namefile(namefile1, extension=ext)
            ufpth = ucn[0][0] if any(ucn) else None
            if ufpth is not None:
                ufpth1 = ufpth
                break
        if ufpth1 is None:
            ufpth1 = os.path.join(os.path.dirname(namefile1), "MT3D001.UCN")
    else:
        if isinstance(files1, (str, os.PathLike)):
            files1 = [files1]
        for file in files1:
            for ext in valid_ext:
                if ext in os.path.basename(file).lower():
                    ufpth1 = file
                    break

    # Get info for second ucn file
    ufpth2 = None
    if files2 is None:
        for ext in valid_ext:
            ucn = get_entries_from_namefile(namefile2, extension=ext)
            ufpth = ucn[0][0] if any(ucn) else None
            if ufpth is not None:
                ufpth2 = ufpth
                break
        if ufpth2 is None:
            ufpth2 = os.path.join(os.path.dirname(namefile2), "MT3D001.UCN")
    else:
        if isinstance(files2, (str, os.PathLike)):
            files2 = [files2]
        for file in files2:
            for ext in valid_ext:
                if ext in os.path.basename(file).lower():
                    ufpth2 = file
                    break

    # confirm that there are two files to compare
    if ufpth1 is None or ufpth2 is None:
        if ufpth1 is None:
            print("  UCN file 1 not set")
        if ufpth2 is None:
            print("  UCN file 2 not set")
        return True

    if not os.path.isfile(ufpth1) or not os.path.isfile(ufpth2):
        if not os.path.isfile(ufpth1):
            print(f"  {ufpth1} does not exist")
        if not os.path.isfile(ufpth2):
            print(f"  {ufpth2} does not exist")
        return True

    # Open output file
    if outfile is not None:
        f = open(outfile, "w")

    # Get stage objects
    uobj1 = flopy.utils.UcnFile(ufpth1, precision=precision, verbose=verbose)
    uobj2 = flopy.utils.UcnFile(ufpth2, precision=precision, verbose=verbose)

    # get times
    times1 = uobj1.get_times()
    times2 = uobj2.get_times()
    nt1 = len(times1)
    nt2 = len(times2)
    nt = min(nt1, nt2)

    for t1, t2 in zip(times1, times2):
        if not np.allclose([t1], [t2]):
            msg = f"times in two ucn files are not equal ({t1},{t2})"
            raise ValueError(msg)

    if nt == nt1:
        kstpkper = uobj1.get_kstpkper()
    else:
        kstpkper = uobj2.get_kstpkper()

    line_separator = 15 * "-"
    header = (
        f"{' ':>15s} {' ':>15s} {'MAXIMUM':>15s}\n"
        + f"{'STRESS PERIOD':>15s} {'TIME STEP':>15s} "
        + f"{'CONC DIFFERENCE':>15s}\n"
        + f"{line_separator:>15s} "
        + f"{line_separator:>15s} "
        + f"{line_separator:>15s}\n"
    )

    if verbose:
        print(f"Comparing results for {len(times1)} times")

    icnt = 0
    # Process cumulative and incremental
    for idx, time in enumerate(times1[0:nt]):
        try:
            u1 = uobj1.get_data(totim=time)
            u2 = uobj2.get_data(totim=time)

            if difftol:
                diffmax, indices = _difftol(u1, u2, ctol)
            else:
                diffmax, indices = _diffmax(u1, u2)

            if outfile is not None:
                if idx < 1:
                    f.write(header)
                f.write(
                    f"{kstpkper[idx][1] + 1:15d} "
                    + f"{kstpkper[idx][0] + 1:15d} "
                    + f"{diffmax:15.6g}\n"
                )

            if diffmax >= ctol:
                icnt += 1
                if outfile is not None:
                    if difftol:
                        ee = (
                            f"Maximum concentration difference ({diffmax})"
                            + f" -- {ctol} tolerance exceeded at "
                            + f"{indices[0].shape[0]} node location(s)"
                        )
                    else:
                        ee = (
                            "Maximum concentration difference "
                            + f"({diffmax}) exceeded "
                            + f"at {indices[0].shape[0]} node location(s)"
                        )
                    e = textwrap.fill(
                        ee + ":",
                        width=70,
                        initial_indent="  ",
                        subsequent_indent="  ",
                    )
                    f.write(f"{e}\n")
                    if verbose:
                        print(ee + f" at time {time}")
                    e = ""
                    for itupe in indices:
                        for ind in itupe:
                            e += f"{ind + 1} "  # convert to one-based
                    e = textwrap.fill(
                        e,
                        width=70,
                        initial_indent="    ",
                        subsequent_indent="    ",
                    )
                    f.write(f"{e}\n")
                    # Write header again, unless it is the last record
                    if idx + 1 < len(times1):
                        f.write(f"\n{header}")
        except:
            print(f"  could not process time={time}")
            print("  terminating ucn processing...")
            break

    # Close output file
    if outfile is not None:
        f.close()

    # test for failure
    success = True
    if icnt > 0:
        success = False
    return success


def compare_stages(
    namefile1: Union[str, os.PathLike] = None,
    namefile2: Union[str, os.PathLike] = None,
    files1: Optional[
        Union[str, os.PathLike, List[Union[str, os.PathLike]]]
    ] = None,
    files2: Optional[
        Union[str, os.PathLike, List[Union[str, os.PathLike]]]
    ] = None,
    htol=0.001,
    outfile: Optional[Union[str, os.PathLike]] = None,
    difftol=False,
    verbose=False,
):
    """Compare SWR process stage results from two simulations.

    Parameters
    ----------
    namefile1 : str or PathLike
        namefile path for base model
    namefile2 : str or PathLike
        namefile path for comparison model
    precision : str
        precision for binary head file ("auto", "single", or "double")
        default is "auto"
    htol : float
        maximum allowed stage difference (default is 0.001)
    outfile : str or PathLike, optional
        head comparison output file name. If outfile is None, no
        comparison output is saved. (default is None)
    files1 : str, PathLike, or list, optional
        base model output file. If files1 is not None, results
        will be extracted from files1 and namefile1 will not be used.
        (default is None)
    files2 : str, PathLike, or list, optional
        comparison model output file. If files2 is not None, results
        will be extracted from files2 and namefile2 will not be used.
        (default is None)
    difftol : bool
        boolean determining if the absolute value of the stage
        difference greater than htol should be evaluated (default is False)
    verbose : bool
        boolean indicating if verbose output should be written to the
        terminal (default is False)

    Returns
    -------
    success : bool
        boolean indicating if the stage differences are less than htol.

    """
    try:
        import flopy
    except:
        msg = "flopy not available - cannot use compare_stages"
        raise ValueError(msg)

    # list of valid extensions
    valid_ext = ["stg"]

    # Get info for first stage file
    sfpth1 = None
    if namefile1 is not None:
        for ext in valid_ext:
            stg = get_entries_from_namefile(namefile1, extension=ext)
            sfpth = stg[0][0] if any(stg) else None
            if sfpth is not None:
                sfpth1 = sfpth
                break
    elif files1 is not None:
        if isinstance(files1, (str, os.PathLike)):
            files1 = [files1]
        for file in files1:
            for ext in valid_ext:
                if ext in os.path.basename(file).lower():
                    sfpth1 = file
                    break

    # Get info for second stage file
    sfpth2 = None
    if namefile2 is not None:
        for ext in valid_ext:
            stg = get_entries_from_namefile(namefile2, extension=ext)
            sfpth = stg[0][0] if any(stg) else None
            if sfpth is not None:
                sfpth2 = sfpth
                break
    elif files2 is not None:
        if isinstance(files2, (str, os.PathLike)):
            files2 = [files2]
        for file in files2:
            for ext in valid_ext:
                if ext in os.path.basename(file).lower():
                    sfpth2 = file
                    break

    # confirm that there are two files to compare
    if sfpth1 is None or sfpth2 is None:
        print("spth1 or spth2 is None")
        print(f"spth1: {sfpth1}")
        print(f"spth2: {sfpth2}")
        return False

    if not os.path.isfile(sfpth1) or not os.path.isfile(sfpth2):
        print("spth1 or spth2 is not a file")
        print(f"spth1 isfile: {os.path.isfile(sfpth1)}")
        print(f"spth2 isfile: {os.path.isfile(sfpth2)}")
        return False

    # Open output file
    if outfile is not None:
        f = open(outfile, "w")

    # Get stage objects
    sobj1 = flopy.utils.SwrStage(sfpth1, verbose=verbose)
    sobj2 = flopy.utils.SwrStage(sfpth2, verbose=verbose)

    # get totim
    times1 = sobj1.get_times()

    # get kswr, kstp, and kper
    kk = sobj1.get_kswrkstpkper()

    line_separator = 15 * "-"
    header = (
        f"{' ':>15s} {' ':>15s} {' ':>15s} {'MAXIMUM':>15s}\n"
        + f"{'STRESS PERIOD':>15s} "
        + f"{'TIME STEP':>15s} "
        + f"{'SWR TIME STEP':>15s} "
        + f"{'STAGE DIFFERENCE':>15s}\n"
        + f"{line_separator:>15s} "
        + f"{line_separator:>15s} "
        + f"{line_separator:>15s} "
        + f"{line_separator:>15s}\n"
    )

    if verbose:
        print(f"Comparing results for {len(times1)} times")

    icnt = 0
    # Process stage data
    for idx, (kon, time) in enumerate(zip(kk, times1)):
        s1 = sobj1.get_data(totim=time)
        s2 = sobj2.get_data(totim=time)

        if s1 is None or s2 is None:
            continue

        s1 = s1["stage"]
        s2 = s2["stage"]

        if difftol:
            diffmax, indices = _difftol(s1, s2, htol)
        else:
            diffmax, indices = _diffmax(s1, s2)

        if outfile is not None:
            if idx < 1:
                f.write(header)
            f.write(
                f"{kon[2] + 1:15d} "
                + f"{kon[1] + 1:15d} "
                + f"{kon[0] + 1:15d} "
                + f"{diffmax:15.6g}\n"
            )

        if diffmax >= htol:
            icnt += 1
            if outfile is not None:
                if difftol:
                    ee = (
                        f"Maximum head difference ({diffmax}) -- "
                        + f"{htol} tolerance exceeded at "
                        + f"{indices[0].shape[0]} node location(s)"
                    )
                else:
                    ee = (
                        "Maximum head difference "
                        + f"({diffmax}) exceeded "
                        + f"at {indices[0].shape[0]} node location(s):"
                    )
                e = textwrap.fill(
                    ee + ":",
                    width=70,
                    initial_indent="  ",
                    subsequent_indent="  ",
                )
                f.write(f"{e}\n")
                if verbose:
                    print(ee + f" at time {time}")
                e = ""
                for itupe in indices:
                    for ind in itupe:
                        e += f"{ind + 1} "  # convert to one-based
                e = textwrap.fill(
                    e,
                    width=70,
                    initial_indent="    ",
                    subsequent_indent="    ",
                )
                f.write(f"{e}\n")
                # Write header again, unless it is the last record
                if idx + 1 < len(times1):
                    f.write(f"\n{header}")

    # Close output file
    if outfile is not None:
        f.close()

    # test for failure
    success = True
    if icnt > 0:
        success = False
    return success


def compare(
    namefile1: Union[str, os.PathLike] = None,
    namefile2: Union[str, os.PathLike] = None,
    precision="auto",
    max_cumpd=0.01,
    max_incpd=0.01,
    htol=0.001,
    outfile1: Optional[Union[str, os.PathLike]] = None,
    outfile2: Optional[Union[str, os.PathLike]] = None,
    files1: Optional[
        Union[str, os.PathLike, List[Union[str, os.PathLike]]]
    ] = None,
    files2: Optional[
        Union[str, os.PathLike, List[Union[str, os.PathLike]]]
    ] = None,
):
    """Compare the budget and head results for two MODFLOW-based model
    simulations.

    Parameters
    ----------
    namefile1 : str or PathLike, optional
        namefile path for base model
    namefile2 : str or PathLike, optional
        namefile path for comparison model
    precision : str
        precision for binary head file ("auto", "single", or "double")
        default is "auto"
    max_cumpd : float
        maximum percent discrepancy allowed for cumulative budget terms
        (default is 0.01)
    max_incpd : float
        maximum percent discrepancy allowed for incremental budget terms
        (default is 0.01)
    htol : float
        maximum allowed head difference (default is 0.001)
    outfile1 : str or PathLike, optional
        budget comparison output file name. If outfile1 is None, no budget
        comparison output is saved. (default is None)
    outfile2 : str or PathLike, optional
        head comparison output file name. If outfile2 is None, no head
        comparison output is saved. (default is None)
    files1 : str, PathLike, or list, optional
        base model output file. If files1 is not None, results
        will be extracted from files1 and namefile1 will not be used.
        (default is None)
    files2 : str, PathLike, or list, optional
        comparison model output file. If files2 is not None, results
        will be extracted from files2 and namefile2 will not be used.
        (default is None)

    Returns
    -------
    success : bool
        boolean indicating if the budget and head differences are less than
        max_cumpd, max_incpd, and htol.

    """

    # Compare budgets from the list files in namefile1 and namefile2
    success1 = compare_budget(
        namefile1,
        namefile2,
        max_cumpd=max_cumpd,
        max_incpd=max_incpd,
        outfile=outfile1,
        files1=files1,
        files2=files2,
    )
    success2 = compare_heads(
        namefile1,
        namefile2,
        precision=precision,
        htol=htol,
        outfile=outfile2,
        files1=files1,
        files2=files2,
    )
    success = False
    if success1 and success2:
        success = True
    return success


def eval_bud_diff(fpth: Union[str, os.PathLike], b0, b1, ia=None, dtol=1e-6):
    # To use this eval_bud_diff function on a gwf or gwt budget file,
    # the function may need ia, in order to exclude comparison of the residual
    # term, which is stored in the diagonal position of the flowja array.
    #  The following code can be used to extract ia from the grb file.
    # get ia/ja from binary grid file
    # fname = '{}.dis.grb'.format(os.path.basename(sim.name))
    # fpth = os.path.join(sim.simpath, fname)
    # grbobj = flopy.mf6.utils.MfGrdFile(fpth)
    # ia = grbobj._datadict['IA'] - 1

    diffmax = 0.0
    difftag = "None"
    difftime = None
    fail = False

    # build list of cbc data to retrieve
    avail = b0.get_unique_record_names()

    # initialize list for storing totals for each budget term terms
    cbc_keys = []
    for t in avail:
        if isinstance(t, bytes):
            t = t.decode()
        t = t.strip()
        cbc_keys.append(t)

    # open a summary file and write header
    f = open(fpth, "w")
    line = f"{'Time':15s}"
    line += f" {'Datatype':15s}"
    line += f" {'File 1':15s}"
    line += f" {'File 2':15s}"
    line += f" {'Difference':15s}"
    f.write(line + "\n")
    f.write(len(line) * "-" + "\n")

    # get data from cbc file
    kk = b0.get_kstpkper()
    times = b0.get_times()
    for idx, (k, t) in enumerate(zip(kk, times)):
        v0sum = 0.0
        v1sum = 0.0
        for key in cbc_keys:
            v0 = b0.get_data(kstpkper=k, text=key)[0]
            v1 = b1.get_data(kstpkper=k, text=key)[0]
            if isinstance(v0, np.recarray):
                v0 = v0["q"].sum()
                v1 = v1["q"].sum()
            else:
                v0 = v0.flatten()
                v1 = v1.flatten()
                if key == "FLOW-JA-FACE":
                    # Set residual (stored in diagonal of flowja) to zero
                    if ia is None:
                        raise Exception("ia is required for model flowja")
                    idiagidx = ia[:-1]
                    v0[idiagidx] = 0.0
                    v1[idiagidx] = 0.0
                v0 = v0.sum()
                v1 = v1.sum()

            # sum all of the values
            if key != "AUXILIARY":
                v0sum += v0
                v1sum += v1

            diff = v0 - v1
            if abs(diff) > abs(diffmax):
                diffmax = diff
                difftag = key
                difftime = t
            if abs(diff) > dtol:
                fail = True
            line = f"{t:15g}"
            line += f" {key:15s}"
            line += f" {v0:15g}"
            line += f" {v1:15g}"
            line += f" {diff:15g}"
            f.write(line + "\n")

    # evaluate the sums
    diff = v0sum - v1sum
    if abs(diff) > dtol:
        fail = True
    line = f"{t:15g}"
    line += f" {'TOTAL':15s}"
    line += f" {v0sum:15g}"
    line += f" {v1sum:15g}"
    line += f" {diff:15g}"
    f.write(line + "\n")

    msg = f"\nSummary of changes in {os.path.basename(fpth)}\n"
    msg += "-" * 72 + "\n"
    msg += f"Maximum cbc difference:        {diffmax}\n"
    msg += f"Maximum cbc difference time:   {difftime}\n"
    msg += f"Maximum cbc datatype:          {difftag}\n"
    if fail:
        msg += f"Maximum cbc criteria exceeded:  {dtol}"
    assert not fail, msg

    # close summary file and print the final message
    f.close()
    print(msg)

    msg = f"sum of first cbc file flows ({v0sum}) " + f"exceeds dtol ({dtol})"
    assert abs(v0sum) < dtol, msg

    msg = f"sum of second cbc file flows ({v1sum}) " + f"exceeds dtol ({dtol})"
    assert abs(v1sum) < dtol, msg
