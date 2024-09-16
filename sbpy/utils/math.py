# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np


def twovec(axdef, indexa, plndef, indexp):
    """Transformation matrix to a new coordinate defined by two input vectors.

    Parameters
    ----------
    axdef : array-like float containing 3 elements
        The vector (x, y, z) that defines one of the principal axes of the new
        coordinate frame.
    indexa : int 0, 1, or 2
        Specify which of the three coordinate axes is defined by `axdef`.  0 for
        x-axis, 1 for y-axis, and 2 for z-axis
    plndef : array-like float containing 3 elements
        The vector (x, y, z) that defines (with `axdef`) a principal plane of
        the new coordinate frame.
    indexp : int 0, 1, or 2
        Specify the second axis of the principal frame determined by `axdef` and
        `plndef`

    Returns
    -------
    numpy array of shape 3x3 The transformation matrix that convert a vector
    from the old coordinate to the coordinate frame defined by the input vectors
    via a dot product.

    Notes
    -----
    This routine is directly translated form SPICE lib routine twovec.f (cf.
    SPICE manual http://www.nis.lanl.gov/~esm/idl/spice-dlm/spice-t.html#TWOVEC)

    The indexing of array elements are different in FORTRAN (that SPICE is
    originally based) from Python.  Here 0-based index is used.

    Note that the original twovec.f in SPICE toolkit returns a matrix that
    converts a vector in the new frame to the original frame, opposite to what
    is implemented here.

    """

    axdef = np.asarray(axdef).flatten()
    plndef = np.asarray(plndef).flatten()

    if np.linalg.norm(np.cross(axdef, plndef)) == 0:
        raise RuntimeError(
            "The input vectors AXDEF and PLNDEF are linearly"
            " correlated and can't define a coordinate frame."
        )

    M = np.eye(3)
    i1 = indexa % 3
    i2 = (indexa + 1) % 3
    i3 = (indexa + 2) % 3

    M[i1, :] = axdef / np.linalg.norm(axdef)
    if indexp == i2:
        xv = np.cross(axdef, plndef)
        M[i3, :] = xv / np.linalg.norm(xv)
        xv = np.cross(xv, axdef)
        M[i2, :] = xv / np.linalg.norm(xv)
    else:
        xv = np.cross(plndef, axdef)
        M[i2, :] = xv / np.linalg.norm(xv)
        xv = np.cross(axdef, xv)
        M[i3, :] = xv / np.linalg.norm(xv)

    return M
