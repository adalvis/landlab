#! /usr/bin/env python
"""Write structured grids to NetCDF files.

Write netcdf
++++++++++++

.. autosummary::

    ~write_netcdf
"""
import pathlib

import numpy as np
import xarray as xr

from landlab.io.netcdf._constants import _AXIS_COORDINATE_NAMES
from landlab.io.netcdf._constants import _AXIS_DIMENSION_NAMES
from landlab.io.netcdf._constants import _NP_TO_NC_TYPE


def _set_netcdf_attributes(root, attrs):
    """Set attributes of a netcdf file.

    Set attributes of the netCDF Database object, *root*. Attributes are
    given as key/value pairs from *attrs*.

    Parameters
    ----------
    root : netcdf_file
        A NetCDF file.
    attrs : dict
        Attributes as key-value pairs.
    """
    for key, val in attrs.items():
        setattr(root, key, val)


def _get_dimension_names(shape):
    """Get dimension names.

    Parameters
    ----------
    shape : tuple of int
        Shape of a structured grid.

    Returns
    -------
    tuple of str
        Dimension names for the NetCDF file.

    Examples
    --------
    >>> from landlab.io.netcdf.write import _get_dimension_names
    >>> _get_dimension_names((4,))
    ['ni']
    >>> _get_dimension_names((4, 5))
    ['nj', 'ni']
    >>> _get_dimension_names((4, 5, 6))
    ['nk', 'nj', 'ni']
    """
    names = _AXIS_DIMENSION_NAMES[-1 : -(len(shape) + 1) : -1]
    return names[::-1]


def _get_dimension_sizes(shape):
    """Get dimension sizes.

    Parameters
    ----------
    shape : tuple of int
        Shape of a structured grid.

    Returns
    -------
    dict
        Dimension sizes.

    Examples
    --------
    >>> from landlab.io.netcdf.write import _get_dimension_sizes
    >>> _get_dimension_sizes((4,))
    {'ni': 4}
    >>> sizes = _get_dimension_sizes((4, 5))
    >>> sizes["ni"], sizes["nj"]
    (5, 4)
    >>> sizes = _get_dimension_sizes((4, 5, 6))
    >>> sizes["ni"], sizes["nj"], sizes["nk"]
    (6, 5, 4)
    """
    names = _AXIS_DIMENSION_NAMES[-1 : -(len(shape) + 1) : -1]

    sizes = {}
    for axis, name in enumerate(names):
        sizes[name] = shape[-(axis + 1)]

    return sizes


def _get_axes_names(shape):
    """Get names of the axes.

    Parameters
    ----------
    shape : tuple of int
        Shape of a structured grid.

    Returns
    -------
    tuple of str
        Names of the axes for the NetCDF file.

    Examples
    --------
    >>> from landlab.io.netcdf.write import _get_axes_names
    >>> _get_axes_names((2,))
    ['x']
    >>> _get_axes_names((2, 3))
    ['y', 'x']
    >>> _get_axes_names((2, 3, 4))
    ['z', 'y', 'x']
    """
    names = _AXIS_COORDINATE_NAMES[-1 : -(len(shape) + 1) : -1]
    return names[::-1]


def _get_cell_bounds(shape, spacing=(1.0, 1.0), origin=(0.0, 0.0)):
    """Get bounds arrays for square cells.

    Parameters
    ----------
    shape : tuple of int
        Shape of the grid in cell corners.
    spacing : tuple of float
        Height and width of cells.
    origin : tuple of float
        Coordinates of lower-left corner of lower-left cell.

    Returns
    -------
    (y, x) : tuple of ndarray
        Tuple of the *y* and *x* coordinates of each cell corner (ordered
        counter-clockwise starting from lower-right. The shape of the returned
        arrays will be *(rows, cols, 4)*.

    Examples
    --------
    >>> from landlab.io.netcdf.write import _get_cell_bounds
    >>> bounds = _get_cell_bounds((3, 4))
    >>> bounds["y_bnds"]
    array([[[0.,  1.,  1.,  0.], [0.,  1.,  1.,  0.], [0.,  1.,  1.,  0.]],
           [[1.,  2.,  2.,  1.], [1.,  2.,  2.,  1.], [1.,  2.,  2.,  1.]]])
    >>> bounds["x_bnds"]
    array([[[1.,  1.,  0.,  0.], [2.,  2.,  1.,  1.], [3.,  3.,  2.,  2.]],
           [[1.,  1.,  0.,  0.], [2.,  2.,  1.,  1.], [3.,  3.,  2.,  2.]]])
    """
    rows = np.arange(shape[0]) * spacing[0] + origin[0]
    cols = np.arange(shape[1]) * spacing[1] + origin[1]

    corner_y, corner_x = np.meshgrid(rows, cols, indexing="ij")

    y_bnds = np.vstack(
        (
            corner_y[:-1, 1:].flat,
            corner_y[1:, 1:].flat,
            corner_y[1:, :-1].flat,
            corner_y[:-1, :-1].flat,
        )
    ).T
    x_bnds = np.vstack(
        (
            corner_x[:-1, 1:].flat,
            corner_x[1:, 1:].flat,
            corner_x[1:, :-1].flat,
            corner_x[:-1, :-1].flat,
        )
    ).T

    return {
        "y_bnds": y_bnds.reshape((shape[0] - 1, shape[1] - 1, 4)),
        "x_bnds": x_bnds.reshape((shape[0] - 1, shape[1] - 1, 4)),
    }


def _set_netcdf_cell_structured_dimensions(root, shape):
    """Set dimensions for a structured grid of cells.

    Parameters
    ----------
    root : netcdf_file
        A NetCDF file.
    shape : tuple of int
        Shape of the cell grid (rows of cells, columns of cells).
    """
    if len(shape) < 1 or len(shape) > 3:
        raise ValueError("grid dimension must be 1, 2, or 3")

    dimensions = _get_dimension_sizes(shape)

    dims = root.dimensions

    if "nt" not in dims:
        root.createDimension("nt", None)

    for name, dim_size in dimensions.items():
        if name not in dims:
            root.createDimension(name, dim_size - 2)

    root.createDimension("nv", 4)


def _set_netcdf_structured_dimensions(root, shape):
    """Set dimensions for a structured grid.

    Add dimensions to *root* for a structured grid of size *shape*. The
    dimension names will be 'ni', 'nj', and 'nk'. 'ni' is the length of the
    fast dimension, followed by 'nj', and then 'nk'.

    For example, a grid with shape (3, 4, 5) will have dimensions ni=5,
    nj=4, and nk=3. Lower dimension grids simply drop the slowest dimension.
    Thus, a grid with shape (3, 4) has dimensions ni=4, and nj=3.

    Parameters
    ----------
    root : netcdf_file
        A NetCDF file.
    shape : tuple of int
        Shape of the grid.
    """
    if len(shape) < 1 or len(shape) > 3:
        raise ValueError("grid dimension must be 1, 2, or 3")

    dimensions = _get_dimension_sizes(shape)

    dims = root.dimensions

    if "nt" not in dims:
        root.createDimension("nt", None)

    for name, dim_size in dimensions.items():
        if name not in dims:
            root.createDimension(name, dim_size)


def _set_netcdf_variables(root, fields, **kwds):
    """Set the field variables.

    First set the variables that define the grid and then the variables
    at the grid nodes and cells.
    """
    names = kwds.pop("names", None)

    _add_spatial_variables(root, fields, **kwds)
    _add_variables_at_points(root, fields, names=names)


def _set_netcdf_raster_variables(root, fields, **kwds):
    """Set the field variables for rasters.

    First set the variables that define the grid and then the variables
    at the grid nodes and cells.
    """
    names = kwds.pop("names", None)

    _add_raster_spatial_variables(root, fields, **kwds)
    _add_variables_at_points(root, fields, names=names)


def _set_netcdf_cell_variables(root, fields, **kwds):
    """Set the cell field variables.

    First set the variables that define the grid and then the variables
    at the grid nodes and cells.
    """
    names = kwds.pop("names", None)

    _add_cell_spatial_variables(root, fields, **kwds)
    _add_variables_at_cells(root, fields, names=names)


def _add_cell_spatial_variables(root, grid, **kwds):
    """Add the spatial variables that describe the cell grid."""
    long_name = kwds.get("long_name", {})

    cell_grid_shape = [dim - 1 for dim in grid.shape]
    spatial_variable_shape = _get_dimension_names(cell_grid_shape)

    bounds = _get_cell_bounds(
        cell_grid_shape,
        spacing=(grid.dy, grid.dx),
        origin=(grid.dy * 0.5, grid.dx * 0.5),
    )

    shape = spatial_variable_shape + ["nv"]
    for name, values in bounds.items():
        # var = root.createVariable(name, 'f8', shape)
        # var[:] = values

        try:
            var = root.variables[name]
        except KeyError:
            var = root.createVariable(name, "f8", shape)

        var[:] = values

        axis = grid.axis_name.index(name[0])

        var.units = str(grid.axis_units[axis])
        try:
            var.long_name = str(long_name[name])
        except KeyError:
            var.long_name = str(grid.axis_name[axis])


def _add_spatial_variables(root, grid, **kwds):
    """Add spatial variables to a NetCDF file.

    Add the variables to *root* that define the structured grid, *grid*.

    Parameters
    ----------
    root : netcdf_file
        A NetCDF file.
    grid : RasterModelGrid
        A structured grid.
    long_name : dict, optional
        Long name for each spatial variable to add. Keys are grid field
        names, values are corresponding long names.
    """
    long_name = kwds.get("long_name", {})

    netcdf_vars = root.variables

    spatial_variable_names = _get_axes_names(grid.shape)
    spatial_variable_shape = _get_dimension_names(grid.shape)

    for axis, name in enumerate(spatial_variable_names):
        try:
            var = netcdf_vars[name]
        except KeyError:
            var = root.createVariable(name, "f8", spatial_variable_shape)

        coords = grid.node_axis_coordinates(axis=axis).view()
        coords.shape = var.shape
        var[:] = coords

        var.units = grid.axis_units[axis].encode("utf-8")
        try:
            var.long_name = long_name[name].encode("utf-8")
        except KeyError:
            var.long_name = grid.axis_name[axis].encode("utf-8")


def _add_raster_spatial_variables(root, grid, **kwds):
    """Add spatial variables to a NetCDF file for rasters.

    Add the variables to *root* that define the structured grid, *grid*.

    Parameters
    ----------
    root : netcdf_file
        A NetCDF file.
    grid : RasterModelGrid
        A structured grid.
    long_name : dict, optional
        Long name for each spatial variable to add. Keys are grid field
        names, values are corresponding long names.
    """
    long_name = kwds.get("long_name", {})

    netcdf_vars = root.variables

    spatial_variable_names = _get_axes_names(grid.shape)
    spatial_variable_shape = _get_dimension_names(grid.shape)

    for axis, name in enumerate(spatial_variable_names):
        try:
            var = netcdf_vars[name]
        except KeyError:
            var = root.createVariable(name, "f8", [spatial_variable_shape[axis]])

        coords = grid.node_axis_coordinates(axis=axis).view().reshape(grid.shape)
        if axis == 1:
            coords = coords[1, :]
        elif axis == 0:
            coords = coords[:, 1]
        else:
            raise NotImplementedError("")
        coords.shape = var.shape

        var[:] = coords

        var.units = str(grid.axis_units[axis])
        try:
            var.long_name = str(long_name[name])
        except KeyError:
            var.long_name = str(grid.axis_name[axis])


def _add_variables_at_points(root, fields, names=None):
    if isinstance(names, str):
        names = [names]
    names = names or fields["node"].keys()

    netcdf_vars = root.variables

    spatial_variable_shape = _get_dimension_names(fields.shape)

    try:
        n_times = len(netcdf_vars["t"]) - 1
    except TypeError:
        n_times = len(netcdf_vars["t"][:]) - 1
    except KeyError:
        n_times = 0

    node_fields = fields["node"]
    for var_name in names:
        try:
            var = netcdf_vars[var_name]
        except KeyError:
            var = root.createVariable(
                var_name,
                _NP_TO_NC_TYPE[str(node_fields[var_name][0].dtype)],
                ["nt"] + spatial_variable_shape,
            )

        if node_fields[var_name].size > 1:
            data = node_fields[var_name].view()
            data.shape = var.shape[1:]
            try:
                var[n_times, :] = data
            except ValueError:
                raise
        else:
            var[n_times] = node_fields[var_name].flat[0]

        var.units = node_fields.units[var_name] or "?"
        var.long_name = var_name

        if hasattr(fields, "grid_mapping"):
            var.grid_mapping = fields.grid_mapping["name"]


def _add_variables_at_cells(root, fields, names=None):
    if isinstance(names, str):
        names = [names]
    names = names or fields["cell"].keys()

    netcdf_vars = root.variables

    cell_grid_shape = [dim - 1 for dim in fields.shape]

    spatial_variable_shape = _get_dimension_names(cell_grid_shape)

    try:
        n_times = len(netcdf_vars["t"]) - 1
    except KeyError:
        n_times = 0

    cell_fields = fields["cell"]
    for var_name in names:
        try:
            var = netcdf_vars[var_name]
        except KeyError:
            var = root.createVariable(
                var_name,
                _NP_TO_NC_TYPE[str(cell_fields[var_name].dtype)],
                ["nt"] + spatial_variable_shape,
            )

        if cell_fields[var_name].size > 1:
            data = cell_fields[var_name].view()
            data.shape = var.shape[1:]
            try:
                var[n_times, :] = data
            except ValueError:
                raise
        else:
            var[n_times] = cell_fields[var_name].flat[0]

        var.units = str(cell_fields.units[var_name] or "?")
        var.long_name = str(var_name)


def _add_time_variable(root, time, **kwds):
    """Add a time value to a NetCDF file.

    Append a new time value to the time variable of a NetCDF file. If there
    is not time variable, create one. The time variable is named, ``t``.

    Parameters
    ----------
    root : netcdf_file
        A NetCDF file.
    time : float
        The time.
    units : str, optional
        Time units.
    reference : str, optional
        Reference time.
    """
    units = kwds.get("units", "days")
    reference = kwds.get("reference", "00:00:00 UTC")

    netcdf_vars = root.variables

    try:
        time_var = netcdf_vars["t"]
    except KeyError:
        time_var = root.createVariable("t", "f8", ("nt",))
        time_var.units = " ".join([units, "since", reference])
        time_var.long_name = "time"

    try:
        n_times = len(time_var)
    except TypeError:
        n_times = len(time_var[:])
    if time is not None:
        time_var[n_times] = time
    else:
        time_var[n_times] = n_times


def _set_netcdf_grid_mapping_variable(root, grid_mapping):
    """Create grid mapping variable, if necessary."""
    name = grid_mapping.pop("name")
    var = root.createVariable(name, "S1", dimensions=())
    for attr in grid_mapping.keys():
        setattr(var, attr, grid_mapping[attr])


_VALID_NETCDF_FORMATS = {
    "NETCDF3_CLASSIC",
    "NETCDF3_64BIT",
    "NETCDF4_CLASSIC",
    "NETCDF4",
}


def _guess_at_location(fields, names):
    """Guess where the values should be located."""
    node_fields = set(fields["node"].keys())
    cell_fields = set(fields["cell"].keys())

    if names is None or len(names) == 0:
        if len(node_fields) > 0:
            at = "node"
        else:
            at = "cell"
    else:
        if node_fields.issuperset(names):
            at = "node"
        elif cell_fields.issuperset(names):
            at = "cell"
        else:
            at = None
    return at


def write_netcdf(
    path,
    grid,
    attrs=None,
    append=False,
    format="NETCDF3_64BIT",
    names=None,
    at=None,
    time=None,
    raster=False,
):
    """Write landlab fields to netcdf.

    Write the data and grid information for *grid* to *path* as NetCDF.
    If the *append* keyword argument in True, append the data to an existing
    file, if it exists. Otherwise, clobber an existing files.

    Parameters
    ----------
    path : str
        Path to output file.
    grid : RasterModelGrid
        Landlab RasterModelGrid object that holds a grid and associated values.
    append : boolean, optional
        Append data to an existing file, otherwise clobber the file.
    format : {'NETCDF3_CLASSIC', 'NETCDF3_64BIT', 'NETCDF4_CLASSIC', 'NETCDF4'}
        Format of output netcdf file.
    attrs : dict
        Attributes to add to netcdf file.
    names : iterable of str, optional
        Names of the fields to include in the netcdf file. If not provided,
        write all fields.
    at : {'node', 'cell'}, optional
        The location where values are defined.
    raster : bool, optional
        Indicate whether spatial dimensions are written as full value arrays
        (default) or just as coordinate dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> from landlab.io.netcdf import write_netcdf

    Create a uniform rectilinear grid with four rows and 3 columns, and add
    some data fields to it.

    >>> rmg = RasterModelGrid((4, 3))
    >>> rmg.at_node["topographic__elevation"] = np.arange(12.0)
    >>> rmg.at_node["uplift_rate"] = 2.0 * np.arange(12.0)

    Create a temporary directory to write the netcdf file into.

    >>> import tempfile, os
    >>> temp_dir = tempfile.mkdtemp()
    >>> os.chdir(temp_dir)

    Write the grid to a netcdf3 file but only include the *uplift_rate*
    data in the file.

    >>> write_netcdf("test.nc", rmg, format="NETCDF3_64BIT", names="uplift_rate")

    Read the file back in and check its contents.

    >>> from scipy.io import netcdf_file
    >>> fp = netcdf_file("test.nc", "r")
    >>> "uplift_rate" in fp.variables
    True
    >>> "topographic__elevation" in fp.variables
    False
    >>> fp.variables["uplift_rate"][:].flatten().astype("=f8")
    array([  0.,   2.,   4.,   6.,   8.,  10.,  12.,  14.,  16.,  18.,  20.,
            22.])

    >>> rmg.at_cell["air__temperature"] = np.arange(2.0)
    >>> write_netcdf(
    ...     "test-cell.nc",
    ...     rmg,
    ...     format="NETCDF3_64BIT",
    ...     names="air__temperature",
    ...     at="cell",
    ... )
    """
    path = pathlib.Path(path)
    if append and not path.exists():
        append = False

    if at not in (None, "cell", "node"):
        raise ValueError("value location not understood")

    if isinstance(names, str):
        names = (names,)

    at = at or _guess_at_location(grid, names) or "node"
    if names is None:
        names = grid[at].keys()

    if not set(grid[at].keys()).issuperset(names):
        raise ValueError("values must be on either cells or nodes, not both")

    attrs = attrs or {}

    dims = ("nt", "nj", "ni")
    shape = grid.shape
    if at == "cell":
        shape = shape[0] - 2, shape[1] - 2

    data = {}
    if append:
        with xr.open_dataset(path) as dataset:
            time_varying_names = [
                name for name in dataset.variables if "nt" in dataset[name].sizes
            ]
            for name in set(time_varying_names) & set(names):
                values = getattr(grid, "at_" + at)[name].reshape((1,) + shape)
                data[name] = (dims, np.concatenate([dataset[name].values, values]))

            if "t" not in dataset.variables:
                times = np.arange(len(dataset["nt"]) + 1)
            else:
                times = np.concatenate((dataset["t"].values, [0.0]))

        if time is None:
            times[-1] = times[-2] + 1.0
        else:
            times[-1] = time
        data["t"] = (("nt",), times)

    if at == "cell":
        data["x_bnds"] = (
            ("nj", "ni", "nv"),
            grid.x_of_corner[grid.corners_at_cell].reshape(shape + (4,)),
        )
        data["y_bnds"] = (
            ("nj", "ni", "nv"),
            grid.y_of_corner[grid.corners_at_cell].reshape(shape + (4,)),
        )
    else:
        if raster:
            data["x"] = (("ni"), grid.x_of_node.reshape(shape)[0, :])
            data["y"] = (("nj"), grid.y_of_node.reshape(shape)[:, 0])
        else:
            data["x"] = (("nj", "ni"), grid.x_of_node.reshape(shape))
            data["y"] = (("nj", "ni"), grid.y_of_node.reshape(shape))

    if not append:
        if time is not None:
            data["t"] = (("nt",), [time])
        for name in names:
            data[name] = (
                dims,
                getattr(grid, "at_" + at)[name].reshape((-1,) + shape),
            )

    dataset = xr.Dataset(data, attrs=attrs)

    dataset.to_netcdf(path, mode="w", format=format, unlimited_dims=("nt",))


def write_raster_netcdf(
    path,
    fields,
    attrs=None,
    append=False,
    time=None,
    format="NETCDF4",
    names=None,
    at=None,
):
    """Write Raster Model Grid landlab fields to netcdf.

    Write the data and grid information for *fields* to *path* as NetCDF.

    This method is for Raster Grids only and takes advantage of regular x and
    y spacing to save memory.

    Rather that writing x and y of node locations at all (nr x nc) locations,
    it writes a 1D array each for x and y.

    A more modern version of this might write x and y location as a netcdf
    coordinate. However, the original version of this function wrote x and y
    as data variables rather than coordinates.

    If the *append* keyword argument in True, append the data to an existing
    file, if it exists. Otherwise, clobber an existing files.

    Parameters
    ----------
    path : str
        Path to output file.
    fields : field-like
        Landlab field object that holds a grid and associated values. This must
        be a Raster type.
    append : boolean, optional
        Append data to an existing file, otherwise clobber the file.
    time : float, optional
        Add a time to the time variable.
    format : {'NETCDF4'}
        Format of output netcdf file.
    attrs : dict
        Attributes to add to netcdf file.
    names : iterable of str, optional
        Names of the fields to include in the netcdf file. If not provided,
        write all fields.
    at : {'node'}, optional
        The location where values are defined. Presently only implemented for
        type 'node'.

    Examples
    --------
    >>> import numpy as np
    >>> from landlab import RasterModelGrid
    >>> from landlab.io.netcdf import write_raster_netcdf

    Create a uniform rectilinear grid with four rows and 3 columns, and add
    some data fields to it.

    >>> rmg = RasterModelGrid((4, 3))
    >>> rmg.shape
    (4, 3)
    >>> rmg.at_node["topographic__elevation"] = np.arange(12.0)
    >>> rmg.at_node["uplift_rate"] = 2.0 * np.arange(12.0)

    Create a temporary directory to write the netcdf file into.

    >>> import tempfile, os
    >>> temp_dir = tempfile.mkdtemp()
    >>> os.chdir(temp_dir)

    Write the grid to a netcdf4 file but only include the *uplift_rate*
    data in the file.

    >>> write_raster_netcdf(
    ...     "test.nc",
    ...     rmg,
    ...     format="NETCDF3_64BIT",
    ...     names="uplift_rate",
    ... )

    Read the file back in and check its contents.

    >>> from scipy.io import netcdf_file
    >>> fp = netcdf_file("test.nc", "r")
    >>> "uplift_rate" in fp.variables
    True
    >>> "topographic__elevation" in fp.variables
    False
    >>> fp.variables["uplift_rate"][:].flatten().astype("=f8")
    array([  0.,   2.,   4.,   6.,   8.,  10.,  12.,  14.,  16.,  18.,  20.,
            22.])
    >>> fp.variables["x"][:].astype("=f8")
    array([0.,  1.,  2.])
    >>> fp.variables["y"][:].astype("=f8")
    array([0.,  1.,  2.,  3.])

    Read now with read_netcdf

    >>> from landlab.io.netcdf import read_netcdf
    >>> grid = read_netcdf("test.nc")
    >>> grid.shape
    (4, 3)
    >>> grid.x_of_node
    array([0.,  1.,  2.,  0.,  1.,  2.,  0.,  1.,  2.,  0.,  1.,  2.])
    >>> grid.y_of_node
    array([0.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  3.])
    >>> grid.at_node["uplift_rate"]
    array([ 0.,   2.,   4.,   6.,   8.,  10.,  12.,  14.,  16.,  18.,  20.,
           22.])
    """
    return write_netcdf(
        path,
        fields,
        attrs=attrs,
        append=append,
        format=format,
        names=names,
        at=at,
        time=time,
        raster=True,
    )
