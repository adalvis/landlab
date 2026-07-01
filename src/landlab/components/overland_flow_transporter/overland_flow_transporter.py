"""Landlab component for calculating shallow overland flow
sediment transport using Govers' equation (1992)

Last updated:  July 01, 2026

.. codeauthor: Amanda Alvis
"""

from landlab import Component
import numpy as np

class OverlandFlowTransporter(Component):
    """Erodes a surface with shallow overland flow using physics-based formulations 
    for entrainment and transport that can directly use sediment size distribution 
    and surface roughness. The sediment transport rate is calculated using Govers' 
    equation (1992) with shear stress partitioning.

    References:
    ----------
    Govers, G. (1992). Evaluation of transporting capacity formulae for overland 
    flow. In Overland Flow: Hydraulics and Erosion Mechanics (pp. 243–273). New 
    York: Chapman and Hall.
    """

    _name = "OverlandFlowTransporter"

    _unit_agnostic = True

    _info = {
        "sediment__mass_influx": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m**3/time",
            "mapping": "node",
            "doc": "Volumetric incoming streamwise sediment transport rate",
        },
        "sediment__mass_outflux": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m**3/time",
            "mapping": "node",
            "doc": "Volumetric outgoing streamwise sediment transport rate",
        },
        "flow__receiver_node": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array of receivers (node that receives flow from current node)",
        },
        "sediment__rate_of_change": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m/time",
            "mapping": "node",
            "doc": "Time rate of change of sediment thickness",
        },
        "surface_water__discharge": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m**3/time",
            "mapping": "node",
            "doc": "Volumetric discharge of surface water",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "topographic__steepest_slope": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "The steepest *downhill* slope",
        },
        "active__depth": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Depth of active layer of sediment of the road cross\
                section",
        },
        "active__depth_fines": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Depth of fine sediment in the active layer",
        },
        "active__depth_coarse": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Depth of coarse sediment in the active layer",
        },
        "active__mass": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "Mass of active layer of sediment of the road cross\
                section",
        },
        "active__mass_fines": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "Mass of fine sediment in the active layer",
        },
        "active__mass_coarse": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "Mass of coarse sediment in the active layer",
        },
        "grain__roughness": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Manning's roughness for fine sediment grains",
        },
        "total__roughness": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Total Manning's roughness",
        },
        "shear_stress__partitioning": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Shear stress partitioning ratio",
        },
        "water__depth": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Depth of water",
        },
        "transport__capacity": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m**3/time",
            "mapping": "node",
            "doc": "Sediment transport capacity",
        },


    }

    def __init__(
        self,
        grid,
        n_c=0.1,
        n_f=0.015,
        rho_w=1000,
        rho_s=2650,
        g=9.81,
        d50=2e-4, # this is so small, originally d50 = 1.8e-5
        tau_c=0.178,
        porosity = 0.35,
        longitudinal_slope = 0.125,
        d95=0.019,
    ):
        """Initialize OverlandFlowTransporter.
        
        Parameters
        ----------
        grid : ModelGrid
            Landlab ModelGrid object
        n_c : float
            The Manning's roughness of the surface's coarse material
        n_f : float
            The Manning's roughness of the surface's fine material
        rho_w : int
            The density of water [kg/m^3]
        rho_s : int
            The density of sediment [kg/m^3]
        g : float
            Acceleration of gravity [m/s^2]
        d50 : float
            The median grain size (d50) of the surface's material [m]
        tau_c : float
            The critical shear stress required to move sediment [Pa]
        """

        super().__init__(grid)
        self._area = grid.dx*grid.dy
        
        # Parameters
        self._phi_f = porosity
        self._n_c = n_c
        self._n_f_ini = n_f
        self._rho_w = rho_w
        self._rho_s = rho_s
        self._g = g
        self._d50 = d50
        self._tau_c = tau_c
        self._longitudinal_slope = longitudinal_slope
        self._d95 = d95
        
        # Fields and arrays
        self._topographic_elev = grid.at_node["topographic__elevation"]
        self._discharge = grid.at_node["surface_water__discharge"]
        self._slope = grid.at_node["topographic__steepest_slope"]
        self._receiver_node = grid.at_node["flow__receiver_node"]
        self._Sa = grid.at_node["active__depth"]
        self._Saf = grid.at_node["active__depth_fines"]
        self._Sac = grid.at_node["active__depth_coarse"]
        self._Ma = grid.at_node["active__mass"]
        self._Maf = grid.at_node["active__mass_fines"]
        self._Mac = grid.at_node["active__mass_coarse"]
        self._road_flag = grid.at_node["flag"]
        
        super().initialize_output_fields()
        self._f_s = grid.at_node["shear_stress__partitioning"]
        self._n_f = grid.at_node["grain__roughness"]
        self._n_t = grid.at_node["total__roughness"]
        self._water_depth = grid.at_node["water__depth"]
        self._transport_capacity = grid.at_node["transport__capacity"]
        self._sediment_influx = grid.at_node["sediment__mass_influx"]
        self._sediment_outflux = grid.at_node["sediment__mass_outflux"]
        self._dmdt = grid.at_node["sediment__rate_of_change"]

    @property
    def shear_stress(self):
        """The shear stress at each node"""
        return self._shear_stress
              
    def calc_overland_roughness(self):
        """Calculate overland flow surface roughness and shear stress partitioning ratio."""
        self._unit_discharge = self._discharge / self.grid.dx

        for i in range(len(self._unit_discharge)):
            if self._unit_discharge[i] > 0:
                self._n_f[i] = self._n_f_ini
                if self._Saf[i] <= self._Sac[i]:
                    self._n_t[i] = self._n_c + (self._Maf[i] / self._Ma[i]) \
                        * (self._n_f[i] - self._n_c)
                else:
                    self._n_t[i] = self._n_f[i]
                self._f_s[i] = (self._n_f[i] / self._n_t[i]) ** 1.5
            else:
                self._n_f[i] = 0
                self._n_t[i] = 0
                self._f_s[i] = 0

    def calc_overland_depth(self):
        """Compute overland flow depth using a safe formulation."""
        self.calc_overland_roughness()

        slope_eps = 1e-8  # prevents divide-by-zero in sqrt(slope)
        self._slope_safe = np.maximum(self._slope, slope_eps)   

        # initialize depth array to zero
        self._water_depth[:] = 0.0

        # loop over nodes (safe computation)
        for i in range(len(self._slope)):
            if self._unit_discharge[i] > 0:
                safe_denom = np.sqrt(self._slope_safe[i])
                raw = (self._n_t[i] * self._unit_discharge[i]) / safe_denom

            # guard against negatives and overflow
                if raw > 0 and np.isfinite(raw):
                    self._water_depth[i] = raw ** (3.0 / 5.0)
                else:
                    self._water_depth[i] = 0.0
            else:
                self._water_depth[i] = 0.0

        # replace any NaN or inf with zero
        self._water_depth[~np.isfinite(self._water_depth)] = 0.0

        # cap to a maximum water depth of 0.3 m
        self._water_depth[:] = np.minimum(self._water_depth, 0.3)

    def calc_overland_shear_stress(self):   
        """Calculate and return overland flow partitioned shear stress.
        """
        self.calc_overland_depth()

        self._shear_stress = self._rho_w*self._g*self._water_depth*self._slope_safe*self._f_s

    def calc_overland_transport_capacity(self):
        """Calculate and return transport capacity.
        """
        self.calc_overland_shear_stress()

        for i in range(len(self._shear_stress)):
            if self._shear_stress[i] >= self._tau_c:
                self._transport_capacity[i] = (
                    ((10**(-4.348))
                    / ((self._d50)**(0.811)))
                    * (self._shear_stress[i]-self._tau_c)**(2.457)
                ) * self.grid.dx #[kg/s]
            else:
                self._transport_capacity[i] = 0.0

        self._transport_capacity[~np.isfinite(self._transport_capacity)] = 0.0
        self._transport_capacity[:] = np.maximum(self._transport_capacity, 0.0)

    def calc_overland_sediment_rate_of_change(self, dt):
        """Update the rate of mass change of sediment at each core node.
        """
        self.calc_overland_transport_capacity()
        cores = self.grid.core_nodes
       
        for i in range(len(self._transport_capacity)):
            self._sediment_outflux[i] = min(
                self._transport_capacity[i], 
                (self._Maf[i] / (dt*86400))
                )

        self._sediment_influx[:] = 0.0
        for c in cores:  # send sediment downstream
            r = self._receiver_node[c]
            self._sediment_influx[r] += self._sediment_outflux[c]

        self._dmdt[cores] = (
            self._sediment_influx[cores] - self._sediment_outflux[cores]
            )
        
    def run_one_step(self, dt):
        """Advance solution by time interval dt.
        """
        self._dt = dt
        self._Sa_init = self._Sa.copy()

        self.calc_overland_sediment_rate_of_change(self._dt)
        
        self._Maf += self._dmdt*self._dt*86400

        Maf_crit = self._phi_f*self._d95*(1-self._phi_f)*self._rho_s*self._area
                
        for i in range(len(self._Maf)):
            if self._Maf[i] <= Maf_crit:
                self._Saf[i] = self._Maf[i]/(self._phi_f*(1-self._phi_f)*self._rho_s*self._area)
            elif self._Maf[i] > Maf_crit:
                self._Saf[i] = (self._Maf[i]/((1-self._phi_f)*self._rho_s*self._area)\
                    + self._d95)*(1/(self._phi_f + 1))

        self._Sa[:] = np.maximum(self._Sac, self._Saf)

        self._topographic_elev += (
            self._Sa - self._Sa_init
        )