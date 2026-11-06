"""Landlab component for calculating shallow overland flow
sediment transport using Govers' equation (1992)

Last updated:  July 01, 2026

.. codeauthor: Amanda Alvis
"""

from landlab import Component
from landlab.components import FlowAccumulator
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
        "flow__upstream_node_order": {
            "dtype": int,
            "intent": "in",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "Node array containing downstream-to-upstream ordered list of node IDs",
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
        "shear__stress": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Shear stress at node",
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
        n_c=0.05,
        n_f=0.015,
        rho_w=1000,
        rho_s=2650,
        g=9.81,
        d50=2e-4, # this is so small, originally d50 = 1.8e-5
        tau_c=0.178,
        porosity_f = 0.35,
        porosity_c = 0.35,
        longitudinal_slope = 0.125,
        d95=0.019,
        row=15,
        flow_accumulator=None,
        depression_finder=None,
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
        self._n_c = n_c
        self._n_f = n_f
        self._rho_w = rho_w
        self._rho_s = rho_s
        self._g = g
        self._d50 = d50
        self._tau_c = tau_c
        self._longitudinal_slope = longitudinal_slope
        self._d95 = d95
        self._row = row
        self._fa = flow_accumulator
        self._df = depression_finder
        
        # Fields and arrays
        self._topographic_elev = grid.at_node["topographic__elevation"]
        self._discharge = grid.at_node["surface_water__discharge"]
        self._slope = grid.at_node["topographic__steepest_slope"]
        self._node_stack= grid.at_node["flow__upstream_node_order"]
        self._receiver_node = grid.at_node["flow__receiver_node"]
        self._Sa = grid.at_node["active__depth"]
        self._Saf = grid.at_node["active__depth_fines"]
        self._Sac = grid.at_node["active__depth_coarse"]
        self._Ma = grid.at_node["active__mass"]
        self._Maf = grid.at_node["active__mass_fines"]
        self._Mac = grid.at_node["active__mass_coarse"]
        self._road_flag = grid.at_node["flag"]

        if "phi_c_a" in grid.at_node:
            self._phi_c_a = grid.at_node["phi_c_a"]
        else:
            self._phi_c_a = grid.add_zeros(
                "phi_c_a", at="node", dtype=float
            )
            self._phi_c_a *= porosity_c
        
        if "phi_f_a" in grid.at_node:
            self._phi_f_a = grid.at_node["phi_f_a"]
        else:
            self._phi_f_a = grid.add_zeros(
                "phi_f_a", at="node", dtype=float
            )
            self._phi_f_a *= porosity_f
        
        super().initialize_output_fields()
        self._f_s = grid.at_node["shear_stress__partitioning"]
        self._n_t = grid.at_node["total__roughness"]
        self._shear_stress = grid.at_node["shear__stress"]
        self._transport_capacity = grid.at_node["transport__capacity"]
        self._sediment_influx = grid.at_node["sediment__mass_influx"]
        self._sediment_outflux = grid.at_node["sediment__mass_outflux"]
        self._dmdt = grid.at_node["sediment__rate_of_change"]

        self._Maf_plot = [self._Maf[self.grid.nodes[self._row, :]]/self._Mac[self.grid.nodes[self._row, :]]]#self._Maf[self.grid.nodes[self._row,:]]]
        self._t_plot = [0]
        
    @property
    def shear_stress(self):
        """The shear stress at each node"""
        return self._shear_stress

    @property
    def mannings_n(self):
        """Manning's roughness at each node"""
        return self._n_t

#%%
    ## ======================================================
    ## Calculate differential
    ## ======================================================
    def dmdt(self, Maf, dt):
        z_init = self._topographic_elev.copy()

        # Define unit discharge for calculations
        self._unit_discharge = self._discharge / self.grid.dx

        # Smallest allowable slope to prevent div by 0
        slope_min = 1e-8*np.ones(len(self._slope))
        self._slope_safe = np.maximum(self._slope, slope_min)
        
        stack_flip_ud = np.flipud(self._node_stack)
        n_nodes = stack_flip_ud.shape[0]
        
        self._sediment_influx[:] = 0
        sediment_influx = self._sediment_influx.copy()
        sediment_outflux = self._sediment_outflux.copy()

        Mc = self._phi_c_a*self._Sac*(1-self._phi_f_a)*self._rho_s*self._area
        zed = Maf / Mc

        (greater_than_Mc,) = np.where(Maf > Mc)
        (less_than_Mc,) = np.where(Maf <= Mc)

        self._n_t[greater_than_Mc] = self._n_f
        self._n_t[less_than_Mc] = self._n_c + zed[less_than_Mc]*(self._n_f - self._n_c) 

        y_d8 = ((self._n_t * self._unit_discharge) / np.sqrt(self._slope_safe)) ** (3/5)

        z_eff = z_init + y_d8
        self._topographic_elev[:] = z_eff

        self._fa.run_one_step()
        self._df.map_depressions()
        Q_flood = self.grid.at_node['surface_water__discharge'].copy()
        q_flood = Q_flood/self.grid.dx

        S_flood = np.maximum(self.grid.at_node['topographic__steepest_slope'], slope_min)
        y_flood = ((self._n_t * q_flood) / np.sqrt(S_flood)) ** (3/5)

        ## Step 3: Restore z
        self._topographic_elev[:] = z_init

        K = 10**(-4.348) / (self._rho_s * self._d50**0.811)
        A = self._rho_w * self._g * S_flood**0.7 *\
            q_flood**0.6 * self._n_f**1.5

        self._shear_stress = A * self._n_t**(-0.9)#(0.6)
        self._f_s[:] = (self._n_f / self._n_t) ** 1.5
        excess = self._shear_stress - self._tau_c

        (excess_more_0,) = np.where(excess > 0.0)
        (excess_less_0,) = np.where(excess <= 0.0)

        E = np.zeros(len(excess))
        E[excess_more_0] = K * excess[excess_more_0]**2.457
        E[excess_less_0] = 0.0
        avail = Maf / dt + self._sediment_influx

        for i in range(n_nodes):
            node_id = stack_flip_ud[i]

            if (self._unit_discharge[node_id] > 0) and (self._receiver_node[node_id] != node_id):
                sediment_outflux[node_id] = min(E[node_id], avail[node_id])
                sediment_influx[self._receiver_node[node_id]] += sediment_outflux[node_id]
            else:
                sediment_outflux[node_id] = 0
        return (sediment_influx-sediment_outflux, sediment_influx)

    ## ======================================================
    ## RK4 single step (fixed dt)
    ## ======================================================
    def rk4_step(self, Maf, dt):
        k1,s1 = self.dmdt(Maf, dt)
        k2,s2 = self.dmdt(Maf + 0.5*dt*k1, dt)
        k3,s3 = self.dmdt(Maf + 0.5*dt*k2, dt)
        k4,s4 = self.dmdt(Maf + dt*k3, dt)
        
        self._sediment_influx[:] = (1/6.0)*(s1 + 2.0*s2 + 2.0*s3 + s4)
        return (Maf + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4))

    def run_one_step(self, dt):
        """Advance solution by time interval dt.
        """
        self._storm_dt = dt*86400
        self._Sa_init = self._Sa.copy()

        ## ======================================================
        ## ADAPTIVE INTEGRATION LOOP
        ## ======================================================
        dt_init = 5000 #sec
        dt_min  = 500
        
        t = 0
        self._step_dt = dt_init

        while t < self._storm_dt:
            self._step_dt = min(self._step_dt, self._storm_dt - t)
            t += self._step_dt

            self._Maf[:] = self.rk4_step(self._Maf, self._step_dt)

            self._Maf[self.grid.nodes[0,:]] -= self._sediment_influx[self.grid.nodes[0,:]]*self._step_dt

        Maf_crit = self._phi_c_a*self._d95*(1-self._phi_f_a)*self._rho_s*self._area
                
        (greater_than_Maf_crit,) = np.where(self._Maf > Maf_crit)
        (less_than_Maf_crit,) = np.where(self._Maf <= Maf_crit)

        self._Saf[less_than_Maf_crit] = self._Maf[less_than_Maf_crit]/\
            (self._phi_c_a[less_than_Maf_crit]*(1-self._phi_f_a[less_than_Maf_crit])*self._rho_s*self._area)
        self._Saf[greater_than_Maf_crit] = (self._Maf[greater_than_Maf_crit]/\
            ((1-self._phi_f_a[greater_than_Maf_crit])*self._rho_s*self._area)\
            + self._d95*((1-self._phi_c_a[greater_than_Maf_crit])/\
            (1-self._phi_f_a[greater_than_Maf_crit])))*(1/(self._phi_c_a[greater_than_Maf_crit] + 1))

        self._Sa[:] = np.maximum(self._Sac, self._Saf)
        
        self._Maf_plot.append(self._Maf[self.grid.nodes[self._row, :]]/self._Mac[self.grid.nodes[self._row, :]])
        self._t_plot.append(t)

        self._topographic_elev += (
            self._Sa - self._Sa_init
        )