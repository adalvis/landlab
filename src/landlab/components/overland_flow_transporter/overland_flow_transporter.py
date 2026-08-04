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
        self._phi_f = porosity_f
        self._phi_c = porosity_c
        self._n_c = n_c
        self._n_f = n_f
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
        self._node_stack= grid.at_node["flow__upstream_node_order"]
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
        self._n_t = grid.at_node["total__roughness"]
        self._shear_stress = grid.at_node["shear__stress"]
        self._transport_capacity = grid.at_node["transport__capacity"]
        self._sediment_influx = grid.at_node["sediment__mass_influx"]
        self._sediment_outflux = grid.at_node["sediment__mass_outflux"]
        self._dmdt = grid.at_node["sediment__rate_of_change"]

        self._Maf_plot = [self._Maf[40]]
        self._t_plot = [0]
        
    @property
    def shear_stress(self):
        """The shear stress at each node"""
        return self._shear_stress
              
    # def calc_overland_roughness(self):
    #     """Calculate overland flow surface roughness and shear stress partitioning ratio."""
    #     self._unit_discharge = self._discharge / self.grid.dx

    #     for i in range(len(self._unit_discharge)):
    #         if self._unit_discharge[i] > 0:
    #             self._n_f[i] = self._n_f[i]
    #             if self._Saf[i] <= self._Sac[i]:
    #                 self._n_t[i] = self._n_c + (self._Maf[i] / self._Ma[i]) \
    #                     * (self._n_f[i] - self._n_c)
    #             else:
    #                 self._n_t[i] = self._n_f[i]
    #             self._f_s[i] = (self._n_f[i] / self._n_t[i]) ** 1.5
    #         else:
    #             self._n_f[i] = 0
    #             self._n_t[i] = 0
    #             self._f_s[i] = 0

    # def calc_overland_depth(self):
    #     """Compute overland flow depth using a safe formulation."""
    #     self.calc_overland_roughness()

    #     slope_eps = 1e-8  # prevents divide-by-zero in sqrt(slope)
    #     self._slope_safe = np.maximum(self._slope, slope_eps)   

    #     # initialize depth array to zero
    #     self._water_depth[:] = 0.0

    #     # loop over nodes (safe computation)
    #     for i in range(len(self._slope)):
    #         if self._unit_discharge[i] > 0:
    #             safe_denom = np.sqrt(self._slope_safe[i])
    #             raw = (self._n_t[i] * self._unit_discharge[i]) / safe_denom

    #         # guard against negatives and overflow
    #             if raw > 0 and np.isfinite(raw):
    #                 self._water_depth[i] = raw ** (3.0 / 5.0)
    #             else:
    #                 self._water_depth[i] = 0.0
    #         else:
    #             self._water_depth[i] = 0.0

    #     # replace any NaN or inf with zero
    #     self._water_depth[~np.isfinite(self._water_depth)] = 0.0

    #     # cap to a maximum water depth of 0.3 m
    #     self._water_depth[:] = np.minimum(self._water_depth, 0.3)

    # def calc_overland_shear_stress(self):   
    #     """Calculate and return overland flow partitioned shear stress.
    #     """
    #     self.calc_overland_depth()

    #     self._shear_stress = self._rho_w*self._g*self._water_depth*self._slope_safe*self._f_s

    # def calc_overland_transport_capacity(self):
    #     """Calculate and return transport capacity.
    #     """
    #     self.calc_overland_shear_stress()

    #     (greater_than_tc,) = np.where(self._shear_stress >= self._tau_c)
    #     (less_than_tc,) = np.where(self._shear_stress < self._tau_c)

    #     self._transport_capacity[greater_than_tc] = (
    #                 ((10**(-4.348))
    #                 / ((self._d50)**(0.811)))
    #                 * (self._shear_stress[greater_than_tc]-self._tau_c)**(2.457)
    #                 ) * self.grid.dx #[kg/s]
    #     self._transport_capacity[less_than_tc] = 0.0

    #     self._transport_capacity[~np.isfinite(self._transport_capacity)] = 0.0
    #     self._transport_capacity[:] = np.maximum(self._transport_capacity, 0.0)

    # def calc_overland_sediment_rate_of_change(self, t, m, dt):
    #     """Update the rate of mass change of sediment at each core node.
    #     """
    #     self.calc_overland_transport_capacity()
    #     stack_flip_ud = np.flipud(self._node_stack)
        
    #     n_nodes = stack_flip_ud.shape[0]
    #     self._sediment_influx[:] = 0.0
    #     for i in range(n_nodes):
    #         node_id = stack_flip_ud[i]

    #         if self._unit_discharge[node_id] > 0 and (self._receiver_node[node_id] != node_id):
    #             self._sediment_outflux[node_id] = min(
    #                 (self._sediment_influx[node_id]+self._Maf[node_id] / (dt)),
    #                 self._transport_capacity[node_id])
                
    #             self._sediment_influx[self._receiver_node[node_id]] += self._sediment_outflux[node_id]
    #         else:
    #             self._sediment_outflux[node_id] = 0

    #         self._dmdt[node_id] = self._sediment_influx[node_id] - self._sediment_outflux[node_id]
        
    #     return(self._dmdt)
       
        # for i in range(len(self._transport_capacity)):
        #     self._sediment_outflux[i] = min(
        #         self._transport_capacity[i], 
        #         (self._Maf[i] / (dt*86400))
        #         )

        # self._sediment_influx[:] = 0.0
        # for c in cores:  # send sediment downstream
        #     r = self._receiver_node[c]
        #     self._sediment_influx[r] += self._sediment_outflux[c]

        # self._dmdt[cores] = (
        #     self._sediment_influx[cores] - self._sediment_outflux[cores]
        #     )
    
    ## ======================================================
    ## Calculate differential
    ## ======================================================
    def dmdt(self, Maf, dt):
        # Define unit discharge for calculations
        self._unit_discharge = self._discharge / self.grid.dx

        # Smallest allowable slope to prevent div by 0
        slope_min = 1e-8*np.ones(len(self._slope))
        self._slope_safe = np.maximum(self._slope, slope_min)
        
        stack_flip_ud = np.flipud(self._node_stack)
        n_nodes = stack_flip_ud.shape[0]
        
        self._sediment_influx[:] = 0

        for i in range(n_nodes):
            node_id = stack_flip_ud[i]

            # Lump some parameters together to make code neater
            K = 10**(-4.348) / (self._rho_s * self._d50**0.811)
            A = self._rho_w * self._g * self._slope_safe[node_id]**0.7 *\
                self._unit_discharge[node_id]**0.6 * self._n_f**1.5
            denom = self._phi_c*self._Sac[node_id]*(1-self._phi_f)*self._rho_s*self._area
            phi = Maf[node_id] / denom if denom != 0 else 0.0
            nt = self._n_c + phi*(self._n_f - self._n_c) \
                if self._Saf[node_id] < self._Sac[node_id] else self._n_f
            
            # print(nt)
            
            self._shear_stress[node_id] = A * nt**(-0.9)
            excess = self._shear_stress[node_id] - self._tau_c
            E = K * excess**2.457 if excess > 0.0 else 0.0
            avail = Maf[node_id] / dt + self._sediment_influx[node_id]

            if (self._unit_discharge[node_id] > 0) and (self._receiver_node[node_id] != node_id):
                self._sediment_outflux[node_id] = min(E, avail)
                self._sediment_influx[self._receiver_node[node_id]] += self._sediment_outflux[node_id]
            else:
                self._sediment_outflux[node_id] = 0
        return (self._sediment_influx-self._sediment_outflux)

    ## ======================================================
    ## RK4 single step (fixed dt)
    ## ======================================================
    def rk4_step(self, Maf, dt):
        k1 = self.dmdt(Maf, dt)
        k2 = self.dmdt(Maf + 0.5*dt*k1, dt)
        k3 = self.dmdt(Maf + 0.5*dt*k2, dt)
        k4 = self.dmdt(Maf + dt*k3, dt)
        return Maf + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)

    ## ======================================================
    ## Adaptive step via step-doubling
    ##    - one full step  (dt)      -> M_big
    ##    - two half steps (dt/2)    -> M_small
    ##    - error estimate: |M_small - M_big|
    ## ======================================================
    def adaptive_step(self, Maf, dt):
        dt_min  = 500
        dt_max = 20000 # largest allowed step
        tol = 1e-6 # absolute error tolerance per step
        safety = 0.9  # safety factor for step update
        p = 4 # order of RK4 (used in step-size formula)

        # one full step
        M_big = self.rk4_step(Maf, dt)
        # two half steps
        M_half = self.rk4_step(Maf, 0.5*dt)
        M_small = self.rk4_step(M_half, 0.5*dt)

        # local truncation error estimate (Richardson)
        err = np.max(abs(M_small - M_big))

        # scaled error relative to tolerance
        # (5th-order accurate combination for the accepted value)
        if err == 0.0:
            dt_new = min(dt * 5.0, dt_max)
            return M_small + (M_small - M_big)/15.0, dt, dt_new, True

        
        # proposed new step size
        dt_new = safety * dt * (tol/err)**(1.0/(p+1))
        dt_new = min(max(dt_new, dt_min), dt_max)

        if err <= tol:
            # ACCEPT: Richardson-extrapolated (5th order) value
            M_acc = M_small + (M_small - M_big)/15.0
            return M_acc, dt, dt_new, True
        else:
            # REJECT: retry with smaller step
            return Maf, dt, dt_new, False


    def run_one_step(self, dt):
        """Advance solution by time interval dt.
        """
        self._storm_dt = dt*86400
        self._Sa_init = self._Sa.copy()

        ## ======================================================
        ## ADAPTIVE INTEGRATION LOOP
        ## ======================================================
        dt_init = 3000 #sec
        dt_min  = 500
        
        t = 0
        self._step_dt = dt_init

        while t < self._storm_dt:
            self._step_dt = min(self._step_dt, self._storm_dt - t)

            t += self._step_dt

            self._Maf[:] = self.rk4_step(self._Maf, self._step_dt)

            self._Maf_plot.append(self._Maf[40])
            self._t_plot.append(t)

            # M_new, dt_used, dt_new, accepted = self.adaptive_step(self._Maf, self._step_dt)

            # if accepted:
            #     t += dt_used
            #     for i in range(len(self._Maf)):
            #         self._Maf[i] = max(M_new[i], 0.0)     ## physical: storage >= 0
            #     self._step_dt = dt_new              ## grow/shrink for next step
            # else:
            #     self._step_dt = dt_new              ## shrink and retry (no advance)

            # if self._step_dt < dt_min:
            #     raise RuntimeError("Step size underflow: problem may be too stiff.")

        Maf_crit = self._phi_c*self._d95*(1-self._phi_f)*self._rho_s*self._area
                
        for i in range(len(self._Maf)):
            if self._Maf[i] <= Maf_crit:
                self._Saf[i] = self._Maf[i]/(self._phi_f*(1-self._phi_f)*self._rho_s*self._area)
            elif self._Maf[i] > Maf_crit:
                self._Saf[i] = (self._Maf[i]/((1-self._phi_f)*self._rho_s*self._area)\
                    + self._d95*((1-self._phi_c)/(1-self._phi_f)))*(1/(self._phi_c + 1))

        self._Sa[:] = np.maximum(self._Sac, self._Saf)

        self._topographic_elev += (
            self._Sa - self._Sa_init
        )