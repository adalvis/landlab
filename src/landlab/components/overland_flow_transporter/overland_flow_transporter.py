from landlab import Component
import numpy as np

class OverlandFlowTransporter(Component):
    """

    """

    _name = "OverlandFlowTransporter"

    _unit_agnostic = True

    _info = {
        "sediment__volume_influx": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m**3/s",
            "mapping": "node",
            "doc": "Volumetric incoming streamwise sediment transport rate",
        },
        "sediment__volume_outflux": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m**3/s",
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
            "units": "m/s",
            "mapping": "node",
            "doc": "Time rate of change of sediment thickness",
        },
        "surface_water__discharge": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m**3/s",
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
            "doc": "depth of active layer of sediment of the road cross\
                section",
        },
        "active__fines": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of active layer of sediment of the road cross\
                section",
        },
        "active__coarse": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of active layer of sediment of the road cross\
                section",
        },
        "sediment__added": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of fine sediment added to active layer",
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
            "doc": "total Manning's roughness",
        },
        "shear_stress__partitioning": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "-",
            "mapping": "node",
            "doc": "shear stress partitioning ratio",
        },
        "water__depth": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "shear stress partitioning ratio",
        },
    }

    def __init__(
        self,
        grid,
        # ditch_treatment,
        d95=0.020,
        n_c=0.4,
        rho_w=1000,
        rho_s=2650,
        g=9.81,
        d50=1.8e-5,
        tau_c=0.052,
    ):
        """Initialize OverlandFlowTransporter."""

        super().__init__(grid)

        # Parameters
        # self._ditch_treatment = ditch_treatment
        self._d95 = d95
        self._n_c = n_c
        self._rho_w = rho_w
        self._rho_s = rho_s
        self._g = g
        self._d50 = d50
        self._tau_c = tau_c

        # Fields and arrays
        self._elev = grid.at_node["topographic__elevation"]
        self._discharge = grid.at_node["surface_water__discharge"]
        self._slope = grid.at_node["topographic__steepest_slope"]
        self._receiver_node = grid.at_node["flow__receiver_node"]
        self._active_depth = grid.at_node["active__depth"]
        # self._active_fines = grid.at_node["active__fines"]
        # self._active_coarse = grid.at_node["active__coarse"]
        self._sed_added = grid.at_node["sediment__added"]
        self._road_flag = grid.at_node["flag"]
        
        super().initialize_output_fields()
        self._f_s = grid.at_node["shear_stress__partitioning"]
        self._n_f = grid.at_node["grain__roughness"]
        self._n_t = grid.at_node["total__roughness"]
        self._water_depth = grid.at_node["water__depth"]
        # self._channel_depth = grid.at_node["channel_depth"]
        self._sediment_influx = grid.at_node["sediment__volume_influx"]
        self._sediment_outflux = grid.at_node["sediment__volume_outflux"]
        self._dzdt = grid.at_node["sediment__rate_of_change"]

    def calc_overland_roughness(self):
        self._unit_discharge = self._discharge/self.grid.dx
        for i in range(len(self._unit_discharge)):
            if self._unit_discharge[i] > 0:
                if self._road_flag[i] == 1:
                    self._n_f[i] = 0.0026*self._unit_discharge[i]**(-0.274)
                    if self._active_depth[i] <= self._d95: #need to fix to be active_coarse and active_fines, but how do I update those??
                        self._n_t[i] = self._n_c + (self._active_depth[i]/self._d95)*(self._n_f[i] - self._n_c)
                        self._f_s[i] = (self._n_f[i]/self._n_t[i])**(1.5)*(self._active_depth[i]/self._d95)
                    else:
                        self._n_t[i] = self._n_f[i]
                        self._f_s[i] = (self._n_f[i]/self._n_t[i])**(1.5)
                # elif self._road_flag[i] == 0:
                #     self._n_f[i] = 0.05
                #     match self._ditch_treatment:
                #         case "armor":
                #             self._n_t[i] = 0.1
                #         case "rock":
                #             self._n_t[i] = 0.25
                #         case "sp_wattles":
                #             self._n_t[i] = 0.4
                #         case "sp_wattles_late":
                #             self._n_t[i] = 0.6
                #         case "grass":
                #             self._n_t[i] = 0.6
                #         case "de_wattles":
                #             self._n_t[i] = 0.725
                #         case _:
                #             self._n_t[i] = 0.05
                #     self._f_s[i] = (self._n_f[i]/self._n_t[i])**(24/13)
            else:
                self._n_f[i] = 0
                self._n_t[i] = 0

    def calc_overland_depth(self):
        self.calc_overland_roughness()
        for i in range(len(self._unit_discharge)):
            if self._unit_discharge[i] > 0:
                if self._road_flag[i] == 1:
                    self._water_depth[i]=((self._n_t[i]*self._unit_discharge[i])/(self._slope[i]**(0.5)))**(3/5)
            else:
                self._water_depth[i]=0

    def calc_overland_shear_stress(self):
        self.calc_overland_depth()
        self._shear_stress = self._rho_w*self._g*self._water_depth*self._slope*self._f_s

    def calc_overland_transport_capacity(self):
        """Calculate and return bed-load transport capacity.
        """
        self.calc_overland_shear_stress()
        for i in range(len(self._shear_stress)):
            if self._shear_stress[i] >= self._tau_c:
                self._sediment_outflux[i] = (
                    ((10**(-4.348))
                    / (self._rho_s*((self._d50)**(0.811))))
                    * (self._shear_stress[i]-self._tau_c)**(2.457)
                )
            else:
                self._sediment_outflux[i] = 0.0

    def calc_overland_sediment_rate_of_change(self):
        """Update the rate of thickness change of sediment at each core node.
        """
        self.calc_overland_transport_capacity()
        cores = self.grid.core_nodes

        # for i in range(len(self._sediment_outflux)):
        #     self._sediment_outflux[i] = min(
        #         self._sediment_outflux[i], ((self._active_fines[i])
        #         * self.grid.area_of_cell[self.grid.cell_at_node[i]])
        #         )
        self._sediment_influx[:] = 0.0
        for c in cores:  # send sediment downstream
            r = self._receiver_node[c]
            self._sediment_influx[r] += self._sediment_outflux[c]
        self._dzdt[cores] = (
            self._sediment_influx[cores] - self._sediment_outflux[cores]
        ) / self.grid.area_of_cell[self.grid.cell_at_node[cores]]

    # def calc_channel_depth(self):
    #     self.calc_roughness()
    #     for i in self.grid.nodes.reshape(np.size(self.grid.nodes))[self._road_flag==0]:
    #         if self._unit_discharge[i]
                        
    #         self._water_depth[i] = ((self._n_t[i]*self._unit_discharge[i])/\
    #             (np.sqrt(6*self._slope[i]/0.718)))**(6/13) #use overall channel S

    # def calc_channel_shear_stress(self):

    def run_one_step(self, dt):
        """Advance solution by time interval dt.
        """
        self.calc_overland_sediment_rate_of_change()
        self._elev += self._dzdt * dt
        # self._active_fines += self._dzdt*dt
        # self._active_depth
