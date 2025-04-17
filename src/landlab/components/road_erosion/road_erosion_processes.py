"""Landlab component for road erosion processes including 
pumping, crushing, scattering (and by default, flow rerouting)

Last updated: April 17, 2025

.. codeauthor: Amanda Alvis
"""

from landlab import Component
from landlab.components import LinearDiffuser
import random as rnd
import numpy as np

_DAY_SEC = 24.*3600.

class TruckPassErosion(Component):
    r"""Calculate sediment depths for forest road cross section layers based
    on traffic-induced, erosion-enhancing processes: pumping, crushing,
    scattering (and by default, flow rerouting).

    References
    ----------
    Alvis, A. D., Luce, C. H., & Istanbulluoglu, E. (2023). How does traffic 
    affect erosion of unpaved forest roads? Environmental Reviews, 31(1), 
    182–194. https://doi.org/10.1139/er-2022-0032
    """

    _name = "TruckPassErosion"

    _unit_agnostic = False
    
    _info = {
        "active__depth": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of active layer of sediment of the road cross\
                section",
        },
        "ballast__depth": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of ballast layer of the road cross section",
        },
        "sediment__added": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of fine sediment added to active layer",
        },
        "surfacing__depth": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of surfacing layer of the road cross section",
        },
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
    }
    
    def __init__(
        self, 
        grid, 
        truck_num = 5,
        d95 = 0.0275,
        u_ps = 5e-7,
        u_pb = 1e-7,
        k_cs = 1e-7,
        k_cb = 1e-7,
        f_af = 0.25,
        f_ac = 0.75,
        f_sf = 0.275,
        f_sc = 0.725,
        f_bf = 0.20,
        f_bc = 0.80,
        e = 0.725,
    ):
        """Initialize TruckPassErosion.

        Parameters
        ----------
        grid : ModelGrid
            Landlab ModelGrid object
        truck_num : int
            Average number of trucks to pass over a road segment in a day
        d95 : float
            d95 of road surfacing material
        u_ps : float
            Pumping rate from surfacing to active layer per truck pass
        u_pb : float
            Pumping rate from ballast to surfacing per truck pass
        k_cs : float
            Crushing rate per truck pass in the surfacing 
        k_cb : float
            Crushing rate per truck pass in the ballast
        f_af : float
            Fraction of fine material in the active layer
        f_ac : float
            Fraction of coarse material in the active layer
        f_sf : float
            Fraction of fine material in the surfacing
        f_sc : float
            Fraction of coarse material in the surfacing
        f_bf : float
            Fraction of fine material in the ballast
        f_bc : float
            Fraction of coarse material in the ballast
        e : float

        """

        super().__init__(grid)

        # Store grid and parameters
        self._grid = grid
        self._d95 = d95
        self._u_ps = u_ps
        self._u_pb = u_pb
        self._k_cs = k_cs
        self._k_cb = k_cb
        self._e = e
        
        # Get elevation field
        self._elev = grid.at_node['topographic__elevation']

        # Get layers for sediment depths
        self._active = grid.at_node['active__depth']
        self._surfacing = grid.at_node['surfacing__depth'] #representative node for 3 nodes across?
        self._ballast = grid.at_node['ballast__depth']

        self._active_fine = self._active*f_af
        self._active_coarse = self._active*f_ac
        self._surf_fine = self._surfacing*f_sf
        self._surf_coarse = self._surfacing*f_sc
        self._ball_fine = self._ballast*f_bf
        self._ball_coarse = self._ballast*f_bc
        
        # Get average number of trucks per day
        self.truck_num_avg = truck_num

        self.initialize_output_fields()
        self._sed_added = grid.at_node["sediment__added"]
	
    @property
    def sed_added(self):
        """The depth of fine sediment added to the active layer at
        each node"""
        return self._sed_added

    def calc_tire_tracks(self, centerline, half_width):
        self._center = self._grid.nodes[:, centerline]

        self._center_tracks = np.append(self._center-half_width,\
            self._center+half_width)
        self._out_center = [self._center_tracks-1, self._center_tracks+1]
        self._back_center = self._center_tracks+\
            self._grid.number_of_node_columns
        
        self._right_tracks = self._center_tracks+1
        self._out_right = [self._right_tracks-1, self._right_tracks+1]
        self._back_right = self._right_tracks+\
            self._grid.number_of_node_columns

        self._left_tracks = self._center_tracks-1
        self._out_left = [self._left_tracks-1, self._left_tracks+1]
        self._back_left = self._left_tracks+\
            self._grid.number_of_node_columns
        
        val = rnd.choice([self._center_tracks, self._right_tracks,\
            self._left_tracks])

        if all(val == self._center_tracks):
            self.tire_tracks = [self._center_tracks, self._out_center[0],\
                self._out_center[1], self._back_center]
        elif all(val == self._right_tracks):
            self.tire_tracks = [self._right_tracks, self._out_right[0],\
                self._out_right[1], self._back_right]    
        else:
            self.tire_tracks = [self._left_tracks, self._out_left[0],\
                self._out_left[1], self._back_left]

        return(self.tire_tracks)

    def run_one_step(self, centerline, half_width): 
        active_init = self._active
        surf_init = self._surfacing
        ball_init = self._ballast
        self.truck_num = np.random.poisson(self.truck_num_avg,1).item()

        for _ in range(self.truck_num):
            tire_tracks = self.calc_tire_tracks(centerline, half_width)

            #scattering 
            for i in range(len(tire_tracks[0]) - 1):
                if (self._active_coarse[tire_tracks[0][i]]) <= 0.01:     
                    self._active_coarse[tire_tracks[1][i]] += self._active_coarse[tire_tracks[0][i]]/2.5
                    self._active_coarse[tire_tracks[2][i]] += self._active_coarse[tire_tracks[0][i]]/2.5
                    if tire_tracks[3][i] <= len(self._active_coarse):
                        self._active_coarse[tire_tracks[3][i]] += self._active_coarse[tire_tracks[0][i]]/5
                    self._active_coarse[tire_tracks[0][i]] -= self._active_coarse[tire_tracks[0][i]]
                else:
                    self._active_coarse[tire_tracks[0][i]] -= 0.01
                    self._active_coarse[tire_tracks[1][i]] += 0.004
                    self._active_coarse[tire_tracks[2][i]] += 0.004
                    if tire_tracks[3][i] <= len(self._active_coarse):
                        self._active_coarse[tire_tracks[3][i]] += 0.001

            #calculate pumping fluxes
            q_ps = self._u_ps*(self._surf_fine/self._surfacing)/_DAY_SEC
            q_pb = self._u_pb*(self._ball_fine/self._ballast)/_DAY_SEC

            #calculate crushing fluxes
            q_cs = self._k_cs*(self._surf_coarse/self._surfacing)/_DAY_SEC
            q_cb = self._k_cb*(self._ball_coarse/self._ballast)/_DAY_SEC

            #update surfacing
            self._surf_coarse[tire_tracks[0]] -= q_cs[tire_tracks[0]]*_DAY_SEC
            self._surf_fine[tire_tracks[0]] += q_cs[tire_tracks[0]]*_DAY_SEC - \
                q_ps[tire_tracks[0]]*_DAY_SEC + q_pb[tire_tracks[0]]*_DAY_SEC

            #update ballast
            self._ball_coarse[tire_tracks[0]] -= q_cb[tire_tracks[0]]*_DAY_SEC
            self._ball_fine[tire_tracks[0]] += q_cb[tire_tracks[0]]*_DAY_SEC - \
                q_pb[tire_tracks[0]]*_DAY_SEC

            #update fines in active layer
            for i in range(len(tire_tracks[0]) - 1):
                if self._d95 >= (self._active_fine[tire_tracks[0][i]]):
                    self._sed_added[tire_tracks[0][i]] += q_ps[tire_tracks[0][i]]*_DAY_SEC/ (1-self._e) #Need to update e. Not a static value.
                else:
                    self._sed_added[tire_tracks[0][i]] += q_ps[tire_tracks[0][i]]*_DAY_SEC
                
                self._active_fine[tire_tracks[0][i]] += self._sed_added[tire_tracks[0][i]]

        #update outputs
        self._ballast = self._ball_coarse + self._ball_fine
        ball_dz = self._ballast - ball_init
        self._surfacing = self._surf_coarse + self._surf_fine
        surf_dz = self._surfacing - surf_init
        self._active = self._active_coarse + self._active_fine
        active_dz = self._active - active_init
        
        self._elev += ball_dz + surf_dz + active_dz

        #TODO: Increase tire tracks to be 3 nodes wide; 
        #change stochasticity of truck paths