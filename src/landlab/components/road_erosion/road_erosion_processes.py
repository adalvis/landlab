"""Landlab component for road erosion processes including 
pumping, crushing, scattering (and by default, flow rerouting)

Last updated: February 27, 2025

.. codeauthor: Amanda Alvis
"""

from landlab import Component
from landlab.components import LinearDiffuser
import random as rnd
import numpy as np

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
        "surfacing__depth": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m/m",
            "mapping": "link",
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
    ):
        """Initialize TruckPassErosion.

        Parameters
        ----------
        grid : ModelGrid
            Landlab ModelGrid object
        truck_num : int, defaults to 5
            Average number of trucks to pass over a road segment in a day
        """

        super().__init__(grid)

        # Store grid and parameters
        self._grid = grid
        
        # Get elevation field
        self._elev = grid.at_node['topographic__elevation']
            
        # Get layers for sediment depths
        self._active = grid.at_node['active__depth']
        self._surfacing = grid.at_node['surfacing__depth']
        self._ballast = grid.at_node['ballast__depth']

        # Get number of trucks per day from random poisson distribution
        self._truck_num = np.random.poisson(truck_num,1).item()
		
    def run_one_step(self, tire_tracks):  
        surf_fine = self._surfacing*0.275
        surf_coarse = self._surfacing*0.725
        ball_fine = self._ballast*0.20
        ball_coarse = self._ballast*0.80

        #scattering---we need to discuss this                
        surf_coarse[tire_tracks[0]] -= 0.001*self._truck_num
        surf_coarse[tire_tracks[1]] -= 0.001*self._truck_num
        surf_coarse[tire_tracks[2]] += 0.0004*self._truck_num
        surf_coarse[tire_tracks[3]] += 0.0004*self._truck_num
        surf_coarse[tire_tracks[4]] += 0.0004*self._truck_num
        surf_coarse[tire_tracks[5]] += 0.0004*self._truck_num
        surf_coarse[tire_tracks[6]] += 0.0002*self._truck_num
        surf_coarse[tire_tracks[7]] += 0.0002*self._truck_num

        #calculate pumping fluxes
        q_ps = u_ps*(surf_fine/self._surfacing)*self._truck_num/(timeStep_Hr*3600)
        q_pb = u_pb*(ball_fine/self._ballast)*self._truck_num/(timeStep_Hr*3600)

        #calculate crushing fluxes
        q_cs = k_cs*(S_sc/S_s)*self._truck_num/(timeStep_Hr*3600)
        q_cb = k_cb*(S_bc/S_b)*self._truck_num/(timeStep_Hr*3600)

        #update surfacing
        surf_coarse[tire_tracks[0]] -= q_cs*(timeStep_Hr*3600)
        surf_coarse[tire_tracks[1]] -= q_cs*(timeStep_Hr*3600)
        surf_fine[tire_tracks[0]] += q_cs*(timeStep_Hr*3600) - \
            q_ps*(timeStep_Hr*3600) + q_pb*(timeStep_Hr*3600)
        surf_fine[tire_tracks[1]] += q_cs*(timeStep_Hr*3600) - \
            q_ps*(timeStep_Hr*3600) + q_pb*(timeStep_Hr*3600)
        self._surfacing = surf_coarse + surf_fine

        #update ballast

                               
        self.elev += surf_coarse