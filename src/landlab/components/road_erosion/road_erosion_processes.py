"""
Purpose: Road erosion processes component including pumping, crushing, scattering (and by default, flow rerouting)
Author: Amanda Alvis
Date: 02/26/2025
"""

from landlab import Component
from landlab.components import LinearDiffuser
import random as rnd
import numpy as np

class TruckPassErosion(Component):
    
    _name = 'TruckPassErosion'
    
    _input_var_names = (
        'topographic__elevation',
        'active__depth',
        'surfacing__depth',
        'ballast__depth',
    )
    
    _output_var_names = (        
        'topographic__elevation',
        'active__depth',
        'surfacing__depth',
        'ballast__depth',
    )
    
    _var_units = {
        'topographic__elevation': 'm',
        'active__depth': 'm',
        'surfacing__depth': 'm',
        'ballast__depth': 'm',
    }
    
    _var_mapping = {
        'topographic__elevation': 'node',
        'active__depth': 'node',
        'surfacing__depth': 'node',
        'ballast__depth': 'node',
    }
    
    _var_doc = {
        'topographic__elevation':
            'elevation of the ground surface relative to some datum; \
                this field gets updated',
        'active__depth':
            'elevation of the sediment surface relative to some datum; \
                this field gets updated',
        'surfacing__depth':
            'depth of surfacing layer of the road cross section relative \
                to some datum; this field gets updated',
        'ballast__depth':
            'depth of ballast layer of the road cross section relative \
                to some datum; this field gets updated',
    }
    
    
    def __init__(self, grid, truck_num = 5, diffusivity = 0.0001, **kwds):
        """Initialize TruckPassErosion.

        Parameters
        ----------
        grid : ModelGrid
            Landlab ModelGrid object

        """
        # Store grid and parameters
        self._grid = grid
        
        # Get elevation field
        try:
            self.elev = self.grid.at_node['topographic__elevation']
        except:
            raise
            
        # Get sediment field
        try:
            self.sed = self.grid.at_node['active__depth']
        except:
            raise
               
        # Instantiate linear diffuser
        self.lin_diffuse1 = LinearDiffuser(grid, linear_diffusivity=self.diffusivity)
        #self.lin_diffuse2 = LinearDiffuser(grid, linear_diffusivity=self.diffusivity, \
        #                                   values_to_diffuse = 'active__depth')
        
        #initialize truck pass and time arrays
        self.truck_pass = []
        self.time = []
		
    def run_one_step(self, tire_tracks):    
        #
        self.sed[tire_tracks] = 0      
        
        rng = np.random.RandomState(2024)
                
        self.sed[tire_tracks[0]] -= 0.001
        self.sed[tire_tracks[1]] -= 0.001
        self.sed[tire_tracks[2]] += 0.0004
        self.sed[tire_tracks[3]] += 0.0004
        self.sed[tire_tracks[4]] += 0.0004
        self.sed[tire_tracks[5]] += 0.0004
        self.sed[tire_tracks[6]] += 0.0002
        self.sed[tire_tracks[7]] += 0.0002
                               
        self.elev += self.sed