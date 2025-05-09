"""Landlab component for road erosion processes including 
pumping, crushing, scattering (and by default, flow rerouting)

Last updated: May 09, 2025

.. codeauthor: Amanda Alvis
"""

from landlab import Component
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
        scat_loss = 0.001,
        scat_out = 0.0004,
        scat_back = 0.0001
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
        scat_loss : float
            Total amount of coarse material being scattered in the active layer
        scat_out : float
            Amount of coarse material scattered to either side of the truck tire
        scat_back : float
            Amount of coarse material scattered immediately behind the truck tire
        """

        super().__init__(grid)

        # Store grid and parameters
        self._grid = grid
        self._d95 = d95
        self._u_ps = u_ps
        self._u_pb = u_pb
        self._k_cs = k_cs
        self._k_cb = k_cb
        self._scat_loss = scat_loss
        self._scat_out = scat_out
        self._scat_back = scat_back
        
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
        self._truck_num_avg = truck_num

        self.initialize_output_fields()
        self._sed_added = grid.at_node["sediment__added"]
	
    @property
    def sed_added(self):
        """The depth of fine sediment added to the active layer at
        each node"""
        return self._sed_added

    def calc_tire_tracks(self, centerline, half_width, full_tire):
        #grab center location of road if given a node, else use the array given
        if np.ndim(centerline) == 0:
            self._center = self._grid.nodes[:, centerline]
        else:
            self._center = centerline

        if full_tire == False:
            self._center_tracks = [
                np.concatenate((self._center-half_width-2, self._center-half_width-1,\
                self._center-half_width, self._center-half_width+1)),\
                np.concatenate((self._center+half_width-1, self._center+half_width,\
                self._center+half_width+1, self._center+half_width+2))
                ]
            self._out_center = [
                np.concatenate((self._center-half_width-3, self._center-half_width+2,\
                self._center+half_width-2,self._center+half_width+3)),\
                np.concatenate((self._center-half_width-4, self._center-half_width+3,\
                self._center+half_width-3,self._center+half_width+4)),\
                ]
            self._back_center = [
                self._center_tracks[0]+ self._grid.number_of_node_columns, 
                self._center_tracks[1]+ self._grid.number_of_node_columns
                ]
            
            self._right_tracks = [self._center_tracks[0]+1, self._center_tracks[1]+1]
            self._out_right = [self._out_center[0]+1, self._out_center[1]+1]
            self._back_right = [self._back_center[0]+1, self._back_center[1]+1]

            self._left_tracks = [self._center_tracks[0]-1, self._center_tracks[1]-1]
            self._out_left = [self._out_center[0]-1, self._out_center[1]-1]
            self._back_left = [self._back_center[0]-1, self._back_center[1]-1]
            
            val = rnd.choice([self._center_tracks[0], self._right_tracks[0],\
                self._left_tracks[0]])

            if all(val == self._center_tracks[0]):
                self._tracks = [self._center_tracks[0], self._center_tracks[1], self._out_center[0],\
                    self._out_center[1], self._back_center[0], self._back_center[1]]
            elif all(val == self._right_tracks[0]):
                self._tracks = [self._right_tracks[0], self._right_tracks[1], self._out_right[0],\
                    self._out_right[1], self._back_right[0], self._back_right[1]]    
            else:
                self._tracks = [self._left_tracks[0], self._left_tracks[1], self._out_left[0],\
                    self._out_left[1], self._back_left[0], self._back_left[1]]
        elif full_tire == True:
            self._right_tracks = np.concatenate((self._center-half_width+1,self._center-half_width,\
                self._center+half_width, self._center+half_width+1))
            self._out_right = [self._right_tracks-1, self._right_tracks+1]
            self._back_right = self._right_tracks+\
                self._grid.number_of_node_columns

            self._left_tracks = self._right_tracks-1
            self._out_left = [self._left_tracks-1, self._left_tracks+1]
            self._back_left = self._left_tracks+\
                self._grid.number_of_node_columns

            val = rnd.choice([self._right_tracks, self._left_tracks])

            if all(val == self._right_tracks):
                self._tracks = [self._right_tracks, self._out_right[0],\
                    self._out_right[1], self._back_right]    
            else:
                self._tracks = [self._left_tracks, self._out_left[0],\
                    self._out_left[1], self._back_left]
        else:
            raise ValueError("Invalid input used for full_tire. Must be True or False.")

        return(self._tracks)

    def run_one_step(self, centerline, half_width, full_tire): 
        self._active_init = self._active
        self._surf_init = self._surfacing
        self._ball_init = self._ballast
        self._truck_num = np.random.poisson(self._truck_num_avg,1).item()

        for _ in range(self._truck_num):
            self._tire_tracks = self.calc_tire_tracks(centerline, half_width, full_tire)

            if full_tire == False:
                #scattering 
                for i in range(len(self._tire_tracks[0]) - 1):
                    #Set bottom boundary of active layer
                    if (self._active_coarse[self._tire_tracks[0][i]] or self._active_coarse[self._tire_tracks[1][i]])\
                        <= self._scat_loss:     
                        self._active_coarse[self._tire_tracks[2][i]] += \
                            (self._active_coarse[self._tire_tracks[0][i]]/2.5)*0.75
                        self._active_coarse[self._tire_tracks[3][i]] += \
                            (self._active_coarse[self._tire_tracks[1][i]]/2.5)*0.25

                        if self._tire_tracks[5][i] < len(self._active_coarse):
                            self._active_coarse[self._tire_tracks[4][i]] += \
                                self._active_coarse[self._tire_tracks[0][i]]/5
                            self._active_coarse[self._tire_tracks[5][i]] += \
                                self._active_coarse[self._tire_tracks[1][i]]/5
                        self._active_coarse[self._tire_tracks[0][i]] -= \
                            self._active_coarse[self._tire_tracks[0][i]]
                        self._active_coarse[self._tire_tracks[1][i]] -= \
                            self._active_coarse[self._tire_tracks[1][i]]
                    else:
                        self._active_coarse[self._tire_tracks[0][i]] -= self._scat_loss
                        self._active_coarse[self._tire_tracks[1][i]] -= self._scat_loss
                        self._active_coarse[self._tire_tracks[2][i]] += self._scat_out*0.75
                        self._active_coarse[self._tire_tracks[3][i]] += self._scat_out*0.25
                        if self._tire_tracks[5][i] <= len(self._active_coarse):
                            self._active_coarse[self._tire_tracks[4][i]] += self._scat_back
                            self._active_coarse[self._tire_tracks[5][i]] += self._scat_back

                #calculate pumping fluxes
                self._q_ps = self._u_ps*(self._surf_fine/self._surfacing)/_DAY_SEC
                self._q_pb = self._u_pb*(self._ball_fine/self._ballast)/_DAY_SEC

                #calculate crushing fluxes
                self._q_cs = self._k_cs*(self._surf_coarse/self._surfacing)/_DAY_SEC
                self._q_cb = self._k_cb*(self._ball_coarse/self._ballast)/_DAY_SEC

                #update surfacing
                self._surf_coarse[self._tire_tracks[0:1]] -= self._q_cs[self._tire_tracks[0:1]]*_DAY_SEC
                self._surf_fine[self._tire_tracks[0:1]] += self._q_cs[self._tire_tracks[0:1]]*_DAY_SEC - \
                    self._q_ps[self._tire_tracks[0:1]]*_DAY_SEC + self._q_pb[self._tire_tracks[0:1]]*_DAY_SEC

                #update ballast
                self._ball_coarse[self._tire_tracks[0:1]] -= self._q_cb[self._tire_tracks[0:1]]*_DAY_SEC
                self._ball_fine[self._tire_tracks[0:1]] += self._q_cb[self._tire_tracks[0:1]]*_DAY_SEC - \
                    self._q_pb[self._tire_tracks[0:1]]*_DAY_SEC

                #update fines in active layer
                for k in range(len(self._tire_tracks[0]) - 1):                
                    #determine the hiding fraction at each location
                    self._hiding_frac = [
                        self._active_coarse[self._tire_tracks[0][k]]/self._active[self._tire_tracks[0][k]],
                        self._active_coarse[self._tire_tracks[1][k]]/self._active[self._tire_tracks[1][k]]
                    ]
                    
                    #if the d95 of the active layer is greater than the depth of fines, use the hiding fraction
                    if self._d95 >= (self._active_fine[self._tire_tracks[0][k]] or \
                        self._active_fine[self._tire_tracks[1][k]]):
                        self._sed_added[self._tire_tracks[0][k]] += self._q_ps[self._tire_tracks[0][k]]*_DAY_SEC/\
                            (1-self._hiding_frac[0])
                        self._sed_added[self._tire_tracks[1][k]] += self._q_ps[self._tire_tracks[1][k]]*_DAY_SEC/\
                            (1-self._hiding_frac[1])  
                    else:
                        self._sed_added[self._tire_tracks[0][k]] += self._q_ps[self._tire_tracks[0][k]]*_DAY_SEC
                        self._sed_added[self._tire_tracks[1][k]] += self._q_ps[self._tire_tracks[1][k]]*_DAY_SEC
                    
                    self._active_fine[self._tire_tracks[0][k]] += self._sed_added[self._tire_tracks[0][k]]
                    self._active_fine[self._tire_tracks[1][k]] += self._sed_added[self._tire_tracks[1][k]]
            
            elif full_tire == True:
                for i in range(len(self._tire_tracks[0]) - 1):
                    if (self._active_coarse[self._tire_tracks[0][i]]) <= self._scat_loss:     
                        self._active_coarse[self._tire_tracks[1][i]] += \
                            self._active_coarse[self._tire_tracks[0][i]]/2.5
                        self._active_coarse[self._tire_tracks[2][i]] += \
                            self._active_coarse[self._tire_tracks[0][i]]/2.5
                        if self._tire_tracks[3][i] <= len(self._active_coarse):
                            self._active_coarse[self._tire_tracks[3][i]] += \
                                self._active_coarse[self._tire_tracks[0][i]]/5
                        self._active_coarse[self._tire_tracks[0][i]] -= \
                            self._active_coarse[self._tire_tracks[0][i]]
                    else:
                        self._active_coarse[self._tire_tracks[0][i]] -= self._scat_loss
                        self._active_coarse[self._tire_tracks[1][i]] += self._scat_out
                        self._active_coarse[self._tire_tracks[2][i]] += self._scat_out
                        if self._tire_tracks[3][i] <= len(self._active_coarse):
                            self._active_coarse[self._tire_tracks[3][i]] += self._scat_back

                #calculate pumping fluxes
                self._q_ps = self._u_ps*(self._surf_fine/self._surfacing)/_DAY_SEC
                self._q_pb = self._u_pb*(self._ball_fine/self._ballast)/_DAY_SEC

                #calculate crushing fluxes
                self._q_cs = self._k_cs*(self._surf_coarse/self._surfacing)/_DAY_SEC
                self._q_cb = self._k_cb*(self._ball_coarse/self._ballast)/_DAY_SEC

                #update surfacing
                self._surf_coarse[self._tire_tracks[0]] -= self._q_cs[self._tire_tracks[0]]*_DAY_SEC
                self._surf_fine[self._tire_tracks[0]] += self._q_cs[self._tire_tracks[0]]*_DAY_SEC - \
                    self._q_ps[self._tire_tracks[0]]*_DAY_SEC + self._q_pb[self._tire_tracks[0]]*_DAY_SEC

                #update ballast
                self._ball_coarse[self._tire_tracks[0]] -= self._q_cb[self._tire_tracks[0]]*_DAY_SEC
                self._ball_fine[self._tire_tracks[0]] += self._q_cb[self._tire_tracks[0]]*_DAY_SEC - \
                    self._q_pb[self._tire_tracks[0]]*_DAY_SEC

                #update fines in active layer
                for k in range(len(self._tire_tracks[0]) - 1):
                    self._hiding_frac = [
                        self._active_coarse[self._tire_tracks[0][k]]/self._active[self._tire_tracks[0][k]]
                    ]
                    if self._d95 >= (self._active_fine[self._tire_tracks[0][k]]):
                        self._sed_added[self._tire_tracks[0][k]] += self._q_ps[self._tire_tracks[0][k]]*_DAY_SEC/\
                            (1-self._hiding_frac[0])
                    else:
                        self._sed_added[self._tire_tracks[0][k]] += self._q_ps[self._tire_tracks[0][k]]*_DAY_SEC
                    
                    self._active_fine[self._tire_tracks[0][k]] += self._sed_added[self._tire_tracks[0][k]]

        #update outputs
        self._ballast = self._ball_coarse + self._ball_fine
        self._ball_dz = self._ballast - self._ball_init
        self._surfacing = self._surf_coarse + self._surf_fine
        self._surf_dz = self._surfacing - self._surf_init
        self._active = self._active_coarse + self._active_fine
        self._active_dz = self._active - self._active_init
        
        self._elev += self._ball_dz + \
            self._surf_dz + \
            self._active_dz