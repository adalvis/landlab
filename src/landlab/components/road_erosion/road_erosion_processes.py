"""Landlab component for road erosion processes including 
pumping, crushing, scattering (and by default, flow rerouting)

Last updated: May 16, 2025

.. codeauthor: Amanda Alvis
"""

from landlab import Component
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
    
    def __init__( #add layer depths as user inputs so the only information required on the DEM is the topographic elevation
        self, 
        grid, 
        centerline,
        half_width,
        full_tire,
        truck_num = 5,
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
        scat_out = 0.0005,
    ):
        """Initialize TruckPassErosion.

        Parameters
        ----------
        grid : ModelGrid
            Landlab ModelGrid object
        truck_num : int
            Average number of trucks to pass over a road segment in a day
        centerline : arraylike of int
            The location of the centerline of the road surface. 
            If using a real DEM, this should be an array that has 
            been pre-extracted. If using a synthetic, rectangular 
            grid, this should be the lower boundary's center node.
        half_width : int
            Number of nodes the truck extends to either side of 
            the centerline of the road. This is dependent on the
            node spacing of the grid.
        full_tire : boolean
            Flag to indicate whether the node spacing is that of a
            full tire width or half tire width.
        u_ps : float
            Pumping rate from surfacing to active layer per truck pass [m/truck]
        u_pb : float
            Pumping rate from ballast to surfacing per truck pass [m/truck]
        k_cs : float
            Crushing rate per truck pass in the surfacing [m/truck]
        k_cb : float
            Crushing rate per truck pass in the ballast [m/truck]
        f_af : float
            Fraction of fine material in the active layer [-]
        f_ac : float
            Fraction of coarse material in the active layer [-]
        f_sf : float
            Fraction of fine material in the surfacing [-]
        f_sc : float
            Fraction of coarse material in the surfacing [-]
        f_bf : float
            Fraction of fine material in the ballast [-]
        f_bc : float
            Fraction of coarse material in the ballast [-]
        scat_loss : float
            Total amount of coarse material being scattered in the active layer [m]
        scat_out : float
            Amount of coarse material scattered to either side of the truck tire [m]
        """

        super().__init__(grid)

        # Store grid and parameters
        self._grid = grid
        self._u_ps = u_ps
        self._u_pb = u_pb
        self._k_cs = k_cs
        self._k_cb = k_cb
        self._scat_loss = scat_loss
        self._scat_out = scat_out
        self._centerline = centerline
        self._half_width = half_width
        self._full_tire = full_tire
        
        # Get elevation fields
        self._elev = grid.at_node['topographic__elevation']
        self._active_elev = grid.at_node['active__elev']
        self._surfacing_elev = grid.at_node['surfacing__elev']
        self._ballast_elev = grid.at_node['ballast__elev']

        # Get layers for sediment depths
        self._active = grid.at_node['active__depth']
        self._surfacing = grid.at_node['surfacing__depth']
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

    def calc_tire_tracks(self):
        #grab center location of road if given a node, else use the array given
        if np.ndim(self._centerline) == 0:
            self._center = self._grid.nodes[:, self._centerline]
        else:
            self._center = self._centerline

        if self._full_tire == False:
            self._center_tracks = [
                np.concatenate((self._center-self._half_width-2, self._center-self._half_width-1,\
                self._center-self._half_width, self._center-self._half_width+1)),\
                np.concatenate((self._center+self._half_width-1, self._center+self._half_width,\
                self._center+self._half_width+1, self._center+self._half_width+2))
                ]
            self._out_center_close = [
                np.concatenate((self._center-self._half_width-3, self._center-self._half_width+2,\
                    self._center-self._half_width-4, self._center-self._half_width+3,)),\
                np.concatenate((self._center+self._half_width-2,self._center+self._half_width+3,\
                    self._center+self._half_width-3, self._center+self._half_width+4,)),\
                ]
            self._out_center_far = [
                np.concatenate((self._center-self._half_width-5, self._center-self._half_width+4,\
                    self._center-self._half_width-6, self._center-self._half_width+5,)),\
                np.concatenate((self._center+self._half_width-4, self._center+self._half_width+5,\
                    self._center+self._half_width-5, self._center+self._half_width+6,)),\
            ]
            
            self._right_tracks = [self._center_tracks[0]+1, self._center_tracks[1]+1]
            self._out_right_close = [self._out_center_close[0]+1, self._out_center_close[1]+1]
            self._out_right_far = [self._out_center_far[0]+1, self._out_center_far[1]+1]

            self._left_tracks = [self._center_tracks[0]-1, self._center_tracks[1]-1]
            self._out_left_close = [self._out_center_close[0]-1, self._out_center_close[1]-1]
            self._out_left_far = [self._out_center_far[0]-1, self._out_center_far[1]-1]
            
            val = rnd.choice([self._center_tracks[0], self._right_tracks[0],\
                self._left_tracks[0]])

            if all(val == self._center_tracks[0]):
                self._tracks = [self._center_tracks[0], self._center_tracks[1], self._out_center_close[0],\
                    self._out_center_close[1], self._out_center_far[0], self._out_center_far[1]]
            elif all(val == self._right_tracks[0]):
                self._tracks = [self._right_tracks[0], self._right_tracks[1], self._out_right_close[0],\
                    self._out_right_close[1], self._out_right_far[0], self._out_right_far[1]]    
            else:
                self._tracks = [self._left_tracks[0], self._left_tracks[1], self._out_left_close[0],\
                    self._out_left_close[1], self._out_left_far[0], self._out_left_far[1]]
        elif self._full_tire == True:
            self._right_tracks = np.concatenate((self._center-self._half_width+1,self._center-self._half_width,\
                self._center+self._half_width, self._center+self._half_width+1))
            self._out_right = [self._right_tracks-1, self._right_tracks+1]

            self._left_tracks = self._right_tracks-1
            self._out_left = [self._left_tracks-1, self._left_tracks+1]

            val = rnd.choice([self._right_tracks, self._left_tracks])

            if all(val == self._right_tracks):
                self._tracks = [self._right_tracks, self._out_right[0],\
                    self._out_right[1]]    
            else:
                self._tracks = [self._left_tracks, self._out_left[0],\
                    self._out_left[1]]
        else:
            raise ValueError("Invalid input used for full_tire. Must be True or False.")

        return(self._tracks)

    def run_one_step(self):
        self._active_init = self._active
        self._surf_init = self._surfacing
        self._ball_init = self._ballast
        self._truck_num = np.random.poisson(self._truck_num_avg,1).item()
        
        if self._truck_num == 0:
            self.tire_tracks = self.calc_tire_tracks()
            pass
        else:
            for _ in range(self._truck_num):
                self.tire_tracks = self.calc_tire_tracks() #is there a way to determine tire_tracks and get the # of truck passes per config?

                if self._full_tire == False:

                    for i in range(len(self.tire_tracks[0])):
                        if self._active_coarse[self.tire_tracks[0][i]] <= self._scat_loss and\
                            self._active_coarse[self.tire_tracks[1][i]] > self._scat_loss:   

                            self._active_coarse[self.tire_tracks[2][i]] += \
                                (self._active_coarse[self.tire_tracks[0][i]])*3/4
                            self._active_coarse[self.tire_tracks[4][i]] += \
                                (self._active_coarse[self.tire_tracks[0][i]])*1/4
                            self._active_coarse[self.tire_tracks[0][i]] -= \
                                self._active_coarse[self.tire_tracks[0][i]]
                            self._active_coarse[self.tire_tracks[3][i]] += self._scat_loss*3/4
                            self._active_coarse[self.tire_tracks[5][i]] += self._scat_loss*1/4
                            self._active_coarse[self.tire_tracks[1][i]] -= self._scat_loss

                        elif self._active_coarse[self.tire_tracks[0][i]] > self._scat_loss and\
                            self._active_coarse[self.tire_tracks[1][i]] <= self._scat_loss: 

                            self._active_coarse[self.tire_tracks[2][i]] += self._scat_loss*3/4
                            self._active_coarse[self.tire_tracks[4][i]] += self._scat_loss*1/4
                            self._active_coarse[self.tire_tracks[0][i]] -= self._scat_loss
                            self._active_coarse[self.tire_tracks[3][i]] += \
                                (self._active_coarse[self.tire_tracks[1][i]])*3/4
                            self._active_coarse[self.tire_tracks[5][i]] += \
                                (self._active_coarse[self.tire_tracks[1][i]])*1/4
                            self._active_coarse[self.tire_tracks[1][i]] -= \
                                 self._active_coarse[self.tire_tracks[1][i]]

                        elif self._active_coarse[self.tire_tracks[0][i]] <= self._scat_loss and\
                            self._active_coarse[self.tire_tracks[1][i]] <= self._scat_loss:

                            self._active_coarse[self.tire_tracks[2][i]] += \
                                (self._active_coarse[self.tire_tracks[0][i]])*3/4
                            self._active_coarse[self.tire_tracks[3][i]] += \
                                (self._active_coarse[self.tire_tracks[1][i]])*3/4
                            self._active_coarse[self.tire_tracks[4][i]] += \
                                (self._active_coarse[self.tire_tracks[0][i]])*1/4
                            self._active_coarse[self.tire_tracks[5][i]] += \
                                (self._active_coarse[self.tire_tracks[1][i]])*1/4
                            self._active_coarse[self.tire_tracks[0][i]] -= \
                                self._active_coarse[self.tire_tracks[0][i]]
                            self._active_coarse[self.tire_tracks[1][i]] -= \
                                self._active_coarse[self.tire_tracks[1][i]]
                        else:
                            self._active_coarse[self.tire_tracks[0][i]] -= self._scat_loss
                            self._active_coarse[self.tire_tracks[1][i]] -= self._scat_loss
                            self._active_coarse[self.tire_tracks[2][i]] += self._scat_loss*3/4
                            self._active_coarse[self.tire_tracks[3][i]] += self._scat_loss*3/4
                            self._active_coarse[self.tire_tracks[4][i]] += self._scat_loss*1/4
                            self._active_coarse[self.tire_tracks[5][i]] += self._scat_loss*1/4


                    #calculate pumping fluxes
                    self._q_ps = self._u_ps*(self._surf_fine/self._surfacing)
                    self._q_pb = self._u_pb*(self._ball_fine/self._ballast)

                    #calculate crushing fluxes
                    self._q_cs = self._k_cs*(self._surf_coarse/self._surfacing)
                    self._q_cb = self._k_cb*(self._ball_coarse/self._ballast)

                    #update surfacing
                    self._surf_coarse[self.tire_tracks[0:2]] -= self._q_cs[self.tire_tracks[0:2]]
                    self._surf_fine[self.tire_tracks[0:2]] += self._q_cs[self.tire_tracks[0:2]] - \
                        self._q_ps[self.tire_tracks[0:2]] + self._q_pb[self.tire_tracks[0:2]]

                    #update ballast
                    self._ball_coarse[self.tire_tracks[0:2]] -= self._q_cb[self.tire_tracks[0:2]]
                    self._ball_fine[self.tire_tracks[0:2]] += self._q_cb[self.tire_tracks[0:2]] - \
                        self._q_pb[self.tire_tracks[0:2]]

                    #update fines in active layer         
                    #determine the hiding fraction at each location
                    self._hiding_frac = [
                            self._active_coarse[self.tire_tracks[0]]/self._active[self.tire_tracks[0]],
                            self._active_coarse[self.tire_tracks[1]]/self._active[self.tire_tracks[1]]
                        ]
                    for k in range(len(self.tire_tracks[0])):
                        
                        #if the d95 of the active layer is greater than the depth of fines, use the hiding fraction
                        if self._active_coarse[self.tire_tracks[0][k]] >= (self._active_fine[self.tire_tracks[0]][k]\
                            /(1-self._hiding_frac[0][k])) and\
                           self._active_coarse[self.tire_tracks[1][k]] >= (self._active_fine[self.tire_tracks[1]][k]\
                            /(1-self._hiding_frac[1][k])):

                            self._sed_added[self.tire_tracks[0][k]] = self._sed_added[self.tire_tracks[0][k]]\
                                *(1-self._hiding_frac[0][k]) + self._q_ps[self.tire_tracks[0][k]]
                            self._active_fine[self.tire_tracks[0][k]] = self._active_fine[self.tire_tracks[0][k]]\
                                *(1-self._hiding_frac[0][k]) + self._q_ps[self.tire_tracks[0][k]]
                            self._sed_added[self.tire_tracks[1][k]] = self._sed_added[self.tire_tracks[1][k]]\
                                *(1-self._hiding_frac[1][k]) + self._q_ps[self.tire_tracks[1][k]]
                            self._active_fine[self.tire_tracks[1][k]] = self._active_fine[self.tire_tracks[1][k]]\
                                *(1-self._hiding_frac[1][k]) + self._q_ps[self.tire_tracks[1][k]]

                        elif self._active_coarse[self.tire_tracks[0][k]] < (self._active_fine[self.tire_tracks[0]][k]\
                            /(1-self._hiding_frac[0][k])) and\
                            self._active_coarse[self.tire_tracks[1][k]] >= (self._active_fine[self.tire_tracks[1]][k]\
                            /(1-self._hiding_frac[1][k])):

                            self._sed_added[self.tire_tracks[0][k]] += self._q_ps[self.tire_tracks[0][k]]
                            self._active_fine[self.tire_tracks[0][k]] += self._q_ps[self.tire_tracks[0][k]]
                            self._sed_added[self.tire_tracks[1][k]] = self._sed_added[self.tire_tracks[1][k]]\
                                *(1-self._hiding_frac[1][k]) + self._q_ps[self.tire_tracks[1][k]]
                            self._active_fine[self.tire_tracks[1][k]] = self._active_fine[self.tire_tracks[1][k]]\
                                *(1-self._hiding_frac[1][k]) + self._q_ps[self.tire_tracks[1][k]]

                        elif self._active_coarse[self.tire_tracks[0][k]] >= (self._active_fine[self.tire_tracks[0]][k]\
                            /(1-self._hiding_frac[0][k])) and\
                            self._active_coarse[self.tire_tracks[1][k]] < (self._active_fine[self.tire_tracks[1]][k]\
                            /(1-self._hiding_frac[1][k])):

                            self._sed_added[self.tire_tracks[0][k]] = self._sed_added[self.tire_tracks[0][k]]\
                                *(1-self._hiding_frac[0][k]) + self._q_ps[self.tire_tracks[0][k]]
                            self._active_fine[self.tire_tracks[0][k]] = self._active_fine[self.tire_tracks[0][k]]\
                                *(1-self._hiding_frac[0][k]) + self._q_ps[self.tire_tracks[0][k]]
                            self._sed_added[self.tire_tracks[1][k]] += self._q_ps[self.tire_tracks[1][k]]
                            self._active_fine[self.tire_tracks[1][k]] += self._q_ps[self.tire_tracks[1][k]]

                        else:
                            self._sed_added[self.tire_tracks[0][k]] += self._q_ps[self.tire_tracks[0][k]]
                            self._active_fine[self.tire_tracks[0][k]] += self._q_ps[self.tire_tracks[0][k]]
                            self._sed_added[self.tire_tracks[1][k]] += self._q_ps[self.tire_tracks[1][k]]
                            self._active_fine[self.tire_tracks[1][k]] += self._q_ps[self.tire_tracks[1][k]]

                    # self._sed_added[self.tire_tracks[0:2]] += self._q_ps[self.tire_tracks[0:2]]
                    # self._active_fine[self.tire_tracks[0:2]] += self._q_ps[self.tire_tracks[0:2]]
                elif self._full_tire == True:

                    for i in range(len(self.tire_tracks[0]) - 1):

                        if (self._active_coarse[self.tire_tracks[0][i]]) <= self._scat_loss:     
                            self._active_coarse[self.tire_tracks[1][i]] += \
                                self._active_coarse[self.tire_tracks[0][i]]/2
                            self._active_coarse[self.tire_tracks[2][i]] += \
                                self._active_coarse[self.tire_tracks[0][i]]/2
                            self._active_coarse[self.tire_tracks[0][i]] -= \
                                self._active_coarse[self.tire_tracks[0][i]]
                        else:
                            self._active_coarse[self.tire_tracks[0][i]] -= self._scat_loss
                            self._active_coarse[self.tire_tracks[1][i]] += self._scat_loss
                            self._active_coarse[self.tire_tracks[2][i]] += self._scat_loss

                    #calculate pumping fluxes
                    self._q_ps = self._u_ps*(self._surf_fine/self._surfacing)
                    self._q_pb = self._u_pb*(self._ball_fine/self._ballast)

                    #calculate crushing fluxes
                    self._q_cs = self._k_cs*(self._surf_coarse/self._surfacing)
                    self._q_cb = self._k_cb*(self._ball_coarse/self._ballast)

                    #update surfacing
                    self._surf_coarse[self.tire_tracks[0]] -= self._q_cs[self.tire_tracks[0]]
                    self._surf_fine[self.tire_tracks[0]] += self._q_cs[self.tire_tracks[0]] - \
                        self._q_ps[self.tire_tracks[0]] + self._q_pb[self.tire_tracks[0]]

                    #update ballast
                    self._ball_coarse[self.tire_tracks[0]] -= self._q_cb[self.tire_tracks[0]]
                    self._ball_fine[self.tire_tracks[0]] += self._q_cb[self.tire_tracks[0]] - \
                        self._q_pb[self.tire_tracks[0]]

                    self._hiding_frac = [
                            self._active_coarse[self.tire_tracks[0]]/self._active[self.tire_tracks[0]]
                        ]
                    #update fines in active layer
                    for k in range(len(self.tire_tracks[0])):
                        if self._active_coarse[self.tire_tracks[0][k]] >= (self._active_fine[self.tire_tracks[0][k]]):
                            self._sed_added[self.tire_tracks[0][k]] += self._q_ps[self.tire_tracks[0][k]]\
                                *(1-self._hiding_frac[0][k])
                            self._active_fine[self.tire_tracks[0][k]] += self._q_ps[self.tire_tracks[0][k]]\
                                *(1-self._hiding_frac[0][k])
                        else:
                            self._sed_added[self.tire_tracks[0][k]] += self._q_ps[self.tire_tracks[0][k]]
                            self._active_fine[self.tire_tracks[0][k]] += self._q_ps[self.tire_tracks[0][k]]

        #update outputs
        self._ball_dz =  (self._ball_coarse + self._ball_fine) - self._ball_init 
        self._ballast += self._ball_dz
        print(sum(self._ball_dz))
        self._surf_dz = (self._surf_coarse + self._surf_fine) - self._surf_init
        self._surfacing += self._surf_dz
        print(sum(self._surf_dz))
        self._active_dz = (self._active_coarse + self._active_fine) - self._active_init
        self._active += self._active_dz
        print(sum(self._active_dz))
        
        self._elev += self._ball_dz + \
            self._surf_dz + \
            self._active_dz