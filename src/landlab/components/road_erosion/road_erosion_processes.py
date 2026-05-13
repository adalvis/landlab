"""Landlab component for road erosion processes including 
pumping, crushing, scattering (and by default, flow rerouting)

Last updated: September 18, 2025

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
        "active__depth_fines": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of fine sediment in the active layer",
        },
        "active__depth_coarse": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of coarse sediment in the active layer",
        },
        "active__mass": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of active layer of sediment of the road cross\
                section",
        },
        "active__mass_fines": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of fine sediment in the active layer",
        },
        "active__mass_coarse": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of coarse sediment in the active layer",
        },
        "ballast__depth": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of ballast layer of the road cross section",
        },
        "ballast__depth_fines": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of fine sediment in the ballast layer",
        },
        "ballast__depth_coarse": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of coarse sediment in the ballast layer",
        },
        "ballast__mass": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of ballast layer of the road cross section",
        },
        "ballast__mass_fines": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of fine sediment in the ballast layer",
        },
        "ballast__mass_coarse": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of coarse sediment in the ballast layer",
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
        "surfacing__depth_fines": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of fine sediment in the surfacing layer",
        },
        "surfacing__depth_coarse": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of coarse sediment in the surfacing layer",
        },
        "surfacing__mass": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of surfacing layer of the road cross section",
        },
        "surfacing__mass_fines": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of fine sediment in the surfacing layer",
        },
        "surfacing__mass_coarse": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of coarse sediment in the surfacing layer",
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
        centerline,
        half_width,
        full_tire,
        truck_num = 5,
        u_ps = 6.3e-6, #(10.3g/m2) converted to depth
        u_pb = 2.3e-6, #current best guess
        k_cs = 6e-7, #current best guess
        k_cb = 2e-7, #current best guess
        scat_loss = 8e-5, #current best gues
    ):
        """Initialize TruckPassErosion.

        Parameters
        ----------
        grid : ModelGrid
            Landlab ModelGrid object
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
        truck_num : int
            Average number of trucks to pass over a road segment in a day
        u_ps : float
            Pumping rate from surfacing to active layer per truck pass [kg/truck]
        u_pb : float
            Pumping rate from ballast to surfacing per truck pass [kg/truck]
        k_cs : float
            Crushing rate per truck pass in the surfacing [kg/truck]
        k_cb : float
            Crushing rate per truck pass in the ballast [kg/truck]
        scat_loss : float
            Total amount of coarse material being scattered in the active layer
            per truck pass [kg/truck]
        """

        super().__init__(grid)

        # Store grid and parameters
        self._grid = grid
        self._u_ps = u_ps
        self._u_pb = u_pb
        self._k_cs = k_cs
        self._k_cb = k_cb
        self._scat_loss = scat_loss

        self._centerline = centerline
        self._half_width = half_width
        self._full_tire = full_tire
        
        # Get initial total sediment depth (storage)  
        # and the depth of fines/coarse  material for each layer
        self._Sa = grid.at_node["active__depth"]
        self._Saf = grid.at_node["active__depth_fines"]
        self._Sac = grid.at_node["active__depth_coarse"]
        self._Ss = grid.at_node["surfacing__depth"]
        self._Ssf = grid.at_node["surfacing__depth_fines"]
        self._Ssc = grid.at_node["surfacing__depth_coarse"]
        self._Sb = grid.at_node["ballast__depth"]
        self._Sbf = grid.at_node["ballast__depth_fines"]
        self._Sbc = grid.at_node["ballast__depth_coarse"]

        # Get initial sediment mass for each layer
        self._Ma = grid.at_node['active__mass']
        self._Maf = grid.at_node["active__mass_fines"]
        self._Mac = grid.at_node["active__mass_coarse"]
        self._Ms = grid.at_node['surfacing__mass']
        self._Msf = grid.at_node["surfacing__mass_fines"]
        self._Msc = grid.at_node["surfacing__mass_coarse"]
        self._Mb = grid.at_node['ballast__mass']
        self._Mbf = grid.at_node["ballast__mass_fines"]
        self._Mbc = grid.at_node["ballast__mass_coarse"]

        # Get elevation fields
        self._topographic_elev = grid.at_node['topographic__elevation']

        if "ballast__elevation" in grid.at_node:
            self._ballast_elev = grid.at_node["ballast__elevation"]
        else:
            self._ballast_elev = grid.add_zeros(
                "ballast__elevation", at="node", dtype=float
            )

            self._ballast_elev[:] = (
                self._topographic_elev - self._active_depth \
                - self._surfacing_depth
            )
        
        if "surfacing__elevation" in grid.at_node:
            self._surfacing_elev = grid.at_node["surfacing__elevation"]
        else:
            self._surfacing_elev = grid.add_zeros(
                "surfacing__elevation", at="node", dtype=float
            )

            self._surfacing_elev[:] = (
                self._topographic_elev - self._active_depth
            )

        # Get average number of trucks per day
        self._truck_num_avg = truck_num

        # Initialize output fields
        self.initialize_output_fields()
        self._sed_added = grid.at_node["sediment__added"]	

    @property
    def sed_added(self):
        """The depth of fine sediment added to the active layer at
        each node"""
        return self._sed_added

    def calc_tire_tracks(self):
        #Grab center location of road if given a node, else use the array given
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

            self._right_tracks = [np.concatenate((self._center-self._half_width+1,self._center-self._half_width)),\
                np.concatenate((self._center+self._half_width, self._center+self._half_width+1))]
            self._out_right = [np.concatenate((self._center-self._half_width+2, self._center-self._half_width-1)),\
                np.concatenate((self._center+self._half_width-1, self._center+self._half_width+2))]

            self._left_tracks = [self._right_tracks[0]-1, self._right_tracks[1]-1]
            self._out_left = [self._out_right[0]-1, self._out_right[1]-1]

            val = rnd.choice([self._right_tracks[0], self._left_tracks[0]])

            if all(val == self._right_tracks[0]):
                self._tracks = [self._right_tracks[0], self._right_tracks[1], self._out_right[0],\
                    self._out_right[1]]    
            else:
                self._tracks = [self._left_tracks[0], self._left_tracks[1], self._out_left[0],\
                    self._out_left[1]]
        else:
            raise ValueError("Invalid input used for full_tire. Must be True or False.")

        return(self._tracks)

    def run_one_step(self):
        self._active_init = self._active_depth.copy()
        self._surf_init = self._surfacing_depth.copy()
        self._ball_init = self._ballast_depth.copy()
        self.truck_num = np.random.poisson(self._truck_num_avg,1).item()
        
        if self.truck_num == 0:
            self.tire_tracks = self.calc_tire_tracks()
            pass
        else:
            for _ in range(self.truck_num):
                self.tire_tracks = self.calc_tire_tracks() 

                if self._full_tire == False:

                    for i in range(len(self.tire_tracks[0])):
                        if self._Mac[self.tire_tracks[0][i]] <= self._scat_loss and\
                            self._Mac[self.tire_tracks[1][i]] > self._scat_loss:   

                            self._Mac[self.tire_tracks[2][i]] += \
                                (self._Mac[self.tire_tracks[0][i]])*3/4
                            self._Mac[self.tire_tracks[4][i]] += \
                                (self._Mac[self.tire_tracks[0][i]])*1/4
                            self._Mac[self.tire_tracks[0][i]] -= \
                                self._Mac[self.tire_tracks[0][i]]
                            self._Mac[self.tire_tracks[3][i]] += self._scat_loss*3/4
                            self._Mac[self.tire_tracks[5][i]] += self._scat_loss*1/4
                            self._Mac[self.tire_tracks[1][i]] -= self._scat_loss

                        elif self._Mac[self.tire_tracks[0][i]] > self._scat_loss and\
                            self._Mac[self.tire_tracks[1][i]] <= self._scat_loss: 

                            self._Mac[self.tire_tracks[2][i]] += self._scat_loss*3/4
                            self._Mac[self.tire_tracks[4][i]] += self._scat_loss*1/4
                            self._Mac[self.tire_tracks[0][i]] -= self._scat_loss
                            self._Mac[self.tire_tracks[3][i]] += \
                                (self._Mac[self.tire_tracks[1][i]])*3/4
                            self._Mac[self.tire_tracks[5][i]] += \
                                (self._Mac[self.tire_tracks[1][i]])*1/4
                            self._Mac[self.tire_tracks[1][i]] -= \
                                 self._Mac[self.tire_tracks[1][i]]

                        elif self._Mac[self.tire_tracks[0][i]] <= self._scat_loss and\
                            self._Mac[self.tire_tracks[1][i]] <= self._scat_loss:

                            self._Mac[self.tire_tracks[2][i]] += \
                                (self._Mac[self.tire_tracks[0][i]])*3/4
                            self._Mac[self.tire_tracks[3][i]] += \
                                (self._Mac[self.tire_tracks[1][i]])*3/4
                            self._Mac[self.tire_tracks[4][i]] += \
                                (self._Mac[self.tire_tracks[0][i]])*1/4
                            self._Mac[self.tire_tracks[5][i]] += \
                                (self._Mac[self.tire_tracks[1][i]])*1/4
                            self._Mac[self.tire_tracks[0][i]] -= \
                                self._Mac[self.tire_tracks[0][i]]
                            self._Mac[self.tire_tracks[1][i]] -= \
                                self._Mac[self.tire_tracks[1][i]]
                        else:
                            self._Mac[self.tire_tracks[0][i]] -= self._scat_loss
                            self._Mac[self.tire_tracks[1][i]] -= self._scat_loss
                            self._Mac[self.tire_tracks[2][i]] += self._scat_loss*3/4
                            self._Mac[self.tire_tracks[3][i]] += self._scat_loss*3/4
                            self._Mac[self.tire_tracks[4][i]] += self._scat_loss*1/4
                            self._Mac[self.tire_tracks[5][i]] += self._scat_loss*1/4
                
                elif self._full_tire == True:

                    for i in range(len(self.tire_tracks[0])):
                        
                        if self._Mac[self.tire_tracks[0][i]] <= self._scat_loss and\
                            self._Mac[self.tire_tracks[1][i]] > self._scat_loss:   

                            self._Mac[self.tire_tracks[2][i]] += \
                                (self._Mac[self.tire_tracks[0][i]])
                            self._Mac[self.tire_tracks[0][i]] -= \
                                self._Mac[self.tire_tracks[0][i]]
                            self._Mac[self.tire_tracks[3][i]] += self._scat_loss
                            self._Mac[self.tire_tracks[1][i]] -= self._scat_loss

                        elif self._Mac[self.tire_tracks[0][i]] > self._scat_loss and\
                            self._Mac[self.tire_tracks[1][i]] <= self._scat_loss: 

                            self._Mac[self.tire_tracks[2][i]] += self._scat_loss
                            self._Mac[self.tire_tracks[0][i]] -= self._scat_loss
                            self._Mac[self.tire_tracks[3][i]] += \
                                (self._Mac[self.tire_tracks[1][i]])
                            self._Mac[self.tire_tracks[1][i]] -= \
                                 self._Mac[self.tire_tracks[1][i]]

                        elif self._Mac[self.tire_tracks[0][i]] <= self._scat_loss and\
                            self._Mac[self.tire_tracks[1][i]] <= self._scat_loss:

                            self._Mac[self.tire_tracks[2][i]] += \
                                (self._Mac[self.tire_tracks[0][i]])
                            self._Mac[self.tire_tracks[3][i]] += \
                                (self._Mac[self.tire_tracks[1][i]])
                            self._Mac[self.tire_tracks[0][i]] -= \
                                self._Mac[self.tire_tracks[0][i]]
                            self._Mac[self.tire_tracks[1][i]] -= \
                                self._Mac[self.tire_tracks[1][i]]
                        else:
                            self._Mac[self.tire_tracks[0][i]] -= self._scat_loss
                            self._Mac[self.tire_tracks[1][i]] -= self._scat_loss
                            self._Mac[self.tire_tracks[2][i]] += self._scat_loss
                            self._Mac[self.tire_tracks[3][i]] += self._scat_loss


                area = self.grid.area_of_cell[self.grid.cell_at_node]

                self._Sac = self._Mac/((1-self._porosity)*rho_s*area)
                
                #Pumping fluxes (this is per truck pass)
                self._q_ps = self._u_ps
                self._q_pb = self._u_pb

                #Crushing fluxes
                self._q_cs = self._k_cs*(self._Msc/self._Ms)
                self._q_cb = self._k_cb*(self._Mbc/self._Mb)

                #update surfacing
                self._Msc[self.tire_tracks[0:2]] -= self._q_cs[self.tire_tracks[0:2]]
                self._Msf[self.tire_tracks[0:2]] += self._q_cs[self.tire_tracks[0:2]] - \
                    self._q_ps[self.tire_tracks[0:2]] + self._q_pb[self.tire_tracks[0:2]]

                #update ballast
                self._Mbc[self.tire_tracks[0:2]] -= self._q_cb[self.tire_tracks[0:2]]
                self._Mbf[self.tire_tracks[0:2]] += self._q_cb[self.tire_tracks[0:2]] - \
                    self._q_pb[self.tire_tracks[0:2]]

                #update fines in active layer         
                # self._sed_added[self.tire_tracks[0:2]] += self._q_ps[self.tire_tracks[0:2]]
                self._Maf[self.tire_tracks[0:2]] += self._q_ps[self.tire_tracks[0:2]]

        #update outputs
        self._ballast_depth += ((self._ball_coarse + self._ball_fine) - self._ball_init)
        self._surfacing_depth += ((self._surf_coarse + self._surf_fine) - self._surf_init)
        self._active_depth += ((self._Mac + self._active_fine) - self._active_init)
        
        self._ballast_elev += (
            self._ballast_depth - self._ball_init
            )
        self._surfacing_elev[:] =(
            self._ballast_elev[:] + self._surfacing_depth[:]
        )
        self._topographic_elev[:] = (
            self._ballast_elev[:] + self._surfacing_depth[:] + self._active_depth[:]
            )
        