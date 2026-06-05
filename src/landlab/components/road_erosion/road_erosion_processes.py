"""Landlab component for road erosion processes including 
pumping, crushing, scattering (and by default, flow rerouting)

Last updated: May 21, 2026

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
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of fine sediment in the active layer",
        },
        "active__depth_coarse": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of coarse sediment in the active layer",
        },
        "active__mass": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of sediment in the active layer",
        },
        "active__mass_fines": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of fine sediment in the active layer",
        },
        "active__mass_coarse": {
            "dtype": float,
            "intent": "out",
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
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of fine sediment in the ballast layer",
        },
        "ballast__depth_coarse": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of coarse sediment in the ballast layer",
        },
        "ballast__mass": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of sediment in the ballast layer",
        },
        "ballast__mass_fines": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of fine sediment in the ballast layer",
        },
        "ballast__mass_coarse": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of coarse sediment in the ballast layer",
        },
        "scattering__flux": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of sediment scattered in the surfacing layer",
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
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of fine sediment in the surfacing layer",
        },
        "surfacing__depth_coarse": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "depth of coarse sediment in the surfacing layer",
        },
        "surfacing__mass": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of sediment in the surfacing layer",
        },
        "surfacing__mass_fines": {
            "dtype": float,
            "intent": "out",
            "optional": False,
            "units": "kg",
            "mapping": "node",
            "doc": "mass of fine sediment in the surfacing layer",
        },
        "surfacing__mass_coarse": {
            "dtype": float,
            "intent": "out",
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
        rho_s = 2650, #kg/m^3
        porosity_c = 0.35,
        porosity_f = 0.35,
        u_ps = 6.3e-6, #(10.3g/m2) converted to depth
        u_pb = 2.3e-6, #current best guess
        k_cs = 6e-7, #current best guess
        k_cb = 2e-7, #current best guess
        scat_loss = 8e-5, #current best guess,
        d95 = 0.019,
        F_af0 = 0.5, #initial fraction of fines in interstitial spaces of coarse in active layer
        F_sf0 = 1,
        F_bc0 = 0.5,
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
        self._area = grid.dx*grid.dy

        # Store grid and parameters
        self._grid = grid
        self._rho_s = rho_s
        self._phi_c = porosity_c
        self._phi_f = porosity_f
        self._u_ps = self._phi_c*u_ps*(1-self._phi_f)*self._rho_s*self._area
        self._u_pb = self._phi_c*u_pb*(1-self._phi_f)*self._rho_s*self._area
        self._k_cs = k_cs*(1-self._phi_c)*self._rho_s*self._area
        self._k_cb = k_cb*(1-self._phi_c)*self._rho_s*self._area
        self._scat_loss = scat_loss
        self._d95 = d95
        self._F_af0 = F_af0
        self._F_sf0 = F_sf0
        self._F_bc0 = F_bc0

        self._centerline = centerline
        self._half_width = half_width
        self._full_tire = full_tire
        
        # Get initial total sediment depth (storage)  
        # and the depth of fines/coarse  material for each layer
        self._Sa = grid.at_node["active__depth"]
        self._Ss = grid.at_node["surfacing__depth"]
        self._Sb = grid.at_node["ballast__depth"]

        # Get elevation fields
        self._topographic_elev = grid.at_node['topographic__elevation']

        if "ballast__elevation" in grid.at_node:
            self._ballast_elev = grid.at_node["ballast__elevation"]
        else:
            self._ballast_elev = grid.add_zeros(
                "ballast__elevation", at="node", dtype=float
            )

            self._ballast_elev[:] = (
                self._topographic_elev - self._Sa \
                - self._Ss
            )
        
        if "surfacing__elevation" in grid.at_node:
            self._surfacing_elev = grid.at_node["surfacing__elevation"]
        else:
            self._surfacing_elev = grid.add_zeros(
                "surfacing__elevation", at="node", dtype=float
            )

            self._surfacing_elev[:] = (
                self._topographic_elev - self._Sa
            )

        # Get average number of trucks per day
        self._truck_num_avg = truck_num

        # Initialize output fields
        self.initialize_output_fields()
        self._sed_added = grid.at_node["sediment__added"]

        self._Saf = grid.at_node["active__depth_fines"]
        self._Sac = grid.at_node["active__depth_coarse"]
        self._Ssf = grid.at_node["surfacing__depth_fines"]
        self._Ssc = grid.at_node["surfacing__depth_coarse"]
        self._Sbf = grid.at_node["ballast__depth_fines"]
        self._Sbc = grid.at_node["ballast__depth_coarse"]	

        self._Ma = grid.at_node["active__mass"]
        self._Maf = grid.at_node["active__mass_fines"]
        self._Mac = grid.at_node["active__mass_coarse"]
        self._Ms = grid.at_node["surfacing__mass"]
        self._Msf = grid.at_node["surfacing__mass_fines"]
        self._Msc = grid.at_node["surfacing__mass_coarse"]
        self._Mb = grid.at_node["ballast__mass"]
        self._Mbf = grid.at_node["ballast__mass_fines"]
        self._Mbc = grid.at_node["ballast__mass_coarse"]

        self._Sac[:] = self._d95
        self._Saf[:] = self._F_af0*self._Sac
        self._Ssc[:] = self._Ss.copy()
        self._Ssf[:] = self._F_sf0*self._Ssc
        self._Sbf[:] = self._Sb.copy()
        self._Sbc[:] = self._F_bc0*self._Sbf

        self._Maf[:] = self._Saf*(1-self._phi_f)*self._rho_s*self._area*self._phi_c
        self._Mac[:] = self._Sac*(1-self._phi_c)*self._rho_s*self._area
        self._Ma[:] = self._Mac + self._Maf
        self._Msc[:] = self._Ssc*(1-self._phi_c)*self._rho_s*self._area
        self._Msf[:] = self._Ssf*(1-self._phi_f)*self._rho_s*self._area*self._phi_c
        self._Ms[:] = self._Msf + self._Msc
        self._Mbc[:] = self._Sbc*(1-self._phi_c)*self._rho_s*self._area*self._phi_f
        self._Mbf[:] = self._Sbf*(1-self._phi_f)*self._rho_s*self._area
        self._Mb[:] = self._Mbf + self._Mbc

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
            
            val = np.random.choice(["center", "right",\
                "left"])

            if val == "center":
                self._tracks = [self._center_tracks[0], self._center_tracks[1], self._out_center_close[0],\
                    self._out_center_close[1], self._out_center_far[0], self._out_center_far[1]]
            elif val == "right":
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

            val = np.random.choice(["right", "left"])

            if val == "right":
                self._tracks = [self._right_tracks[0], self._right_tracks[1], self._out_right[0],\
                    self._out_right[1]]    
            else:
                self._tracks = [self._left_tracks[0], self._left_tracks[1], self._out_left[0],\
                    self._out_left[1]]
        else:
            raise ValueError("Invalid input used for full_tire. Must be True or False.")

        return(self._tracks)

    def run_one_step(self):
        self._elev_init = self._topographic_elev.copy()
        self._ball_init = self._ballast_elev.copy()
        self._surf_init = self._surfacing_elev.copy()
        self._Ma_init = self._Ma.copy()
        self._Sa_init = self._Sa.copy()
        self._Saf_init = self._Saf.copy()
        self._Ms_init = self._Ms.copy()
        self._Ss_init = self._Ss.copy()
        self._Mb_init = self._Mb.copy()
        self._Sb_init = self._Sb.copy()
        self.truck_num = np.random.poisson(self._truck_num_avg,1).item()
        # self.truck_num=4
        
        if self.truck_num == 0:
            self.tire_tracks = self.calc_tire_tracks()
            pass
        else:
            for _ in range(self.truck_num):
                self.tire_tracks = self.calc_tire_tracks() 
                if self._full_tire == False:
                    for i in range(len(self.tire_tracks[0:2])):
                    
                        #Scattering flux
                        if self._Saf[i] <= self._Sac[i]:
                            self._q_scat_f = self._scat_loss*(1 - self._Saf[i]/self._Sac[i])*\
                                (1-self._phi_f)*self._phi_c*self._rho_s*self._area
                            self._q_scat_c = self._scat_loss*(1 - self._Saf[i]/self._Sac[i])*\
                                (1-self._phi_c)*self._rho_s*self._area
                        else:
                            self._q_scat_f = 0
                            self._q_scat_c = 0

                        self._Msc[self.tire_tracks[2:4][i]] += \
                            self._q_scat_c*3/4
                        self._Msc[self.tire_tracks[4:][i]] += \
                            self._q_scat_c*1/4
                        self._Msc[self.tire_tracks[0:2][i]] -= \
                            self._q_scat_c

                        self._Msf[self.tire_tracks[0:2][i]] -= \
                            self._q_scat_f

                        self._Maf[self.tire_tracks[0:2][i]] += \
                            self._q_scat_f
                        
                        
                elif self._full_tire == True:
                    for i in range(len(self.tire_tracks[0])):
                        #Scattering flux
                        if self._Saf[i] <= self._Sac[i]:
                            self._q_scat_f = self._scat_loss*(1 - self._Saf[i]/self._Sac[i])*\
                                (1-self._phi_f)*self._phi_c*self._rho_s*self._area
                            self._q_scat_c = self._scat_loss*(1 - self._Saf[i]/self._Sac[i])*\
                                (1-self._phi_c)*self._rho_s*self._area
                        else:
                            self._q_scat_f = 0
                            self._q_scat_c = 0

                        self._Msc[self.tire_tracks[2:4][i]] += \
                            self._q_scat_c
                        self._Msc[self.tire_tracks[0:2][i]] -= \
                            self._q_scat_c

                        self._Msf[self.tire_tracks[0:2][i]] -= \
                            self._q_scat_f

                        self._Maf[self.tire_tracks[0:2][i]] += \
                            self._q_scat_f

                #Pumping fluxes (this is per truck pass)
                self._q_ps = self._u_ps*np.ones(len(self._Ms))
                self._q_pb = self._u_pb*np.ones(len(self._Mb))

                #Crushing fluxes
                self._q_cs = self._k_cs*(self._Msc/self._Ms)
                self._q_cb = self._k_cb*(self._Mbc/self._Mb)

                #update surfacing post-scatter
                self._Msc[self.tire_tracks[0:2]] -= self._q_cs[self.tire_tracks[0:2]]
                self._Msf[self.tire_tracks[0:2]] += self._q_cs[self.tire_tracks[0:2]] - \
                    self._q_ps[self.tire_tracks[0:2]] + self._q_pb[self.tire_tracks[0:2]]

                #update ballast
                self._Mbc[self.tire_tracks[0:2]] -= self._q_cb[self.tire_tracks[0:2]]
                self._Mbf[self.tire_tracks[0:2]] += self._q_cb[self.tire_tracks[0:2]] - \
                    self._q_pb[self.tire_tracks[0:2]]

                #update fines in active layer         
                self._sed_added[self.tire_tracks[0:2]] += self._q_ps[self.tire_tracks[0:2]]
                self._Maf[self.tire_tracks[0:2]] += self._q_ps[self.tire_tracks[0:2]]
                
                Maf_crit = self._phi_c*self._d95*(1-self._phi_f)*self._rho_s*self._area
                
                for i in range(len(self._Maf)):
                    if self._Maf[i] <= Maf_crit:
                        self._Saf[i] = self._Maf[i]/(self._phi_c*(1-self._phi_f)*self._rho_s*self._area)
                    elif self._Maf[i] > Maf_crit:
                        self._Saf[i] = (self._Maf[i]/((1-self._phi_f)*self._rho_s*self._area)\
                            + self._d95*((1-self._phi_c)/(1-self._phi_f)))*(1/(self._phi_c + 1))
                        

                self._Ssc[:] = self._Msc/((1-self._phi_c)*self._rho_s*self._area)
                Msf_crit = self._phi_c*self._Ssc*(1-self._phi_f)*self._rho_s*self._area

                for i in range(len(self._Msf)):
                    if self._Msf[i] <= Msf_crit[i]:
                        self._Ssf[i] = self._Msf[i]/(self._phi_c*(1-self._phi_f)*self._rho_s*self._area)
                    else:
                        self._Ssf[i] = (self._Msf[i]/((1-self._phi_f)*self._rho_s*self._area) \
                            + self._Ssc[i]*((1-self._phi_c)/(1-self._phi_f)))*(1/(self._phi_c + 1))

                self._Sbf[:] = self._Mbf/((1-self._phi_c)*self._rho_s*self._area)
                Mbc_crit = self._phi_c*self._Sbf*(1-self._phi_f)*self._rho_s*self._area
                
                for i in range(len(self._Mbc)):
                    if self._Mbc[i] <= Mbc_crit[i]:
                        self._Sbc[i] = self._Mbc[i]/(self._phi_f*(1-self._phi_c)*self._rho_s*self._area)
                    else:
                        self._Sbc[i] = (self._Mbc[i]/((1-self._phi_c)*self._rho_s*self._area) \
                            + self._Sbf[i]*((1-self._phi_f)/(1-self._phi_c)))*(1/(self._phi_f + 1))

        #update outputs
        self._Mb[:] = self._Mbf + self._Mbc
        self._Ms[:] = self._Msf + self._Msc
        self._Ma[:] = self._Maf + (self._d95*(1-self._phi_c)*self._rho_s*self._area)
        self._Sb[:] = np.maximum(self._Sbf, self._Sbc)
        self._Ss[:] = np.maximum(self._Ssf, self._Ssc)
        self._Sa[:] = np.maximum(self._Saf, self._d95)

        print("Mass conservation:", np.round((self._Ma - self._Ma_init).sum()+\
            (self._Ms - self._Ms_init).sum()+(self._Mb - self._Mb_init).sum(),4))
        print("Depth (?) conservation:", (self._Sa - self._Sa_init).sum()+\
            (self._Ss - self._Ss_init).sum() + (self._Sb - self._Sb_init).sum())

        self._ballast_elev += (
            self._Sb - self._Sb_init
            )

        self._surfacing_elev += (
            (self._Sb - self._Sb_init)
            + (self._Ss - self._Ss_init)
        )
        
        self._topographic_elev += (
            (self._Sb - self._Sb_init)
            + (self._Ss - self._Ss_init)
            + (self._Sa - self._Sa_init)
            )
        