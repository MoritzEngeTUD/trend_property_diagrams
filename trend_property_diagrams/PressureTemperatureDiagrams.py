import trend_property_diagrams.trend_wrapper as trend
import trend_property_diagrams as tpd
import numpy as np
from matplotlib import pyplot as plt
import os
from parallelbar import progress_map
import pandas as pd
from pathlib import Path
from tqdm import tqdm

P_MIN = 1e-3  # Pa
P_MAX = 210*1e6  # Pa

T_AMB = 25 + 273.15  # K
p_AMB = 101325  # Pa


class PropertyDiagrams():
    def __init__(self, fluid_name: str, N : int = 100, n_cpu :int = os.cpu_count()-1):
        self.fluid_name = fluid_name
        self.N = N
        self.n_cpu = n_cpu

        self.main_path = Path(tpd.__file__).parent.parent
        if self.main_path.joinpath('res').exists():
            print(f"Using existing results folder: {self.main_path.joinpath('res')}")
        else:
            print(f"Creating results folder: {self.main_path.joinpath('res')}")
            self.main_path.joinpath('res').mkdir(parents=True, exist_ok=True)
        self.save_path = self.main_path.joinpath('res')

        if not self.main_path.joinpath('res').joinpath('data').exists():
            print(f"Creating data results folder: {self.main_path.joinpath('res').joinpath('data')}")
            self.main_path.joinpath('res').joinpath('data').mkdir(parents=True, exist_ok=True)
        if not self.main_path.joinpath('res').joinpath('plots').exists():
            print(f"Creating plots results folder: {self.main_path.joinpath('res').joinpath('plots')}")
            self.main_path.joinpath('res').joinpath('plots').mkdir(parents=True, exist_ok=True)

        self.get_fluid_data()
        self.calc_data()
        
    def get_fluid_data(self):

        if self.fluid_name == "Water":

            self.PTRIP,self.TTRIP = 611.66,273.16
            self.PCRIT,self.TCRIT = 22064000,647.096
        elif self.fluid_name == "CO2":

            self.PTRIP,self.TTRIP = 517964,216.592
            self.PCRIT,self.TCRIT = 7377297,304.128

        elif self.fluid_name == "Propane":
            self.PTRIP,self.TTRIP = 0.00017185,85.525
            self.PCRIT,self.TCRIT = 4251165,369.89
        else: print("current fluid not supported")

    def plot_isolines(self, out,prop1, values1,prop2, values2,color,unit,levels, show=True):
        X,Y,Z = trend.calc_Property_Array_2D(out,prop1,values1,prop2,values2,self.fluid_name)
        print(f"Plotting isolines for {out} with {prop1} and {prop2}...")
        print(Y)
        match prop1:
            case "T": X -= 273.15
            #case "P": X *= 1e-5
            case "H": X /= 1000
            case "S": X /= 1000

        match prop2:
            case "T": Y -= 273.15
            #case "P": Y *= 1e-5
            case "H": Y /= 1000
            case "S": Y /= 1000

        match out:
            case "T": Z -= 273.15
            #case "P": Z *= 1e-5
            case "H": Z /= 1000
            case "S": Z /= 1000


        z = plt.contour(X,Y,Z,colors=color,levels=levels)
        plt.clabel(z,z.levels, inline=True, fontsize=8, fmt='%1.1f '+unit, colors=color, zorder=5)


    def calc_data(self):

        #props = ["T","H","S","D","CP","CV","WS","U","ST","ETA","TCX","JTCO"]
        props = ["P","T","H","S","D","CP","CV","WS","U","ST","ETA","TCX","JTCO"]

        all_files_in_dir = [f.name for f in self.save_path.joinpath('data').iterdir() if f.is_file()]
        vle_files = [f for f in all_files_in_dir if f.startswith(f"vle_arrays_{self.fluid_name}")and f.endswith(".pkl")]
        sve_files = [f for f in all_files_in_dir if f.startswith(f"sve_arrays_{self.fluid_name}")and f.endswith(".pkl")]
        sle_files = [f for f in all_files_in_dir if f.startswith(f"sle_arrays_{self.fluid_name}")and f.endswith(".pkl")]
        
        load_vle_N = max([int(f.split(".")[0].split("_")[-1]) for f in vle_files]) if len(vle_files) > 0 else 0
        load_sve_N = max([int(f.split(".")[0].split("_")[-1]) for f in sve_files]) if len(sve_files) > 0 else 0
        load_sle_N = max([int(f.split(".")[0].split("_")[-1]) for f in sle_files]) if len(sle_files) > 0 else 0

        bool_calc_vle = False if load_vle_N >= self.N else True
        bool_calc_sve = False if load_sve_N >= self.N else True
        bool_calc_sle = False if load_sle_N >= self.N else True

        # CALC VLE
        if not bool_calc_vle:
            self.vle_arrays = pd.read_pickle(self.save_path.joinpath('data').joinpath(f"vle_arrays_{self.fluid_name}_{load_vle_N}.pkl"))
            self.liq_arrays_vle = pd.DataFrame.from_dict({prop : self.vle_arrays[prop+"''"] for prop in props})
            self.vap_arrays_vle = pd.DataFrame.from_dict({prop : self.vle_arrays[prop+"'"] for prop in props})
        else:
            self.liq_arrays_vle,self.vap_arrays_vle = self.get_VLE_dome(props)
            self.vle_arrays = pd.DataFrame()
            for prop in props:
                self.vle_arrays[prop+"''"] = self.liq_arrays_vle[prop] 
                self.vle_arrays[prop+"'"] = self.vap_arrays_vle[prop]
            
            if len(vle_files) > 0:
                for filename in vle_files:
                    print(f"Removing old files")
                    os.remove(self.save_path.joinpath('data').joinpath(filename))
                    os.remove(self.save_path.joinpath('data').joinpath(filename.split(".")[0]+".xlsx"))
                    os.remove(self.save_path.joinpath('data').joinpath(filename.split(".")[0]+".csv"))

            self.vle_arrays.to_csv(self.save_path.joinpath('data').joinpath(f"vle_arrays_{self.fluid_name}_{self.N}.csv"), index=False)
            self.vle_arrays.to_pickle(self.save_path.joinpath('data').joinpath(f"vle_arrays_{self.fluid_name}_{self.N}.pkl"))
            self.vle_arrays.to_excel(self.save_path.joinpath('data').joinpath(f"vle_arrays_{self.fluid_name}_{self.N}.xlsx"), index=False)

            

        #CALC SVE
        if not bool_calc_sve:
            self.sve_arrays = pd.read_pickle(self.save_path.joinpath('data').joinpath(f"sve_arrays_{self.fluid_name}_{load_sve_N}.pkl"))
            self.sol_arrays_sve = pd.DataFrame.from_dict({prop : self.sve_arrays[prop+"''"] for prop in props})
            self.vap_arrays_sve = pd.DataFrame.from_dict({prop : self.sve_arrays[prop+"'"] for prop in props})
        else:
            self.sol_arrays_sve,self.vap_arrays_sve = self.get_SVE_dome(props)
            self.sve_arrays = pd.DataFrame()
            for prop in props:
                self.sve_arrays[prop+"''"] = self.sol_arrays_sve[prop] 
                self.sve_arrays[prop+"'"] = self.vap_arrays_sve[prop]

            

            if len(sve_files) > 0:
                for filename in sve_files:
                    print(f"Removing old file: {self.save_path.joinpath('data').joinpath(filename)}")
                    os.remove(self.save_path.joinpath('data').joinpath(filename))
                    os.remove(self.save_path.joinpath('data').joinpath(filename.split(".")[0]+".xlsx"))
                    os.remove(self.save_path.joinpath('data').joinpath(filename.split(".")[0]+".csv"))

            self.sve_arrays.to_csv(self.save_path.joinpath('data').joinpath(f"sve_arrays_{self.fluid_name}_{self.N}.csv"), index=False)
            self.sve_arrays.to_pickle(self.save_path.joinpath('data').joinpath(f"sve_arrays_{self.fluid_name}_{self.N}.pkl"))
            self.sve_arrays.to_excel(self.save_path.joinpath('data').joinpath(f"sve_arrays_{self.fluid_name}_{self.N}.xlsx"), index=False)

        #CALC SLE
        if not bool_calc_sle:
            self.sle_arrays = pd.read_pickle(self.save_path.joinpath('data').joinpath(f"sle_arrays_{self.fluid_name}_{load_sle_N}.pkl"))
            self.sol_arrays_sle = pd.DataFrame.from_dict({prop : self.sle_arrays[prop+"''"] for prop in props})
            self.liq_arrays_sle = pd.DataFrame.from_dict({prop : self.sle_arrays[prop+"'"] for prop in props})
        else:
            self.sol_arrays_sle,self.liq_arrays_sle = self.get_SLE_dome(props)
            self.sle_arrays = pd.DataFrame()

            for prop in props:
                self.sle_arrays[prop+"''"] = self.sol_arrays_sle[prop] 
                self.sle_arrays[prop+"'"] = self.liq_arrays_sle[prop]

            if len(sle_files) > 0:
                for filename in sle_files:
                    print(f"Removing old file: {self.save_path.joinpath('data').joinpath(filename)}")
                    os.remove(self.save_path.joinpath('data').joinpath(filename))
                    os.remove(self.save_path.joinpath('data').joinpath(filename.split(".")[0]+".xlsx"))
                    os.remove(self.save_path.joinpath('data').joinpath(filename.split(".")[0]+".csv"))

            self.sle_arrays.to_csv(self.save_path.joinpath('data').joinpath(f"sle_arrays_{self.fluid_name}_{self.N}.csv"), index=False)
            self.sle_arrays.to_pickle(self.save_path.joinpath('data').joinpath(f"sle_arrays_{self.fluid_name}_{self.N}.pkl"))
            self.sle_arrays.to_excel(self.save_path.joinpath('data').joinpath(f"sle_arrays_{self.fluid_name}_{self.N}.xlsx"), index=False)
    
    def _get_prop_vle_vap(self, prop):
        p_data_vle = np.geomspace(self.PTRIP*1.01, self.PCRIT*0.99999, self.N)
        return trend.calc_Property_Array(prop, "PVAP", p_data_vle.copy(), "", np.ones_like(p_data_vle), self.fluid_name, use_tqdm=False)

    def _get_prop_vle_liq(self, prop):
        p_data_vle = np.geomspace(self.PTRIP*1.01, self.PCRIT*0.99999, self.N)
        return trend.calc_Property_Array(prop, "PLIQ", p_data_vle.copy(), "", np.ones_like(p_data_vle), self.fluid_name, use_tqdm=False)
    
    def _get_prop_sve_vap(self, prop):
        p_data_sve = np.geomspace(P_MIN, self.PTRIP*0.99999, self.N)
        return trend.calc_Property_Array(prop, "PSUBV+", p_data_sve.copy(), "", np.ones_like(p_data_sve), self.fluid_name, use_tqdm=False)
    
    def _get_prop_sve_sol(self, prop):
        p_data_sve = np.geomspace(P_MIN, self.PTRIP*0.99999, self.N)
        return trend.calc_Property_Array(prop, "PSUBS+", p_data_sve.copy(), "", np.ones_like(p_data_sve), self.fluid_name, use_tqdm=False)
    
    def _get_prop_sle_liq(self, prop):
        p_data_sle = np.geomspace(self.PTRIP*1.01,P_MAX, self.N)
        return trend.calc_Property_Array(prop, "PMLTL+", p_data_sle.copy(), "", np.ones_like(p_data_sle), self.fluid_name, use_tqdm=False)
    
    def _get_prop_sle_sol(self, prop):
        p_data_sle = np.geomspace(self.PTRIP*1.01,P_MAX, self.N)
        return trend.calc_Property_Array(prop, "PMLTS+", p_data_sle.copy(), "", np.ones_like(p_data_sle), self.fluid_name, use_tqdm=False)

    def get_VLE_dome(self,props):

        p_data_vle = np.geomspace(self.PTRIP*1.01, self.PCRIT*0.99999, self.N)

        list_of_properties = props.copy()
        list_of_properties.remove("P")  # P is added later

        print(f"Calculating VLE dome Vapour for {self.fluid_name} with {self.N} points...")
        data_vle_vap = {prop : np.array(i) for i,prop in zip(progress_map(self._get_prop_vle_vap, list_of_properties,n_cpu=self.n_cpu),list_of_properties)}
        print(f"Calculating VLE dome Liquid for {self.fluid_name} with {self.N} points...")
        data_vle_liq = {prop : np.array(i) for i,prop in zip(progress_map(self._get_prop_vle_liq, list_of_properties,n_cpu=self.n_cpu),list_of_properties)}

        data_vle_vap["P"] = p_data_vle.copy()
        data_vle_liq["P"] = p_data_vle.copy()
          
        return pd.DataFrame.from_dict(data_vle_liq), pd.DataFrame.from_dict(data_vle_vap)
    
    def get_SVE_dome(self,props):

        p_data_sve = np.geomspace(P_MIN, self.PTRIP*0.99999, self.N)

        list_of_properties = props.copy()
        list_of_properties.remove("P")  # P is added later


        print(f"Calculating SVE dome Vapour for {self.fluid_name} with {self.N} points...")
        data_sve_vap = {prop : np.array(i) for i,prop in zip(progress_map(self._get_prop_sve_vap, list_of_properties,n_cpu=self.n_cpu),list_of_properties)}
        print(f"Calculating SVE dome Solid for {self.fluid_name} with {self.N} points...")
        data_sve_sol = {prop : np.array(i) for i,prop in zip(progress_map(self._get_prop_sve_sol, list_of_properties,n_cpu=self.n_cpu),list_of_properties)}

        data_sve_vap["P"] = p_data_sve.copy()
        data_sve_sol["P"] = p_data_sve.copy()
          
        return pd.DataFrame.from_dict(data_sve_sol), pd.DataFrame.from_dict(data_sve_vap)

    def get_SLE_dome(self,props):

        p_data_sle = np.geomspace(self.PTRIP*1.01,P_MAX, self.N)

        list_of_properties = props.copy()
        list_of_properties.remove("P")  # P is added later


        print(f"Calculating SLE dome Liquid for {self.fluid_name} with {self.N} points...")
        data_sle_liq = {prop : np.array(i) for i,prop in zip(progress_map(self._get_prop_sle_liq, list_of_properties,n_cpu=self.n_cpu),list_of_properties)}
        print(f"Calculating SLE dome Solid for {self.fluid_name} with {self.N} points...")
        data_sle_sol = {prop : np.array(i) for i,prop in zip(progress_map(self._get_prop_sle_sol, list_of_properties,n_cpu=self.n_cpu),list_of_properties)}

        data_sle_liq["P"] = p_data_sle.copy()
        data_sle_sol["P"] = p_data_sle.copy()
          
        return pd.DataFrame.from_dict(data_sle_sol), pd.DataFrame.from_dict(data_sle_liq)

    def plot_phase_diagram(self,show = True):

        plt.scatter(self.TTRIP-273.15, self.PTRIP*1e-5, label='Triple Point', color='red',zorder=10)
        plt.scatter(self.TCRIT-273.15, self.PCRIT*1e-5, label='Critical Point', color='green',zorder=10)
        
        plt.semilogy(self.vap_arrays_vle["T"]-273.15,self.vap_arrays_vle["P"]*1e-5, label='Condensation', color='black') # p : MPa in bar
        plt.semilogy(self.liq_arrays_vle["T"]-273.15,self.liq_arrays_vle["P"]*1e-5, label='Evaporation', color='black')

        plt.semilogy(self.vap_arrays_sve["T"]-273.15,self.vap_arrays_sve["P"]*1e-5, label='Resublimation', color='black')
        plt.semilogy(self.sol_arrays_sve["T"]-273.15,self.sol_arrays_sve["P"]*1e-5, label='Sublimation', color='black')

        plt.semilogy(self.sol_arrays_sle["T"]-273.15,self.sol_arrays_sle["P"]*1e-5, label='Melting', color='black')
        plt.semilogy(self.liq_arrays_sle["T"]-273.15,self.liq_arrays_sle["P"]*1e-5, label='Freezing', color='black')

        plt.text(self.TTRIP-273.15, self.PTRIP*1e-5, '  Triple Point S-L-V', color='red', fontsize=10, ha='left', va='top')
        plt.text(self.TCRIT-273.15, self.PCRIT*1e-5, '  Critical Point', color='green', fontsize=10, ha='left', va='top')

        T_MIN_PLOT = min([i for i in np.append(self.vap_arrays_vle["T"],[self.liq_arrays_vle["T"],self.vap_arrays_sve["T"],self.sol_arrays_sve["T"],self.liq_arrays_sle["T"],self.sol_arrays_sle["T"]]) if i > 0])
        T_MAX_PLOT = max([i for i in np.append(self.vap_arrays_vle["T"],[self.liq_arrays_vle["T"],self.vap_arrays_sve["T"],self.sol_arrays_sve["T"],self.liq_arrays_sle["T"],self.sol_arrays_sle["T"]]) if i > 0])

        P_MIN_PLOT = min([i for i in np.append(self.vap_arrays_vle["P"],[self.liq_arrays_vle["P"],self.vap_arrays_sve["P"],self.sol_arrays_sve["P"],self.liq_arrays_sle["P"],self.sol_arrays_sle["P"]]) if i > 0])
        P_MAX_PLOT = max([i for i in np.append(self.vap_arrays_vle["P"],[self.liq_arrays_vle["P"],self.vap_arrays_sve["P"],self.sol_arrays_sve["P"],self.liq_arrays_sle["P"],self.sol_arrays_sle["P"]]) if i > 0])

        #self.plot_isolines("H", "T", np.linspace(T_MIN_PLOT, T_MAX_PLOT, 10), "P", np.geomspace(P_MIN_PLOT,P_MAX_PLOT,10), color='blue', unit='kJ/kg', show=False)
        
        plt.semilogy([(T_MIN_PLOT - 0.05 * (T_MAX_PLOT-T_MIN_PLOT))-273.15, (T_MAX_PLOT + 0.5 * (T_MAX_PLOT-T_MIN_PLOT))-273.15], [p_AMB*1e-5,p_AMB*1e-5], color='grey', linestyle='--', label='Ambient Pressure', zorder=5)
        plt.semilogy([T_AMB-273.15,T_AMB-273.15],[0,P_MAX*1e-5], color='grey', linestyle='--', label='Ambient Pressure', zorder=5)

        plt.xlabel('Temperature (°C)')
        plt.xlim((T_MIN_PLOT - 0.05 * (T_MAX_PLOT-T_MIN_PLOT))-273.15,(T_MAX_PLOT + 0.5 * (T_MAX_PLOT-T_MIN_PLOT))-273.15)
        plt.ylabel('Pressure (bar)')
        plt.yticks([1e-8,1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],["1mPa","10 mPa", "100 mPa", "1 Pa", "10 Pa", "1 mbar", "10 mbar", "100 mbar", "1 bar", "10 bar", "100 bar", "1000 bar"])
        plt.title(f'Phase Diagram for {self.fluid_name}')
        plt.grid()
        plt.tight_layout()
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"phase_diagram_{self.fluid_name}.png"), dpi=600)
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"phase_diagram_{self.fluid_name}.pdf"))
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"phase_diagram_{self.fluid_name}.svg"),transparent=True)

        if show : plt.show()
        plt.close()

    def plot_p_h(self,show = True):

        plt.semilogy(self.vap_arrays_vle["H"]*1e-3,self.vap_arrays_vle["P"]*1e-5, color='black') # p : MPa in bar
        plt.semilogy(self.liq_arrays_vle["H"]*1e-3,self.liq_arrays_vle["P"]*1e-5, color='black')

        plt.semilogy(self.vap_arrays_sve["H"]*1e-3,self.vap_arrays_sve["P"]*1e-5, color='black')
        plt.semilogy(self.sol_arrays_sve["H"]*1e-3,self.sol_arrays_sve["P"]*1e-5, color='black')

        plt.semilogy(self.sol_arrays_sle["H"]*1e-3,self.sol_arrays_sle["P"]*1e-5, color='black')
        plt.semilogy(self.liq_arrays_sle["H"]*1e-3,self.liq_arrays_sle["P"]*1e-5, color='black')

        P_CRIT_PLOT = self.PCRIT*1e-5
        H_CRIT_PLOT = trend.calc_Property("H", "P", self.PCRIT, "T", self.TCRIT, self.fluid_name)/1e3
        plt.scatter(H_CRIT_PLOT, P_CRIT_PLOT, label='Critical Point', color='green',zorder=10)
        
        P_TRIP_PLOT = self.PTRIP*1e-5
        H_TRIP_PLOT_MIN = trend.calc_Property("H", "PSUBS+", self.PTRIP, "", 1, self.fluid_name)/1e3
        H_TRIP_PLOT_MAX = trend.calc_Property("H", "P", self.PTRIP*1.005, "Q", 1, self.fluid_name)/1e3

        plt.semilogy([H_TRIP_PLOT_MIN, H_TRIP_PLOT_MAX], [P_TRIP_PLOT, P_TRIP_PLOT], label='Triple Point Line', color='black',linestyle="dashed",zorder=10)

        plt.xlabel('Enthalpy (kJ/kg)')
        plt.ylabel('Pressure (bar)')
        plt.yticks([1e-8,1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],["1mPa","10 mPa", "100 mPa", "1 Pa", "10 Pa", "1 mbar", "10 mbar", "100 mbar", "1 bar", "10 bar", "100 bar", "1000 bar"])
        plt.title(f'log(p)-h-Diagram {self.fluid_name}')
        plt.grid()
        plt.tight_layout()
        plt.legend()
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"log_p_h_{self.fluid_name}.png"), dpi=600)
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"log_p_h_{self.fluid_name}.pdf"))
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"log_p_h_{self.fluid_name}.svg"),transparent=True)
        if show : plt.show()
        plt.close()

    def plot_T_s(self,show = True):

        plt.plot(self.vap_arrays_vle["S"]*1e-3,self.vap_arrays_vle["T"]-273.15, label='Condensation', color='black') # p : MPa in bar
        plt.plot(self.liq_arrays_vle["S"]*1e-3,self.liq_arrays_vle["T"]-273.15, label='Evaporation', color='black')

        plt.plot(self.vap_arrays_sve["S"]*1e-3,self.vap_arrays_sve["T"]-273.15, label='Resublimation', color='black')
        plt.plot(self.sol_arrays_sve["S"]*1e-3,self.sol_arrays_sve["T"]-273.15, label='Sublimation', color='black')

        plt.plot(self.sol_arrays_sle["S"]*1e-3,self.sol_arrays_sle["T"]-273.15, label='Melting', color='black')
        plt.plot(self.liq_arrays_sle["S"]*1e-3,self.liq_arrays_sle["T"]-273.15, label='Freezing', color='black')

        T_CRIT_PLOT = self.TCRIT-273.15
        S_CRIT_PLOT = trend.calc_Property("S", "P", self.PCRIT, "T", self.TCRIT, self.fluid_name)/1e3
        plt.scatter(S_CRIT_PLOT, T_CRIT_PLOT, label='Critical Point', color='green',zorder=10)

        T_TRIP_PLOT = self.TTRIP-273.15
        S_TRIP_PLOT_MIN = trend.calc_Property("S", "PSUBS+", self.PTRIP, "", 1, self.fluid_name)/1e3
        S_TRIP_PLOT_MAX = trend.calc_Property("S", "P", self.PTRIP*1.005, "Q", 1, self.fluid_name)/1e3

        plt.plot([S_TRIP_PLOT_MIN, S_TRIP_PLOT_MAX], [T_TRIP_PLOT, T_TRIP_PLOT], label='Triple Point Line', color='black',linestyle="dashed",zorder=10)

        plt.xlabel('Entropie (kJ/kgK)')
        #plt.xlim((T_MIN - 0.05 * (T_MAX-T_MIN))-273.15,(T_MAX + 0.5 * (T_MAX-T_MIN))-273.15)
        plt.ylabel('Temperature (°C)')
        #plt.yticks([1e-8,1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],["1mPa","10 mPa", "100 mPa", "1 Pa", "10 Pa", "1 mbar", "10 mbar", "100 mbar", "1 bar", "10 bar", "100 bar", "1000 bar"])
        plt.title(f'T-s-Diagram for {self.fluid_name}')
        plt.grid()
        plt.tight_layout()
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"T_s_{self.fluid_name}.png"), dpi=600)
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"T_s_{self.fluid_name}.pdf"))
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"T_s_{self.fluid_name}.svg"),transparent=True)
        if show : plt.show()
        plt.close()

    def plot_T_h(self,show = True):

        plt.plot(self.vap_arrays_vle["H"]*1e-3,self.vap_arrays_vle["T"]-273.15, label='Condensation', color='black') # p : MPa in bar
        plt.plot(self.liq_arrays_vle["H"]*1e-3,self.liq_arrays_vle["T"]-273.15, label='Evaporation', color='black')

        plt.plot(self.vap_arrays_sve["H"]*1e-3,self.vap_arrays_sve["T"]-273.15, label='Resublimation', color='black')
        plt.plot(self.sol_arrays_sve["H"]*1e-3,self.sol_arrays_sve["T"]-273.15, label='Sublimation', color='black')

        plt.plot(self.sol_arrays_sle["H"]*1e-3,self.sol_arrays_sle["T"]-273.15, label='Melting', color='black')
        plt.plot(self.liq_arrays_sle["H"]*1e-3,self.liq_arrays_sle["T"]-273.15, label='Freezing', color='black')

        T_CRIT_PLOT = self.TCRIT-273.15
        H_CRIT_PLOT = trend.calc_Property("H", "P", self.PCRIT, "T", self.TCRIT, self.fluid_name)/1e3
        plt.scatter(H_CRIT_PLOT, T_CRIT_PLOT, label='Critical Point', color='green',zorder=10)

        T_TRIP_PLOT = self.TTRIP-273.15
        H_TRIP_PLOT_MIN = trend.calc_Property("H", "PSUBS+", self.PTRIP, "", 1, self.fluid_name)/1e3
        H_TRIP_PLOT_MAX = trend.calc_Property("H", "P", self.PTRIP*1.005, "Q", 1, self.fluid_name)/1e3

        plt.plot([H_TRIP_PLOT_MIN, H_TRIP_PLOT_MAX], [T_TRIP_PLOT, T_TRIP_PLOT], label='Triple Point Line', color='black',linestyle="dashed",zorder=10)

        plt.xlabel('Enthalpy (kJ/kg)')
        #plt.xlim((T_MIN - 0.05 * (T_MAX-T_MIN))-273.15,(T_MAX + 0.5 * (T_MAX-T_MIN))-273.15)
        plt.ylabel('Temperature (°C)')
        #plt.yticks([1e-8,1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],["1mPa","10 mPa", "100 mPa", "1 Pa", "10 Pa", "1 mbar", "10 mbar", "100 mbar", "1 bar", "10 bar", "100 bar", "1000 bar"])
        plt.title(f'T-h-Diagram for {self.fluid_name}')
        plt.grid()
        plt.tight_layout()
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"T_h_{self.fluid_name}.png"), dpi=600)
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"T_h_{self.fluid_name}.pdf"))
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"T_h_{self.fluid_name}.svg"),transparent=True)
        if show : plt.show()
        plt.close()

    def plot_h_s(self,show = True):

        plt.plot(self.vap_arrays_vle["S"]*1e-3,self.vap_arrays_vle["H"]*1e-3, label='Condensation', color='black') # p : MPa in bar
        plt.plot(self.liq_arrays_vle["S"]*1e-3,self.liq_arrays_vle["H"]*1e-3, label='Evaporation', color='black')

        plt.plot(self.vap_arrays_sve["S"]*1e-3,self.vap_arrays_sve["H"]*1e-3, label='Resublimation', color='black')
        plt.plot(self.sol_arrays_sve["S"]*1e-3,self.sol_arrays_sve["H"]*1e-3, label='Sublimation', color='black')

        plt.plot(self.sol_arrays_sle["S"]*1e-3,self.sol_arrays_sle["H"]*1e-3, label='Melting', color='black')
        plt.plot(self.liq_arrays_sle["S"]*1e-3,self.liq_arrays_sle["H"]*1e-3, label='Freezing', color='black')

        H_CRIT_PLOT = trend.calc_Property("H", "P", self.PCRIT, "T", self.TCRIT, self.fluid_name)/1e3
        S_CRIT_PLOT = trend.calc_Property("S", "P", self.PCRIT, "T", self.TCRIT, self.fluid_name)/1e3
        plt.scatter(S_CRIT_PLOT, H_CRIT_PLOT, label='Critical Point', color='green',zorder=10)

        T_TRIP_PLOT = self.TTRIP-273.15
        H_TRIP_PLOT_MIN = trend.calc_Property("H", "PSUBS+", self.PTRIP, "", 1, self.fluid_name)/1e3
        H_TRIP_PLOT_MAX = trend.calc_Property("H", "P", self.PTRIP*1.005, "Q", 1, self.fluid_name)/1e3
        S_TRIP_PLOT_MIN = trend.calc_Property("S", "PSUBS+", self.PTRIP, "", 1, self.fluid_name)/1e3
        S_TRIP_PLOT_MAX = trend.calc_Property("S", "P", self.PTRIP*1.005, "Q", 1, self.fluid_name)/1e3

        #s_array_trip = np.linspace(S_TRIP_PLOT_MIN, S_TRIP_PLOT_MAX, self.N)
        #h_array_trip = trend.calc_Property_Array("S", "P", [self.PTRIP for i in s_array_trip] , "S", s_array_trip, self.fluid_name, use_tqdm=False)/1e3

        #plt.plot([S_TRIP_PLOT_MIN, S_TRIP_PLOT_MAX], h_array_trip, label='Triple Point Line', color='black',linestyle="dashed",zorder=10)

        plt.xlabel('Entropy (kJ/kgK)')
        #plt.xlim((T_MIN - 0.05 * (T_MAX-T_MIN))-273.15,(T_MAX + 0.5 * (T_MAX-T_MIN))-273.15)
        plt.ylabel('Enthalpy (kJ/kg)')
        #plt.yticks([1e-8,1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],["1mPa","10 mPa", "100 mPa", "1 Pa", "10 Pa", "1 mbar", "10 mbar", "100 mbar", "1 bar", "10 bar", "100 bar", "1000 bar"])
        plt.title(f'h-s-Diagram for {self.fluid_name}')
        plt.grid()
        plt.tight_layout()
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"h_s_{self.fluid_name}.png"), dpi=600)
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"h_s_{self.fluid_name}.pdf"))
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"h_s_{self.fluid_name}.svg"),transparent=True)
        if show : plt.show()
        plt.close()

    def plot_p_s(self,show = True):

        plt.semilogy(self.vap_arrays_vle["S"]*1e-3,self.vap_arrays_vle["P"]*1e-5, label='Condensation', color='black') # p : MPa in bar
        plt.semilogy(self.liq_arrays_vle["S"]*1e-3,self.liq_arrays_vle["P"]*1e-5, label='Evaporation', color='black')

        plt.semilogy(self.vap_arrays_sve["S"]*1e-3,self.vap_arrays_sve["P"]*1e-5, label='Resublimation', color='black')
        plt.semilogy(self.sol_arrays_sve["S"]*1e-3,self.sol_arrays_sve["P"]*1e-5, label='Sublimation', color='black')

        plt.semilogy(self.sol_arrays_sle["S"]*1e-3,self.sol_arrays_sle["P"]*1e-5, label='Melting', color='black')
        plt.semilogy(self.liq_arrays_sle["S"]*1e-3,self.liq_arrays_sle["P"]*1e-5, label='Freezing', color='black')

        P_CRIT_PLOT = self.PCRIT*1e-5
        S_CRIT_PLOT = trend.calc_Property("S", "P", self.PCRIT, "T", self.TCRIT, self.fluid_name)/1e3
        plt.scatter(S_CRIT_PLOT, P_CRIT_PLOT, label='Critical Point', color='green',zorder=10)

        P_TRIP_PLOT = self.PTRIP*1e-5
        S_TRIP_PLOT_MIN = trend.calc_Property("S", "PSUBS+", self.PTRIP, "", 1, self.fluid_name)/1e3
        S_TRIP_PLOT_MAX = trend.calc_Property("S", "P", self.PTRIP*1.005, "Q", 1, self.fluid_name)/1e3

        plt.plot([S_TRIP_PLOT_MIN, S_TRIP_PLOT_MAX], [P_TRIP_PLOT, P_TRIP_PLOT], label='Triple Point Line', color='black',linestyle="dashed",zorder=10)

        plt.xlabel('Entropy (kJ/kgK)')
        #plt.xlim((T_MIN - 0.05 * (T_MAX-T_MIN))-273.15,(T_MAX + 0.5 * (T_MAX-T_MIN))-273.15)
        plt.ylabel('Pressure')
        plt.yticks([1e-8,1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],["1mPa","10 mPa", "100 mPa", "1 Pa", "10 Pa", "1 mbar", "10 mbar", "100 mbar", "1 bar", "10 bar", "100 bar", "1000 bar"])
        plt.title(f'p-s-Diagram for {self.fluid_name}')
        plt.grid()
        plt.tight_layout()
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"log_p_s_{self.fluid_name}.png"), dpi=600)
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"log_p_s_{self.fluid_name}.pdf"))
        plt.savefig(self.save_path.joinpath('plots').joinpath(f"log_p_s_{self.fluid_name}.svg"),transparent=True)
        if show : plt.show()
        plt.close()

if __name__ == "__main__":
    print("This module is not meant to be run directly. Please import it in your script.")
    PD = PropertyDiagrams("Water",N=50000)
    PD.plot_phase_diagram(show=True)
    PD.plot_p_h()
    PD.plot_T_s()
    PD.plot_T_h()
    PD.plot_h_s()
    PD.plot_p_s()
