"""
GUI for the simulator
Author: Davide Tomasella
"""
# In[]
import os
from qtpy import QtCore
from qtpy import QtWidgets
from qtpy import uic
from qtpy.QtCore import Slot
from qtpy.QtCore import QObject
from qtpy.QtGui import QPainter
from simulator_UI_autogen import Ui_Simulator
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter, CSVExporter
import numpy as np
import re
import json
from cavity_solver import BaseCavitySolver
from analytical_cavity_solver import AnalyticalCavitySolver

_use_autogen = False
from simulator_UI_autogen import Ui_Simulator

class Simulator_MainWindow_Autogen(QtWidgets.QMainWindow,Ui_Simulator):
    """
    The main window for the cavity control GUI. 
    It extends the standard QMainWindow, but it also inherit the method setupUi() 
    form the autogen file we can create with pyuic5.
    
    To generate the gui from the .ui file:
     - move to the directory where the .ui file is located
     - execute the command: pyuic5 simulator_UI.ui -o simulator_UI_autogen.py
    """

    def __init__(self):
        # Init autogen class
        super(Simulator_MainWindow_Autogen,self).__init__()
        # Load the UI
        self.setupUi(self)

class Simulator_MainWindow(QtWidgets.QMainWindow):
    """
    The main window for the cavity control GUI. 
    It extends the standard QMainWindow
    """

    def __init__(self):
        super().__init__()
        # Load the UI
        uic.loadUi('./simulator_UI.ui', self)


class Simulator(QObject):
    """
    This is the GUI Class for creating the grafical interface
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self._mw = MainWindow()

        self._cavitySolver : BaseCavitySolver = None
        self._config_last_run : dict = None
        self._mw = None
    
    def on_activate(self):
        """ 
        Activate the GUI and show the main window that is part of the Simulator class.
        We configure the dynamical elements, the event handlers, and the pyqtgraph library.
        """
        # setting up main window
        if _use_autogen:
            self._mw = Simulator_MainWindow_Autogen()
        else:
            self._mw = Simulator_MainWindow()

        self.init_GUI_elements()
        self.init_plot()

        # show window
        self.show()

    def on_deactivate(self):
        """
        Reverse steps of activation

        Returns:
            error code (int): 0 if OK, -1 if error
        """
        self._mw.close()
        return

    def show(self):
        """ Show the ui """
        self._mw.show()

    def init_GUI_elements(self):
        """ Configure the GUI window with dynamical elements and connect the signals to the slots"""

        # NOTE: GUI input settings (min, max, default, ...) are set in the .ui file
        # Here we define the connectors and we initialize the "conditional" elements

        # Connect signals to slots
        # file and folder selection
        self._mw.F_folder_button.clicked.connect(self.onClk_browse_folders)
        self._mw.F_folder_button.setEnabled(1)
        self._mw.F_file_button.clicked.connect(self.onClk_browse_files)
        self._mw.F_file_button.setEnabled(1)
        self._mw.F_file_name.editingFinished.connect(self.onChanged_file_name)
        # save and start buttons
        self._mw.RUN_save_button.clicked.connect(self.onClk_save_data)
        self._mw.RUN_save_button.setEnabled(1)
        self._mw.RUN_save_button.setShortcut("Ctrl+S")
        self._mw.RUN_start_button.clicked.connect(self.onClk_start_simulation)
        self._mw.RUN_start_button.setEnabled(1)
        self._mw.RUN_start_button.setShortcut("Ctrl+R")
        # select sliders for the type of simulation and the sideband
        self._mw.T_sideband_select.valueChanged.connect(self.onChange_sideband_select)
        self._mw.T_model_select.valueChanged.connect(self.onChange_model_select)
        # checkboxes for the configuration (enable/diable possible combinations)
        self._mw.C_use_optical.stateChanged.connect(self.onChange_use_optical_mode)
        self._mw.C_use_mech.stateChanged.connect(self.onChange_use_mechanical_mode)
        self._mw.C_use_bath.stateChanged.connect(self.onChange_use_quantum_bath_temperature)
        self._mw.C_use_time_evolution.stateChanged.connect(self.onChange_use_quantum_time_evolution)
        self._mw.C_scan_D.stateChanged.connect(self.onChange_scan_optical_FSR_detuning)
        self._mw.C_scan_P.stateChanged.connect(self.onChange_scan_optical_pump_power)

        # Set the initial configuration
        self._mw.F_folder_name.setText(os.path.join(os.getcwd(),"data"))
        self._mw.F_file_name.setText("Simulation")
        self._mw.C_use_optical.setChecked(True)
        self._mw.C_use_mech.setChecked(True)
        self._mw.RUN_save_button.setEnabled(False)
        
        return

    def init_plot(self):
        """Configuration of the gui module corresponding to the plot window"""
        #self._plot_curve : pg.GraphicsObject = pg.PlotDataItem() # TODO maybe different elements for 3d...
        #self._mw.plotwindow.addItem(self._plot_curve)
        self._mw.plotwindow.setRenderHints(QPainter.Antialiasing)
        self.update_plot()
    

    @Slot()
    def onChange_sideband_select(self):
        """ Update the graphical elements when we are measuring the optomechanical response """
        if self._mw.C_use_mech.isChecked():
            if self._mw.T_sideband_select.value() == 1: # AntiStokes
                self._mw.meas_sideband_marker1.setStyleSheet("background-color: rgb(0, 85, 255)")
                self._mw.meas_sideband_marker2.setStyleSheet("background-color: rgb(0, 85, 255)")
            else: # Stokes
                self._mw.meas_sideband_marker1.setStyleSheet("background-color: rgb(170, 0, 0)")
                self._mw.meas_sideband_marker2.setStyleSheet("background-color: rgb(170, 0, 0)")
            print('Changed measurement sideband for optomechanical response')
        else:
            self._mw.meas_sideband_marker1.setStyleSheet("background-color: #c1c1c1")
            self._mw.meas_sideband_marker2.setStyleSheet("background-color: #c1c1c1")

    @Slot()
    def onChange_model_select(self):
        """ Update the graphical elements when we are measuring the optomechanical response """
        isQuantum = self._mw.T_model_select.value() == 1 #quantum
        # we can use the bath only if both quantum simulation and mechanical mode are enabled
        self._mw.C_use_bath.setEnabled(isQuantum and self._mw.C_use_mech.isChecked())
        self._mw.C_use_time_evolution.setEnabled(isQuantum)
        self._mw.C_scan_D.setEnabled(not isQuantum)
        self._mw.C_scan_P.setEnabled(not isQuantum)
        print('Changed model for the simulation')
    
    @Slot()
    def onChange_use_optical_mode(self):
        """ Update the graphical elements when we are measuring the optical response """
        # We don't support not using the optical mode, so this is "readOnly".
        self._mw.C_use_optical.setChecked(True)
    
    @Slot()
    def onChange_use_mechanical_mode(self):
        """ Update the graphical elements when we are measuring the optomechanical response """
        self.onChange_sideband_select()
        self.onChange_model_select()
    
    @Slot()
    def onChange_use_quantum_bath_temperature(self):
        """ Update the graphical elements when we are setting the bath temperature """
        if self._mw.C_use_bath.isChecked() and self._mw.T_model_select.value() == 0:
            raise ValueError('Cannot use the bath temperature in classical mode')

    @Slot()
    def onChange_use_quantum_time_evolution(self):
        """ Update the graphical elements when we are displaying the time evolution """
        self.set_plot_axis()
        if self._mw.C_use_time_evolution.isChecked() and self._mw.T_model_select.value() == 0:
            raise ValueError('Cannot use the time evolution in classical mode')
    
    @Slot()
    def onChange_scan_optical_FSR_detuning(self):
        """ Update the graphical elements when we are scanning the optical FSR detuning """
        self._mw.C_scan_P.setEnabled(not self._mw.C_scan_D.isChecked())
        if self._mw.C_scan_D.isChecked() and self._mw.T_model_select.value() == 1:
            raise ValueError('Cannot scan the detuning in quantum mode')
            
    @Slot()
    def onChange_scan_optical_pump_power(self):
        """ Update the graphical elements when we are scanning the optical pump power """
        self._mw.C_scan_D.setEnabled(not self._mw.C_scan_P.isChecked())
        if self._mw.C_scan_P.isChecked() and self._mw.T_model_select.value() == 1:
            raise ValueError('Cannot scan the pump power in quantum mode')


    @Slot()
    def onClk_browse_folders(self):
        """ Select the folder where data will be saved """
        defaultFolderName = os.path.join(os.getcwd(),"data")
        if not os.path.exists(defaultFolderName):
            os.makedirs(defaultFolderName)
        folderName = QtWidgets.QFileDialog.getExistingDirectory(self._mw, "Select saving directory", directory = defaultFolderName)
        self._mw.F_folder_name.setText(folderName)

    @Slot()
    def onClk_browse_files(self):
        """ Browse files to select a file name; it's meant to provide a faster way of choosing a name """
        defaultFolderName = self._mw.F_folder_name.text()
        fileName = QtWidgets.QFileDialog.getOpenFileName(self._mw, "Select file name", defaultFolderName)
        self._mw.F_file_name.setText(fileName[0].split("/")[-1].split(".")[0])

    @Slot()
    def onChanged_file_name(self):
        """ Update the file name when the user changes it """
        # if we have an extension, remove it
        self._mw.F_file_name.setText(self._mw.F_file_name.text().split(".")[0])
        # check if the file name is valid
        if not re.match(r'^[a-zA-Z0-9_]+$', self._mw.F_file_name.text()):
            self._mw.F_file_name.setText("Simulation")
            
    @Slot()
    def onClk_start_simulation(self):
        """ Start the simulation using the current configuration """
        self.enable_interface(False)
        print('Start simulation')
        self._cavitySolver, self._config_last_run = self.get_and_configure_solver_from_ui()
        if self._config_last_run["use_time_evolution"]:
            data = self._cavitySolver.solve_cavity_time_evolution()
        else:
            data = self._cavitySolver.solve_cavity_RT()
        self.update_plot(data, self._config_last_run["use_time_evolution"])
        self.enable_interface(True)

    def get_and_configure_solver_from_ui(self) -> tuple[BaseCavitySolver, bool]:
        """ Create the solver object and configure it according to the GUI configuration """
        config = {}
        config["use_optical_mode"] = self._mw.C_use_optical.isChecked()
        config["use_mechanical_mode"] = self._mw.C_use_mech.isChecked()
        config["use_quantum_bath"] = self._mw.C_use_bath.isChecked()
        config["use_time_evolution"] = self._mw.C_use_time_evolution.isChecked()
        config["scan_FSR_detuning"] = self._mw.C_scan_D.isChecked()
        config["scan_pump_power"] = self._mw.C_scan_P.isChecked()
        config["is_sideband_stokes"] = self._mw.T_sideband_select.value() == 0
        config["is_solver_quantum"] = self._mw.T_model_select.value() == 1

        # construct the solver
        solver : BaseCavitySolver = None
        if config["is_solver_quantum"]:
            if config["use_mechanical_mode"]:
                config["solver"] = "quantum_optomechanical_cavity"
                #solver = QuantumOptomechanicalCavitySolver()
            else:
                config["solver"] = "quantum_optical_cavity"
                #solver = QuantumOpticalCavitySolver()
        else:
            config["solver"] = "classical_cavity"
            solver = AnalyticalCavitySolver()
        # configure the solver
        solver.configure(is_optomechanical=config["use_mechanical_mode"],
                         is_sideband_stokes=config["is_sideband_stokes"],
                         #is_quantum=config["is_solver_quantum"],
                         scan_FSR_detuning=config["scan_FSR_detuning"],
                         scan_pump_power=config["scan_pump_power"],
                         #use_bath=config["use_quantum_bath"],
                         #use_time_evolution=config["use_time_evolution"],
                         omega_p=3e8/(self._mw.C_omega_p_nm.value()/1e9),
                         kappa_ext1_s=self._mw.C_kappa_ext_1_MHz.value()*1e6,
                         kappa_ext2_s=self._mw.C_kappa_ext_1_MHz.value()*1e6,
                         kappa_0_s=self._mw.C_kappa_0_MHz.value()*1e6,
                         Omega_m=self._mw.C_Omega_m_GHz.value()*1e9,
                         gamma_m=self._mw.C_gamma_MHz.value()*1e6,
                         G0=self._mw.C_G0_Hz.value(),
                         FSR_s=self._mw.C_FSR_GHz.value()*1e9,
                         detuning_s_0=self._mw.C_f_start_MHz.value()*1e6,
                         detuning_s_1=self._mw.C_f_stop_MHz.value()*1e6,
                         power_p=self._mw.C_P_mW.value()/1e3,
                         power_p_0=self._mw.C_P_start_mW.value()/1e3,
                         power_p_1=self._mw.C_P_stop_mW.value()/1e3,
                         bath_T=self._mw.C_bath_K.value())
        config["solver_params"] = solver.get_current_configuration()
        if config["use_time_evolution"]:
            self._mw.C_time_ms.setText("%.3f"%(config["solver_params"]["max_t_evolution"]*1e3))

        return solver, config
    
    def enable_interface(self, enable=True):
        """ 
        Enable or disable the interface buttons
        Parameters:
        ----------
        enable : bool
            True if the interface should be enabled, False if it should be disabled
        """
        # we disable the gui when we start a measurement so we don't try to start multiple measurements (i.e. threads)
        en=int(enable)
        self._mw.RUN_start_button.setEnabled(en)
        self._mw.RUN_save_button.setEnabled(en)
        if enable:
            pass
            #self._mw.RUN_start_button.setStyleSheet("background-color: none; color: black;")
            #self._mw.RUN_save_button.setStyleSheet("background-color: none; color: black;")
      
    def update_plot(self, data=None, is_time_evolution=False):
        """ 
        Update the plot with the current data 

        Parameters
        ----------
        data : np.ndarray
            The data to be plotted. It can be a 1D array for the reflectivity/transmissivity or a 2D array for the time evolution
        """
        # temp creation of random data
        if data is None:
            data=[np.linspace(12,13,100),np.random.rand(100),np.random.rand(100)]
            
        self.set_plot_data(data, is_time_evolution)        
        self.set_plot_axis(is_time_evolution)
        print('Update plot')

    def set_plot_data(self, data, is_time_evolution):
        """ 
        Set the data to be plotted 

        Parameters
        ----------
        data : np.ndarray
            The data to be plotted. It can be a 1D array for the reflectivity/transmissivity or a 2D array for the time evolution
        is_time_evolution : bool
            True if we are plotting the time evolution, False if we are plotting the reflectivity/transmissivity
        """
        if is_time_evolution:
            # TODO define 3d data, maybe we have to keep 2 elements (_plot_curve and ...) and work with the visibility / remove them when needed
            pass
        else:
            frequencies = data[0]
            reflectivity = data[1]
            transmissivity = data[2]
            self._mw.plotwindow.clear()
            if len(reflectivity.shape) > 1:
                cm=pg.colormap.get("viridis").getLookupTable(nPts=reflectivity.shape[0])
                for i in range(reflectivity.shape[0]):
                    self._mw.plotwindow.plot(frequencies,reflectivity[i], pen=pg.mkPen(cm[i], width=2))
                    self._mw.plotwindow.plot(frequencies,transmissivity[i], pen=pg.mkPen(cm[i], width=2, dash=[2, 4]))
            else:
                self._mw.plotwindow.plot(frequencies,reflectivity, pen=pg.mkPen('b', width=2))
                self._mw.plotwindow.plot(frequencies,transmissivity, pen=pg.mkPen('b', width=2, dash=[2, 4]))
            #self._plot_curve.setData(frequencies,reflectivity, pen='b')
            #self._plot_curve.setData(frequencies,transmissivity, pen='r')
    
    def set_plot_axis(self, is_time_evolution):
        """ Set the axis label for the plot """
        if is_time_evolution:
            # TODO define 3d axis
            self._mw.plotwindow.setLabel(axis='bottom', text='Time', units='s',pen='k')
            self._mw.plotwindow.setLabel(axis='left', text='Photon/Phonon population', units='',pen='k')
        else:
            viewBox=self._mw.plotwindow.getViewBox()
            viewBox.updateAutoRange()
            viewBox.enableAutoRange(axis='x', enable=True)
            viewBox.enableAutoRange(axis='y', enable=True)
            viewBox.setMouseEnabled(x=True, y=False)
            self._mw.plotwindow.getAxis('left').setTextPen('k')
            self._mw.plotwindow.getAxis('bottom').setTextPen('k')
            self._mw.plotwindow.setLabel(axis='bottom', text='Frequency', units='Hz',pen='k')
            self._mw.plotwindow.setLabel(axis='left', text='Reflected/Transmitted power', units='',pen='k')

    @Slot()
    def onClk_save_data(self):
        """ Save the plot to a png file and the config to a txt file"""
        filepath = self.get_unique_filepath()
        # save the configuration in a json file
        if self._config_last_run is not None:
            config = self._config_last_run
            # TODO decide if the current configuration can be changed during the run
            config["solver_params"] = self._cavitySolver.get_current_configuration()
            with open(filepath+"_cfg.json", 'w') as f:
                json.dump(config, f, indent=2, cls=self.NumpyEncoder)
        plot = self._mw.plotwindow.getPlotItem()
        exporter = CSVExporter(plot)
        exporter.export(filepath+"_data.txt")
        exporter = ImageExporter(plot)
        exporter.parameters()['width'] = 2000
        exporter.export(filepath+"_plot.png")
        # TODO for time evolution
        # glview.grabFrameBuffer().save('fileName.png')
        print('Save data')
    
    def get_unique_filepath(self, extension="_data.txt"):
        """ Get a unique file path for the data file to not overwrite existing files

        Parameters
        ----------
        extension : str
            The extension of the file to be saved specifying the type of data and the format
        
        Returns
        -------
        filePath : str
            The unique file path after checking the existing files in the folder
        """
        l_ext = len(extension)
        dir = self._mw.F_folder_name.text()
        filename = self._mw.F_file_name.text()
        # File numbering has 6 values, add them to the filename if not present

        indexes= [np.int32(name.split("_")[-2][-6:])
                  for name in os.listdir(dir) if name.startswith(filename)]
        num_file = np.max(indexes+[0]) + 1

        filePath = os.path.join(dir,filename+'{:06d}'.format(num_file))
        return filePath
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

if __name__ == "__main__":
    import platform
    import ctypes
    if platform.system()=='Windows' and int(platform.release()) >= 8:   
        ctypes.windll.shcore.SetProcessDpiAwareness(False)
    import sys
    app = QtWidgets.QApplication([])
    app.setAttribute(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    gui = Simulator()
    gui.on_activate()
    sys.exit(app.exec_())
