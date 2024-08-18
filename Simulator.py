"""
GUI for the simulator
Author: Davide Tomasella
"""
# In[]
import os
from qtpy import QtCore
from qtpy.QtCore import Slot
from qtpy.QtCore import QObject
from qtpy import QtWidgets
from simulator_UI_autogen import Ui_Simulator
from qtpy import uic
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter, CSVExporter
import numpy as np
from cavity_solver import BaseCavitySolver
import re
import json

#ifdef
_use_autogen=False
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
        # Load it
        super(Simulator_MainWindow_Autogen,self).__init__()

        self.setupUi(self)

class Simulator_MainWindow(QtWidgets.QMainWindow):
    """
    The main window for the cavity control GUI. 
    It extends the standard QMainWindow
    """

    def __init__(self):
        # Load it
        super(Simulator_MainWindow,self).__init__()
        # Load the UI
        uic.loadUi('./simulator_UI.ui', self)


class Simulator(QObject):
    """
    This is the GUI Class for creating the grafical interface
    """

    # declare connectors

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self._mw = MainWindow()

        self._cavitySolver : BaseCavitySolver = None
    
    def show(self):
        self._mw.show()

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
        self._mw.show()

    def on_deactivate(self):
        """
        Reverse steps of activation

        Returns:
            error code (int): 0 if OK, -1 if error
        """
        self._mw.close()
        return

    def init_GUI_elements(self):
        """ Configure the GUI window with dynamical elements and connect the signals to the slots"""

        # NOTE: GUI input settings (min, max, default, ...) are set in the .ui file
        # Here we set only the  text values since they may be customized later

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
        self._mw.RUN_start_button.clicked.connect(self.onClk_start_simulation)
        self._mw.RUN_start_button.setEnabled(1)
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
        self._plot_curve : pg.GraphicsObject = pg.PlotDataItem() # TODO maybe different elements for 3d...
        self._mw.plotwindow.addItem(self._plot_curve)
        self.update_plot()
        return
    

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
        self._mw.C_scan_P.setEnabled(self._mw.C_scan_D.isChecked())
        if self._mw.C_scan_D.isChecked() and self._mw.T_model_select.value() == 1:
            raise ValueError('Cannot scan the detuning in quantum mode')
            
    @Slot()
    def onChange_scan_optical_pump_power(self):
        """ Update the graphical elements when we are scanning the optical pump power """
        self._mw.C_scan_D.setEnabled(self._mw.C_scan_P.isChecked())
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
    def onClk_save_data(self):
        """ Save the plot to a png file and the config to a txt file"""
        filepath = self.get_unique_filepath()
        # save the configuration in a json file
        try:
            config = self._cavitySolver.get_current_configuration()
            with open(filepath+"_config.json", 'w') as f:
                json.dump(config, f, indent=2)
        except: pass
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

        indexes= [np.int32(name[-6-l_ext:-l_ext])
                  for name in os.listdir(dir) if name.startswith(filename)]
        num_file = np.max(indexes+[0]) + 1

        filePath = os.path.join(dir,filename+'{:06d}'.format(num_file))
        return filePath
    
    @Slot()
    def onClk_start_simulation(self):
        """ Start the simulation using the current configuration """
        self.enable_interface(False)
        print('Start simulation')
        self._cavitySolver, isTimeEvolution = self.get_and_configure_solver_from_config()
        if isTimeEvolution:
            data = self._cavitySolver.solve_cavity_time_evolution()
        else:
            data = self._cavitySolver.solve_cavity_RT()
        self.update_plot(data, isTimeEvolution)
        self.enable_interface(True)

    def get_and_configure_solver_from_config(self) -> tuple[BaseCavitySolver, bool]:
        """ Create the solver object and configure it according to the GUI configuration """
        solver = BaseCavitySolver()
        # TODO
        isTimeEvolution = self._mw.C_use_time_evolution.isChecked()
        return solver, isTimeEvolution
      
    @Slot()
    def update_plot(self, data=None, isTimeEvolution=False):
        """ 
        Update the plot with the current data 

        Parameters
        ----------
        data : np.ndarray
            The data to be plotted. It can be a 1D array for the reflectivity/transmissivity or a 2D array for the time evolution
        isTimeEvolution : bool
            True if the data is a time evolution, False if it is a reflectivity/transmissivity
        """
        # temp creation of random data
        if data is None:
            data=np.array([np.linspace(12,13,100),np.random.rand(100)])
            
        self.set_plot_data(data, isTimeEvolution)        
        self.set_plot_axis(isTimeEvolution)
        print('Update plot')

    def set_plot_data(self, data, isTimeEvolution):
        """ 
        Set the data to be plotted 

        Parameters
        ----------
        data : np.ndarray
            The data to be plotted. It can be a 1D array for the reflectivity/transmissivity or a 2D array for the time evolution
        isTimeEvolution : bool
            True if the data is a time evolution, False if it is a reflectivity/transmissivity
        """
        if isTimeEvolution:
            # TODO define 3d data, maybe we have to keep 2 elements (_plot_curve and ...) and work with the visibility / remove them when needed
            pass
        else:
            self._frequencies=data[0,:]
            self._powers=data[1,:]
            self._plot_curve.setData(self._frequencies,self._powers, pen='b')
    
    def set_plot_axis(self, isTimeEvolution):
        """ 
        Set the axis label for the plot 

        Parameters
        ----------
        isTimeEvolution : bool
            True if the data is a time evolution, False if it is a reflectivity/transmissivity
        """
        self._plot_curve.getViewBox().updateAutoRange()
        if isTimeEvolution:
            # TODO define 3d axis
            self._mw.plotwindow.setLabel(axis='bottom', text='Time', units='s',pen='k')
            self._mw.plotwindow.setLabel(axis='left', text='Photon/Phonon population', units='1',pen='k')
        else:
            self._mw.plotwindow.getAxis('left').setTextPen('k')
            self._mw.plotwindow.getAxis('bottom').setTextPen('k')
            self._mw.plotwindow.setLabel(axis='bottom', text='Frequency', units='GHz',pen='k')
            self._mw.plotwindow.setLabel(axis='left', text='Reflected/Transmitted power', units='1',pen='k')


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


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication([])
    gui = Simulator()
    gui.on_activate()
    sys.exit(app.exec_())