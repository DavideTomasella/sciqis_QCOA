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
from SimulatorUI_autogen import Ui_Simulator
from qtpy import uic
import pyqtgraph as pg
import numpy as np

#ifdef
_use_autogen=False
from SimulatorUI_autogen import Ui_Simulator

class Simulator_MainWindow_Autogen(QtWidgets.QMainWindow,Ui_Simulator):
    """
    The main window for the cavity control GUI. 
    It extends the standard QMainWindow, but it also inherit the method setupUi() 
    form the autogen file we can create with pyuic5.
    
    To generate the gui from the .ui file:
     - move to the directory where the .ui file is located
     - execute the command: pyuic5 SimulatorUI.ui -o SimulatorUI_autogen.py
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
        uic.loadUi('./SimulatorUI.ui', self)


class Simulator(QObject):
    """
    This is the GUI Class for creating the grafical interface
    """

    # declare connectors

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self._mw = MainWindow()
    
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

        self.config_GUI_elements()
        self.config_plot()

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

    def config_GUI_elements(self):
        """ Configure the GUI window with dynamical elements and connect the signals to the slots"""

        # NOTE: GUI input settings (min, max, default, ...) are set in the .ui file
        # Here we set only the  text values since they may be customized later
        self._mw.F_folder_name.setText(os.path.join(os.getcwd(),"data"))
        self._mw.F_file_name.setText("Simulation.png")


        # Connect signals to slots
        # file and folder selection
        self._mw.F_folder_button.clicked.connect(self.onClk_browse_folders)
        self._mw.F_folder_button.setEnabled(1)
        self._mw.F_file_button.clicked.connect(self.onClk_browse_files)
        self._mw.F_file_button.setEnabled(1)
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
        self._mw.C_use_optical.setChecked(True)
        self._mw.C_use_mech.setChecked(True)
        self._mw.RUN_save_button.setEnabled(False)
        
        return

    def config_plot(self):
        """Configuration of the gui module corresponding to the plot window"""
        self._plot_curve = pg.PlotDataItem()
        self._mw.plotwindow.addItem(self._plot_curve)
        self._mw.plotwindow.setLabel(axis='bottom', text='Frequency', units='GHz')
        self._mw.plotwindow.setLabel(axis='left', text='Reflected/Transmitted power', units='1')
        self.on_update_view()
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
        if self._mw.C_use_bath.isChecked() and self._mw.T_model_select.value() == 0:
            raise ValueError('Cannot use the bath temperature in classical mode')

    @Slot()
    def onChange_use_quantum_time_evolution(self):
        if self._mw.C_use_time_evolution.isChecked() and self._mw.T_model_select.value() == 0:
            raise ValueError('Cannot use the time evolution in classical mode')
    
    @Slot()
    def onChange_scan_optical_FSR_detuning(self):
        self._mw.C_scan_P.setEnabled(self._mw.C_scan_D.isChecked())
        if self._mw.C_scan_D.isChecked() and self._mw.T_model_select.value() == 1:
            raise ValueError('Cannot scan the detuning in quantum mode')
            
    @Slot()
    def onChange_scan_optical_pump_power(self):
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
        self._mw.F_file_name.setText(fileName[0].split("/")[-1])

    @Slot()
    def onClk_save_data(self):
        """ Save the plot to a png file and the config to a txt file"""
        print('Save data')
    
    @Slot()
    def onClk_start_simulation(self):
        """ Start the simulation using the current configuration """
        print('Start simulation')
        self.enable_interface(False)
      
    @Slot()
    def on_update_view(self):
        """ Update the plot with the current data """
        # temp creation of random data
        data=np.array([np.linspace(12,13,100),np.random.rand(100)])
        self._frequencies=data[0,:]
        self._powers=data[1,:]
        self._plot_curve.setData(self._frequencies,self._powers)
        self._plot_curve.getViewBox().updateAutoRange()
        #self.enable_interface(True) # TODO move to caller

    def enable_interface(self, enable=True):
        """ Enable or disable the interface buttons"""
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
