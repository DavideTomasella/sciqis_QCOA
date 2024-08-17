"""
GUI for the simulator
Author: Davide Tomasella
"""
# In[]
import os
from qtpy import QtCore
from qtpy.QtCore import Slot
from qtpy import QtWidgets
from SimulatorUI_autogen import UI_Simulator
from qtpy import uic
import pyqtgraph as pg
import numpy as np

#ifdef
_use_autogen=False
if _use_autogen:
    from SimulatorUI_autogen import UI_Simulator

class Simulator_MainWindow_Autogen(QtWidgets.QMainWindow,UI_Simulator):
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


class Simulator():
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
        return

    def config_plot(self):
        """Configuration of the gui module corresponding to the plot window"""
        self._plot_curve = pg.PlotDataItem()
        self._plot_curve = pg.PlotDataItem(x=[0,1,1,0],y=[0,1,0,1])
        self._mw.plotwindow.addItem(self._plot_curve)
        self._frequencies=[]
        self._powers=[]
        self._mw.plotwindow.setLabel(axis='bottom', text='Frequency', units='MHz')
        self._mw.plotwindow.setLabel(axis='left', text='Power', units='dBm')
        #elf._mw.plotwindow.setTitle('Measurement')
        return
    
    @Slot()
    def onClk_start_single_measurement(self):
        print('Start single measurement')
      
    @Slot()
    def on_update_view(self):
        self._plot_curve.getViewBox().updateAutoRange()
        data=np.array([np.linspace(12,13,100),np.random.rand(100)])
        self._frequencies=data[0,:]
        self._powers=data[1,:]
        self._plot_curve.setData(self._frequencies,self._powers)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication([])
    gui = Simulator()
    gui.on_activate()
    sys.exit(app.exec_())
