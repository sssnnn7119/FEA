"""
Main UI launcher for FEA application
"""

import os
import sys
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import Qt

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .device_selection import DeviceSelectionDialog
from .fea_widget import FEAWidget


class MainWindow(QMainWindow):
    """Main application window for FEA UI"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Finite Element Analysis Application")
        self.resize(1200, 800)
        
        # Show device selection dialog first
        self.select_device()
    
    def select_device(self):
        """Show device selection dialog and initialize UI"""
        dialog = DeviceSelectionDialog(self)
        result = dialog.exec_()
        
        if result == 1:  # Accepted
            device = dialog.get_selected_device()
            dtype = dialog.get_selected_dtype()
            self.init_ui(device, dtype)
        else:
            # Cancel was clicked, exit the application
            sys.exit(0)
    
    def init_ui(self, device, dtype):
        """Initialize the main UI after device selection"""
        # Create FEA widget
        self.fea_widget = FEAWidget(device, dtype, parent=self)
        self.setCentralWidget(self.fea_widget)
        
        # Set window properties
        self.setWindowTitle(f"FEA Application - {device} - {dtype}")


def launch_ui():
    """Launch the FEA UI application"""
    # Create application
    app = QApplication(sys.argv)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Execute application
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch_ui()