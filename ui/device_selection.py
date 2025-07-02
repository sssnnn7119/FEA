"""
Dialog for selecting computation device for FEA
"""

import torch
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QComboBox,
                            QPushButton, QHBoxLayout, QGroupBox)


class DeviceSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FEA Device Selection")
        self.resize(400, 200)
        self.selected_device = None
        self.selected_dtype = torch.float64  # Default dtype
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Device selection section
        device_group = QGroupBox("Computation Device")
        device_layout = QVBoxLayout()
        
        # Detect available devices
        self.available_devices = []
        device_names = []
        
        # Always add CPU
        self.available_devices.append(torch.device('cpu'))
        device_names.append("CPU")
        
        # Check for CUDA devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f'cuda:{i}')
                self.available_devices.append(device)
                device_name = torch.cuda.get_device_name(i)
                device_names.append(f"CUDA:{i} - {device_name}")
        
        self.device_label = QLabel("Select computation device:")
        self.device_combo = QComboBox()
        self.device_combo.addItems(device_names)
        
        device_layout.addWidget(self.device_label)
        device_layout.addWidget(self.device_combo)
        device_group.setLayout(device_layout)
        
        # Precision selection
        precision_group = QGroupBox("Numeric Precision")
        precision_layout = QVBoxLayout()
        
        self.precision_label = QLabel("Select numeric precision:")
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(["Float64 (double precision)", "Float32 (single precision)"])
        
        precision_layout.addWidget(self.precision_label)
        precision_layout.addWidget(self.precision_combo)
        precision_group.setLayout(precision_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        
        self.ok_button.clicked.connect(self.accept_selection)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        
        # Add all components to main layout
        layout.addWidget(device_group)
        layout.addWidget(precision_group)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def accept_selection(self):
        device_index = self.device_combo.currentIndex()
        self.selected_device = self.available_devices[device_index]
        
        # Set precision
        precision_index = self.precision_combo.currentIndex()
        self.selected_dtype = torch.float64 if precision_index == 0 else torch.float32
        
        self.accept()
    
    def get_selected_device(self):
        return self.selected_device
    
    def get_selected_dtype(self):
        return self.selected_dtype