"""
Object manager for FEA UI - manages collections of boundary conditions, loads, etc.
"""

import os
import sys
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QListWidget, QListWidgetItem,
                            QDialog, QFormLayout, QDialogButtonBox, QLineEdit,
                            QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox,
                            QGroupBox, QMenu, QMessageBox, QApplication,
                            QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QIcon, QCursor
import torch

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import FEA


class FEAObjectItem:
    """Class to represent a single FEA object (constraint, load, etc.)"""
    
    def __init__(self, name, obj_type, obj_instance, params=None):
        """
        Initialize FEA object item
        
        Args:
            name: Object name
            obj_type: Object type (constraint, load, etc.)
            obj_instance: The actual instance of the FEA object
            params: Dict of parameters used to create the object
        """
        self.name = name
        self.obj_type = obj_type
        self.obj_instance = obj_instance
        self.params = params or {}
    
    def __str__(self):
        """String representation of the object"""
        if self.obj_type == 'boundary_condition':
            return f"边界条件: {self.name} ({len(self.params.get('indexDOF', [])) // 3} 节点)"
        elif self.obj_type == 'load':
            if 'pressure' in self.params:
                return f"压力载荷: {self.name} (p={self.params.get('pressure', '?')})"
            elif 'force' in self.params:
                return f"集中力: {self.name} (F={self.params.get('force', '?')})"
            elif 'moment' in self.params:
                return f"力矩: {self.name} (M={self.params.get('moment', '?')})"
            else:
                return f"载荷: {self.name}"
        elif self.obj_type == 'reference_point':
            coords = self.params.get('coordinates', [0, 0, 0])
            return f"参考点: {self.name} ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})"
        elif self.obj_type == 'coupling':
            return f"耦合约束: {self.name} ({len(self.params.get('indexNodes', []))} 节点)"
        else:
            return f"{self.obj_type}: {self.name}"


class ObjectListWidget(QListWidget):
    """Enhanced list widget with context menu support"""
    
    deleteRequested = pyqtSignal(int)
    editRequested = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        
    def showContextMenu(self, position):
        if self.count() > 0 and self.currentRow() >= 0:
            menu = QMenu()
            edit_action = menu.addAction("编辑")
            delete_action = menu.addAction("删除")
            
            action = menu.exec_(QCursor.pos())
            
            if action == delete_action:
                self.deleteRequested.emit(self.currentRow())
            elif action == edit_action:
                self.editRequested.emit(self.currentRow())


class ObjectManagerWidget(QWidget):
    """Widget for managing FEA objects like boundary conditions, loads, etc."""
    
    objectAdded = pyqtSignal(str, object)  # type, object
    objectRemoved = pyqtSignal(str, str)   # type, name
    objectModified = pyqtSignal(str, str, object)  # type, name, object
    
    def __init__(self, fe_model=None, fem_data=None, parent=None):
        super().__init__(parent)
        # Allow this widget to expand
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fe_model = fe_model  # The FEA model
        self.fem_data = fem_data  # The INP data
        
        # Dictionary to store all FEA objects by type
        self.objects = {
            'boundary_condition': [],  # Boundary conditions
            'load': [],                # Loads
            'reference_point': [],     # Reference points
            'coupling': [],            # Couplings
        }
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout()
        # Ensure this widget expands vertically (handled via stretch on child widgets)
        # layout.setStretchFactor(layout, 0)  # removed incorrect usage
        
        # Boundary Conditions Section
        bc_group = QGroupBox("边界条件")
        bc_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        bc_layout = QVBoxLayout()
        
        # Header with buttons
        bc_header = QHBoxLayout()
        bc_title = QLabel("<b>边界条件</b>")
        bc_add_btn = QPushButton("添加")
        bc_add_btn.clicked.connect(lambda: self.add_object_dialog('boundary_condition'))
        bc_header.addWidget(bc_title)
        bc_header.addStretch()
        bc_header.addWidget(bc_add_btn)
        bc_layout.addLayout(bc_header)
        
        # List of boundary conditions
        self.bc_list = ObjectListWidget()
        self.bc_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.bc_list.setMinimumHeight(80)
        self.bc_list.deleteRequested.connect(lambda idx: self.delete_object('boundary_condition', idx))
        self.bc_list.editRequested.connect(lambda idx: self.edit_object('boundary_condition', idx))
        # Debug: Add border to list for visibility
        self.bc_list.setStyleSheet("border: 1px solid red;")
        bc_layout.addWidget(self.bc_list, stretch=1)
        
        # Delete button for boundary conditions
        bc_delete_btn = QPushButton("删除所选")
        bc_delete_btn.clicked.connect(lambda: self.delete_selected_object('boundary_condition'))
        bc_layout.addWidget(bc_delete_btn)
        
        bc_group.setLayout(bc_layout)
        
        # Loads Section
        loads_group = QGroupBox("载荷")
        loads_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        loads_layout = QVBoxLayout()
        
        # Header with buttons
        loads_header = QHBoxLayout()
        loads_title = QLabel("<b>载荷</b>")
        loads_add_btn = QPushButton("添加")
        loads_add_btn.clicked.connect(lambda: self.add_object_dialog('load'))
        loads_header.addWidget(loads_title)
        loads_header.addStretch()
        loads_header.addWidget(loads_add_btn)
        loads_layout.addLayout(loads_header)
        
        # List of loads
        self.loads_list = ObjectListWidget()
        self.loads_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.loads_list.setMinimumHeight(80)
        self.loads_list.deleteRequested.connect(lambda idx: self.delete_object('load', idx))
        self.loads_list.editRequested.connect(lambda idx: self.edit_object('load', idx))
        # Debug: Add border to list for visibility
        self.loads_list.setStyleSheet("border: 1px solid green;")
        loads_layout.addWidget(self.loads_list, stretch=1)
        
        # Delete button for loads
        loads_delete_btn = QPushButton("删除所选")
        loads_delete_btn.clicked.connect(lambda: self.delete_selected_object('load'))
        loads_layout.addWidget(loads_delete_btn)
        
        loads_group.setLayout(loads_layout)
        
        # Reference Points Section
        rp_group = QGroupBox("参考点")
        rp_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rp_layout = QVBoxLayout()
        
        # Header with buttons
        rp_header = QHBoxLayout()
        rp_title = QLabel("<b>参考点</b>")
        rp_add_btn = QPushButton("添加")
        rp_add_btn.clicked.connect(lambda: self.add_object_dialog('reference_point'))
        rp_header.addWidget(rp_title)
        rp_header.addStretch()
        rp_header.addWidget(rp_add_btn)
        rp_layout.addLayout(rp_header)
        
        # List of reference points
        self.rp_list = ObjectListWidget()
        self.rp_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.rp_list.setMinimumHeight(80)
        self.rp_list.deleteRequested.connect(lambda idx: self.delete_object('reference_point', idx))
        self.rp_list.editRequested.connect(lambda idx: self.edit_object('reference_point', idx))
        # Debug: Add border to list for visibility
        self.rp_list.setStyleSheet("border: 1px solid blue;")
        rp_layout.addWidget(self.rp_list, stretch=1)
        
        # Delete button for reference points
        rp_delete_btn = QPushButton("删除所选")
        rp_delete_btn.clicked.connect(lambda: self.delete_selected_object('reference_point'))
        rp_layout.addWidget(rp_delete_btn)
        
        rp_group.setLayout(rp_layout)
        
        # Couplings Section
        coupling_group = QGroupBox("耦合约束")
        coupling_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        coupling_layout = QVBoxLayout()
        
        # Header with buttons
        coupling_header = QHBoxLayout()
        coupling_title = QLabel("<b>耦合约束</b>")
        coupling_add_btn = QPushButton("添加")
        coupling_add_btn.clicked.connect(lambda: self.add_object_dialog('coupling'))
        coupling_header.addWidget(coupling_title)
        coupling_header.addStretch()
        coupling_header.addWidget(coupling_add_btn)
        coupling_layout.addLayout(coupling_header)
        
        # List of couplings
        self.coupling_list = ObjectListWidget()
        self.coupling_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.coupling_list.setMinimumHeight(80)
        self.coupling_list.deleteRequested.connect(lambda idx: self.delete_object('coupling', idx))
        self.coupling_list.editRequested.connect(lambda idx: self.edit_object('coupling', idx))
        # Debug: Add border to list for visibility
        self.coupling_list.setStyleSheet("border: 1px solid orange;")
        coupling_layout.addWidget(self.coupling_list, stretch=1)
        
        # Delete button for couplings
        coupling_delete_btn = QPushButton("删除所选")
        coupling_delete_btn.clicked.connect(lambda: self.delete_selected_object('coupling'))
        coupling_layout.addWidget(coupling_delete_btn)
        
        coupling_group.setLayout(coupling_layout)
        
        # Add all sections to main layout with stretch
        layout.addWidget(bc_group, stretch=1)
        layout.addWidget(loads_group, stretch=1)
        layout.addWidget(rp_group, stretch=1)
        layout.addWidget(coupling_group, stretch=1)
        
        # Populate lists initially
        self.refresh_lists()
        self.setLayout(layout)
    
    def delete_selected_object(self, obj_type):
        """Delete the currently selected object from the list"""
        list_widget = self._get_list_widget(obj_type)
        if list_widget and list_widget.currentRow() >= 0:
            idx = list_widget.currentRow()
            self.delete_object(obj_type, idx)
    
    def set_models(self, fe_model, fem_data):
        """Set the FE model and FEM data"""
        self.fe_model = fe_model
        self.fem_data = fem_data
        self.refresh_lists()
    
    def add_object(self, obj_type, name, obj_instance, params=None):
        """Add an object to the manager"""
        item = FEAObjectItem(name, obj_type, obj_instance, params)
        self.objects[obj_type].append(item)
        # Refresh all lists and update UI
        self.refresh_lists()
        # Select the newly added item in its list
        list_widget = self._get_list_widget(obj_type)
        if list_widget:
            idx = list_widget.count() - 1
            if idx >= 0:
                list_widget.setCurrentRow(idx)
                list_widget.scrollToItem(list_widget.item(idx))
        # Emit signal
        self.objectAdded.emit(obj_type, obj_instance)
        return item

    def delete_object(self, obj_type, index):
        """Delete an object from the manager"""
        if 0 <= index < len(self.objects[obj_type]):
            item = self.objects[obj_type].pop(index)
            # Directly remove from the list widget
            list_widget = self._get_list_widget(obj_type)
            if list_widget:
                list_widget.blockSignals(True)
                list_widget.takeItem(index)
                list_widget.blockSignals(False)
                list_widget.repaint()
            # Emit signal
            self.objectRemoved.emit(obj_type, item.name)
            return True
        return False
    
    def edit_object(self, obj_type, index):
        """Edit an object in the manager"""
        if 0 <= index < len(self.objects[obj_type]):
            item = self.objects[obj_type][index]
            
            # Open edit dialog based on object type
            if obj_type == 'reference_point':
                self._edit_reference_point(index, item)
            elif obj_type == 'boundary_condition':
                self._edit_boundary_condition(index, item)
            elif obj_type == 'load':
                self._edit_load(index, item)
            elif obj_type == 'coupling':
                self._edit_coupling(index, item)
            
            return True
        return False
    
    def refresh_lists(self):
        """Refresh all list widgets"""
        self._refresh_list('boundary_condition', self.bc_list)
        self._refresh_list('load', self.loads_list)
        self._refresh_list('reference_point', self.rp_list)
        self._refresh_list('coupling', self.coupling_list)
    
    def _refresh_list(self, obj_type, list_widget):
        """Refresh a specific list widget"""
        list_widget.clear()
        for item in self.objects[obj_type]:
            list_widget.addItem(str(item))
    
    def _get_list_widget(self, obj_type):
        """Get the list widget for a specific object type"""
        if obj_type == 'boundary_condition':
            return self.bc_list
        elif obj_type == 'load':
            return self.loads_list
        elif obj_type == 'reference_point':
            return self.rp_list
        elif obj_type == 'coupling':
            return self.coupling_list
        return None
    
    def add_object_dialog(self, obj_type):
        """Show dialog to add a new object"""
        if obj_type == 'reference_point':
            self._add_reference_point()
        elif obj_type == 'boundary_condition':
            self._add_boundary_condition()
        elif obj_type == 'load':
            self._add_load()
        elif obj_type == 'coupling':
            self._add_coupling()
    
    def _add_reference_point(self):
        """Add a new reference point"""
        if not self.fe_model:
            QMessageBox.warning(self, "添加参考点", "未加载FEA模型")
            return
            
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("添加参考点")
        layout = QFormLayout()
        
        # Add name field
        name_input = QLineEdit()
        name_input.setText(f"RP-{len(self.objects['reference_point'])+1}")
        layout.addRow("名称:", name_input)
        
        # Add coordinate fields
        x_input = QDoubleSpinBox()
        x_input.setRange(-10000, 10000)
        x_input.setDecimals(6)
        layout.addRow("X 坐标:", x_input)
        
        y_input = QDoubleSpinBox()
        y_input.setRange(-10000, 10000)
        y_input.setDecimals(6)
        layout.addRow("Y 坐标:", y_input)
        
        z_input = QDoubleSpinBox()
        z_input.setRange(-10000, 10000)
        z_input.setDecimals(6)
        layout.addRow("Z 坐标:", z_input)
        
        # Add buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            name = name_input.text()
            x = x_input.value()
            y = y_input.value()
            z = z_input.value()
            
            # Create reference point
            rp = FEA.ReferencePoint([x, y, z])
            rp_name = self.fe_model.add_reference_point(rp)
            
            # Add to manager
            params = {'coordinates': [x, y, z]}
            self.add_object('reference_point', rp_name, rp, params)
    
    def _edit_reference_point(self, index, item):
        """Edit an existing reference point"""
        if not self.fe_model:
            QMessageBox.warning(self, "编辑参考点", "未加载FEA模型")
            return
            
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("编辑参考点")
        layout = QFormLayout()
        
        # Add name field (read-only)
        name_input = QLineEdit()
        name_input.setText(item.name)
        name_input.setReadOnly(True)
        layout.addRow("名称:", name_input)
        
        # Get current coordinates
        coords = item.params.get('coordinates', [0, 0, 0])
        
        # Add coordinate fields
        x_input = QDoubleSpinBox()
        x_input.setRange(-10000, 10000)
        x_input.setDecimals(6)
        x_input.setValue(coords[0])
        layout.addRow("X 坐标:", x_input)
        
        y_input = QDoubleSpinBox()
        y_input.setRange(-10000, 10000)
        y_input.setDecimals(6)
        y_input.setValue(coords[1])
        layout.addRow("Y 坐标:", y_input)
        
        z_input = QDoubleSpinBox()
        z_input.setRange(-10000, 10000)
        z_input.setDecimals(6)
        z_input.setValue(coords[2])
        layout.addRow("Z 坐标:", z_input)
        
        # Add buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            x = x_input.value()
            y = y_input.value()
            z = z_input.value()
            
            # Update reference point
            self.fe_model.rPs[item.name].coordinate = np.array([x, y, z])
            
            # Update in manager
            item.params['coordinates'] = [x, y, z]
            self._refresh_list('reference_point', self.rp_list)
            
            # Emit signal
            self.objectModified.emit('reference_point', item.name, self.fe_model.rPs[item.name])
    
    def _add_boundary_condition(self):
        """Add a new boundary condition"""
        if not self.fe_model or not self.fem_data:
            QMessageBox.warning(self, "添加边界条件", "未加载FEA模型")
            return
            
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("添加边界条件")
        layout = QFormLayout()
        
        # Add name field
        name_input = QLineEdit()
        name_input.setText(f"BC-{len(self.objects['boundary_condition'])+1}")
        layout.addRow("名称:", name_input)
        
        # Add node set selector
        node_set_combo = QComboBox()
        # Populate node sets from FEA_Main
        for set_name in self.fe_model.node_sets.keys():
            node_set_combo.addItem(set_name, set_name)
        layout.addRow("节点集:", node_set_combo)
        
        # Create group box for translational DOFs
        trans_group = QGroupBox("平移自由度")
        trans_layout = QFormLayout()
        
        # Translational DOF checkboxes and value inputs
        x_dof = QCheckBox("约束X方向")
        x_dof.setChecked(True)
        x_value = QDoubleSpinBox()
        x_value.setRange(-1000, 1000)
        x_value.setValue(0)
        x_value.setDecimals(6)
        trans_layout.addRow(x_dof, x_value)
        
        y_dof = QCheckBox("约束Y方向")
        y_dof.setChecked(True)
        y_value = QDoubleSpinBox()
        y_value.setRange(-1000, 1000)
        y_value.setValue(0)
        y_value.setDecimals(6)
        trans_layout.addRow(y_dof, y_value)
        
        z_dof = QCheckBox("约束Z方向")
        z_dof.setChecked(True)
        z_value = QDoubleSpinBox()
        z_value.setRange(-1000, 1000)
        z_value.setValue(0)
        z_value.setDecimals(6)
        trans_layout.addRow(z_dof, z_value)
        
        trans_group.setLayout(trans_layout)
        layout.addRow(trans_group)
        
        # Create group box for rotational DOFs
        rot_group = QGroupBox("转动自由度")
        rot_layout = QFormLayout()
        
        # Rotational DOF checkboxes and value inputs
        rx_dof = QCheckBox("约束Rx方向")
        rx_dof.setChecked(False)
        rx_value = QDoubleSpinBox()
        rx_value.setRange(-3.15, 3.15)
        rx_value.setValue(0)
        rx_value.setDecimals(6)
        rx_value.setEnabled(False)
        rot_layout.addRow(rx_dof, rx_value)
        
        ry_dof = QCheckBox("约束Ry方向")
        ry_dof.setChecked(False)
        ry_value = QDoubleSpinBox()
        ry_value.setRange(-3.15, 3.15)
        ry_value.setValue(0)
        ry_value.setDecimals(6)
        ry_value.setEnabled(False)
        rot_layout.addRow(ry_dof, ry_value)
        
        rz_dof = QCheckBox("约束Rz方向")
        rz_dof.setChecked(False)
        rz_value = QDoubleSpinBox()
        rz_value.setRange(-3.15, 3.15)
        rz_value.setValue(0)
        rz_value.setDecimals(6)
        rz_value.setEnabled(False)
        rot_layout.addRow(rz_dof, rz_value)
        
        # Connect checkboxes to enable/disable value inputs
        x_dof.toggled.connect(lambda checked: x_value.setEnabled(checked))
        y_dof.toggled.connect(lambda checked: y_value.setEnabled(checked))
        z_dof.toggled.connect(lambda checked: z_value.setEnabled(checked))
        rx_dof.toggled.connect(lambda checked: rx_value.setEnabled(checked))
        ry_dof.toggled.connect(lambda checked: ry_value.setEnabled(checked))
        rz_dof.toggled.connect(lambda checked: rz_value.setEnabled(checked))
        
        rot_group.setLayout(rot_layout)
        layout.addRow(rot_group)
        
        # Add buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            name = name_input.text()
            
            # Get selected node set
            idx = node_set_combo.currentIndex()
            if idx < 0:
                QMessageBox.warning(self, "添加边界条件", "未选择节点集")
                return
            set_name = node_set_combo.itemData(idx)
            # Get nodes from FEA_Main
            nodes = self.fe_model.node_sets.get(set_name, [])
            bc_dof = []
            
            # Store displacement values
            disp_values = []
            
            # Create DOF indices and corresponding displacement values for translational DOFs
            for node in nodes:
                if x_dof.isChecked():
                    bc_dof.append(node * 3)
                    disp_values.append(x_value.value())
                if y_dof.isChecked():
                    bc_dof.append(node * 3 + 1)
                    disp_values.append(y_value.value())
                if z_dof.isChecked():
                    bc_dof.append(node * 3 + 2)
                    disp_values.append(z_value.value())
            
            bc_dof = np.array(bc_dof)
            disp_values = np.array(disp_values)
            
            # Any rotational DOFs active?
            has_rotational = rx_dof.isChecked() or ry_dof.isChecked() or rz_dof.isChecked()
            
            # Create boundary condition
            bc = FEA.constraints.Boundary_Condition(
                indexDOF=bc_dof,
                dispValue=torch.tensor(disp_values, dtype=torch.float32),
                rotational=has_rotational,
            )
            
            # Add rotational DOFs if needed
            if has_rotational:
                rx_indices = []
                ry_indices = []
                rz_indices = []
                
                for node in nodes:
                    if rx_dof.isChecked():
                        rx_indices.append(node * 3)
                    if ry_dof.isChecked():
                        ry_indices.append(node * 3 + 1)
                    if rz_dof.isChecked():
                        rz_indices.append(node * 3 + 2)
                        
                if rx_indices:
                    bc.set_rotation_indices(rx_indices=np.array(rx_indices))
                if ry_indices:
                    bc.set_rotation_indices(ry_indices=np.array(ry_indices))
                if rz_indices:
                    bc.set_rotation_indices(rz_indices=np.array(rz_indices))
            
            # Add to FE model and get the actual name used in the backend
            actual_name = self.fe_model.add_constraint(bc, name=name)
        
            # Add to manager with the actual name from the backend
            params = {
                'indexDOF': bc_dof,
                'dispValues': disp_values.tolist(),
                'node_set': set_name,
                'dof': {
                    'x': {'constrained': x_dof.isChecked(), 'value': x_value.value()},
                    'y': {'constrained': y_dof.isChecked(), 'value': y_value.value()},
                    'z': {'constrained': z_dof.isChecked(), 'value': z_value.value()},
                    'rx': {'constrained': rx_dof.isChecked(), 'value': rx_value.value()},
                    'ry': {'constrained': ry_dof.isChecked(), 'value': ry_value.value()},
                    'rz': {'constrained': rz_dof.isChecked(), 'value': rz_value.value()}
                },
                'rotational': has_rotational
            }
            self.add_object('boundary_condition', actual_name, bc, params)
    
    def _edit_boundary_condition(self, index, item):
        """Edit an existing boundary condition"""
        QMessageBox.information(self, "编辑边界条件", "边界条件参数不可修改，请删除后重新创建")
    
    def _add_load(self):
        """Add a new load"""
        if not self.fe_model or not self.fem_data:
            QMessageBox.warning(self, "添加载荷", "未加载FEA模型")
            return
            
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("添加载荷")
        layout = QFormLayout()
        
        # Add name field
        name_input = QLineEdit()
        name_input.setText(f"Load-{len(self.objects['load'])+1}")
        layout.addRow("名称:", name_input)
        
        # Add load type selector
        load_type = QComboBox()
        load_type.addItem("压力", "pressure")
        load_type.addItem("集中力", "force")
        load_type.addItem("力矩", "moment")
        layout.addRow("载荷类型:", load_type)
        
        # Add surface selector (for pressure)
        surface_group = QGroupBox("压力载荷设置")
        surface_layout = QVBoxLayout()
        
        surface_combo = QComboBox()
        # Populate surface sets from FEA_Main
        for surface_name in self.fe_model.surface_sets.keys():
            surface_combo.addItem(surface_name, surface_name)
        
        surface_layout.addWidget(QLabel("表面:"))
        surface_layout.addWidget(surface_combo)
        
        pressure_value = QDoubleSpinBox()
        pressure_value.setRange(-1000, 1000)
        pressure_value.setValue(0.06)
        pressure_value.setDecimals(6)
        surface_layout.addWidget(QLabel("压力值:"))
        surface_layout.addWidget(pressure_value)
        
        surface_group.setLayout(surface_layout)
        
        # Add force settings (for force)
        force_group = QGroupBox("集中力载荷设置")
        force_layout = QVBoxLayout()
        
        # Reference point selector for force
        rp_force_combo = QComboBox()
        # Populate reference points
        for rp_item in self.objects['reference_point']:
            rp_force_combo.addItem(rp_item.name, rp_item)
        force_layout.addWidget(QLabel("参考点:"))
        force_layout.addWidget(rp_force_combo)
        
        # Force components
        force_x = QDoubleSpinBox()
        force_x.setRange(-1000, 1000)
        force_x.setValue(0)
        force_x.setDecimals(6)
        force_layout.addWidget(QLabel("X分量:"))
        force_layout.addWidget(force_x)
        
        force_y = QDoubleSpinBox()
        force_y.setRange(-1000, 1000)
        force_y.setValue(0)
        force_y.setDecimals(6)
        force_layout.addWidget(QLabel("Y分量:"))
        force_layout.addWidget(force_y)
        
        force_z = QDoubleSpinBox()
        force_z.setRange(-1000, 1000)
        force_z.setValue(0)
        force_z.setDecimals(6)
        force_layout.addWidget(QLabel("Z分量:"))
        force_layout.addWidget(force_z)
        
        force_group.setLayout(force_layout)
        
        # Add moment settings (for moment)
        moment_group = QGroupBox("力矩载荷设置")
        moment_layout = QVBoxLayout()
        
        # Reference point selector for moment
        rp_moment_combo = QComboBox()
        for rp_item in self.objects['reference_point']:
            rp_moment_combo.addItem(rp_item.name, rp_item)
        moment_layout.addWidget(QLabel("参考点:"))
        moment_layout.addWidget(rp_moment_combo)
        
        # Moment components
        moment_x = QDoubleSpinBox()
        moment_x.setRange(-1000, 1000)
        moment_x.setValue(0)
        moment_x.setDecimals(6)
        moment_layout.addWidget(QLabel("X分量:"))
        moment_layout.addWidget(moment_x)
        
        moment_y = QDoubleSpinBox()
        moment_y.setRange(-1000, 1000)
        moment_y.setValue(0)
        moment_y.setDecimals(6)
        moment_layout.addWidget(QLabel("Y分量:"))
        moment_layout.addWidget(moment_y)
        
        moment_z = QDoubleSpinBox()
        moment_z.setRange(-1000, 1000)
        moment_z.setValue(0)
        moment_z.setDecimals(6)
        moment_layout.addWidget(QLabel("Z分量:"))
        moment_layout.addWidget(moment_z)
        
        moment_group.setLayout(moment_layout)
        
        # Add all load specific groups
        layout.addRow(surface_group)
        layout.addRow(force_group)
        layout.addRow(moment_group)
        
        # Hide all groups initially
        surface_group.setVisible(True)
        force_group.setVisible(False)
        moment_group.setVisible(False)
        
        # Connect load type selector
        def on_load_type_changed(index):
            load_type_val = load_type.itemData(index)
            surface_group.setVisible(load_type_val == "pressure")
            force_group.setVisible(load_type_val == "force")
            moment_group.setVisible(load_type_val == "moment")
            
        load_type.currentIndexChanged.connect(on_load_type_changed)
        
        # Add buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            name = name_input.text()
            load_type_val = load_type.itemData(load_type.currentIndex())
            
            # Create load based on type
            if load_type_val == "pressure":
                # Get selected surface set
                idx = surface_combo.currentIndex()
                if idx < 0:
                    QMessageBox.warning(self, "添加载荷", "未选择表面集")
                    return
                surface_set_name = surface_combo.itemData(idx)
                pressure = pressure_value.value()
                
                # Create pressure load
                load = FEA.loads.Pressure(
                    surface_set=surface_set_name,
                    pressure=pressure
                )
                
                # Add to FE model and get the actual name
                actual_name = self.fe_model.add_load(load, name=name)
                
                # Add to manager with actual name from backend
                params = {
                    'pressure': pressure,
                    'surface': surface_set_name
                }
                self.add_object('load', actual_name, load, params)
                
            elif load_type_val == "force":
                idx = rp_force_combo.currentIndex()
                if idx < 0:
                    QMessageBox.warning(self, "添加载荷", "未选择参考点")
                    return
                rp_item = rp_force_combo.itemData(idx)
                # Assuming rp_item.obj_instance provides node ID
                node = rp_item.obj_instance.id if hasattr(rp_item.obj_instance, 'id') else None
                if node is None:
                    QMessageBox.warning(self, "添加载荷", "参考点无效")
                    return

                fx = force_x.value()
                fy = force_y.value()
                fz = force_z.value()
                
                # Create force load
                load = FEA.loads.Concentrate_Force(
                    nodeID=node,
                    force=[fx, fy, fz]
                )
                
                # Add to FE model and get the actual name
                actual_name = self.fe_model.add_load(load, name=name)
                
                # Add to manager with the actual name from the backend
                params = {
                    'nodeID': node,
                    'force': [fx, fy, fz]
                }
                self.add_object('load', actual_name, load, params)
                
            elif load_type_val == "moment":
                idx = rp_moment_combo.currentIndex()
                if idx < 0:
                    QMessageBox.warning(self, "添加载荷", "未选择参考点")
                    return
                rp_item = rp_moment_combo.itemData(idx)
                node = rp_item.obj_instance.id if hasattr(rp_item.obj_instance, 'id') else None
                if node is None:
                    QMessageBox.warning(self, "添加载荷", "参考点无效")
                    return

                mx = moment_x.value()
                my = moment_y.value()
                mz = moment_z.value()
                
                # Create moment load
                load = FEA.loads.Moment(
                    nodeID=node,
                    moment=[mx, my, mz]
                )
                
                # Add to FE model and get the actual name
                actual_name = self.fe_model.add_load(load, name=name)
                
                # Add to manager with the actual name from the backend
                params = {
                    'nodeID': node,
                    'moment': [mx, my, mz]
                }
                self.add_object('load', actual_name, load, params)
    
    def _edit_load(self, index, item):
        """Edit an existing load"""
        QMessageBox.information(self, "编辑载荷", "载荷参数不可修改，请删除后重新创建")
    
    def _add_coupling(self):
        """Add a new coupling"""
        if not self.fe_model:
            QMessageBox.warning(self, "添加耦合约束", "未加载FEA模型")
            return
            
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("添加耦合约束")
        layout = QFormLayout()
        
        # Add name field
        name_input = QLineEdit()
        name_input.setText(f"Coupling-{len(self.objects['coupling'])+1}")
        layout.addRow("名称:", name_input)
        
        # Add reference point selector
        rp_combo = QComboBox()
        
        # Populate reference points from FE model if available
        if self.fe_model and hasattr(self.fe_model, 'rPs'):
            for rp_name in self.fe_model.rPs.keys():
                rp_combo.addItem(rp_name, rp_name)
        else:
            # Fallback to manager's reference points
            for rp_item in self.objects['reference_point']:
                rp_combo.addItem(rp_item.name, rp_item.name)
            
        layout.addRow("参考点:", rp_combo)
        
        # Add node set selector instead of Z-coordinate
        node_set_combo = QComboBox()
        # Populate node sets from FEA_Main
        for set_name in self.fe_model.node_sets.keys():
            node_set_combo.addItem(set_name, set_name)
        layout.addRow("节点集:", node_set_combo)
        
        # Add buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        
        dialog.setLayout(layout)
        
        # Show dialog
        if rp_combo.count() == 0:
            QMessageBox.warning(self, "添加耦合约束", "请先添加参考点")
            return
        if dialog.exec_() == QDialog.Accepted:
            name = name_input.text()
            # Get selected reference point
            rp_idx = rp_combo.currentIndex()
            if rp_idx < 0:
                QMessageBox.warning(self, "添加耦合约束", "未选择参考点")
                return
            rp_name = rp_combo.itemData(rp_idx)
            # Get selected node set
            set_idx = node_set_combo.currentIndex()
            if set_idx < 0:
                QMessageBox.warning(self, "添加耦合约束", "未选择节点集")
                return
            set_name = node_set_combo.itemData(set_idx)
            # Get nodes array from FEA_Main
            indexNodes = np.array(self.fe_model.node_sets.get(set_name, []), dtype=int)
            
            if len(indexNodes) == 0:
                QMessageBox.warning(self, "添加耦合约束", f"节点集 {set_name} 中未找到节点")
                return
                
            # Create coupling
            coupling = FEA.constraints.Couple(
                indexNodes=indexNodes,
                rp_name=rp_name
            )
            
            # Add to FE model and get the actual name used in the backend
            actual_name = self.fe_model.add_constraint(coupling, name=name)
            
            # Add to manager with the actual name from the backend
            params = {
                'indexNodes': indexNodes,
                'node_set': set_name,
                'rp_name': rp_name
            }
            self.add_object('coupling', actual_name, coupling, params)
    
    def _edit_coupling(self, index, item):
        """Edit an existing coupling"""
        QMessageBox.information(self, "编辑耦合约束", "耦合约束参数不可修改，请删除后重新创建")
    
    def clear_all_objects(self):
        """Clear all objects from the manager"""
        # Clear all objects from the internal dictionary
        for obj_type in self.objects:
            self.objects[obj_type] = []
        
        # Refresh all list widgets
        self.refresh_lists()
        
        # Emit signals for each removed object (if needed)
        # This informs other components that objects have been removed
        # Note: We're not emitting signals here as it could cause performance issues
        # when clearing many objects at once
        
        return True