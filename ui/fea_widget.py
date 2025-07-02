"""
FEA operations widget for the UI
"""

import os
import sys
import numpy as np
import torch
import time
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QFileDialog, QTreeWidget, QTreeWidgetItem,
                            QGroupBox, QSplitter, QComboBox, QDoubleSpinBox, 
                            QLineEdit, QMessageBox, QTabWidget, QTextEdit,
                            QProgressDialog, QApplication, QStackedWidget)
from PyQt5.QtCore import Qt, pyqtSignal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import FEA
from .visualization import MayaviVisualization
from .object_manager import ObjectManagerWidget


class FEAWidget(QWidget):
    """Main widget for FEA operations"""
    
    visualization_requested = pyqtSignal(object, object, object)
    
    def __init__(self, device, dtype, parent=None):
        super().__init__(parent)
        self.device = device
        self.dtype = dtype
        
        # Set torch default device and dtype
        torch.set_default_device(self.device)
        torch.set_default_dtype(self.dtype)
        
        # Initialize FEA objects
        self.fem = None
        self.fe = None
        self.inp_file = None
        
        self.init_ui()
    
    def init_ui(self):
        main_layout = QHBoxLayout()
        
        # Left panel for controls
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        
        # File operations
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout(file_group)
        
        self.load_inp_btn = QPushButton("加载INP文件")
        self.load_inp_btn.clicked.connect(self.load_inp_file)
        self.file_path_label = QLabel("未加载文件")
        
        file_layout.addWidget(self.load_inp_btn)
        file_layout.addWidget(self.file_path_label)
        
        # Model tree
        model_group = QGroupBox("模型结构")
        model_layout = QVBoxLayout(model_group)
        self.model_tree = QTreeWidget()
        self.model_tree.setHeaderLabels(["对象", "详情"])
        model_layout.addWidget(self.model_tree)
        
        # Object Manager
        self.object_manager = ObjectManagerWidget(parent=self)
        self.object_manager.objectAdded.connect(self.on_object_added)
        self.object_manager.objectRemoved.connect(self.on_object_removed)
        self.object_manager.objectModified.connect(self.on_object_modified)
        
        # Solve settings group
        solve_group = QGroupBox("求解设置")
        solve_layout = QVBoxLayout(solve_group)
        
        # Tolerance Error
        tol_layout = QHBoxLayout()
        tol_layout.addWidget(QLabel("容差:"))
        self.tol_error = QDoubleSpinBox()
        self.tol_error.setRange(0.001, 10)
        self.tol_error.setValue(0.1)
        self.tol_error.setSingleStep(0.01)
        self.tol_error.setDecimals(6)
        tol_layout.addWidget(self.tol_error)
        solve_layout.addLayout(tol_layout)
        
        # Element order
        order_layout = QHBoxLayout()
        order_layout.addWidget(QLabel("单元阶数:"))
        self.element_order_combo = QComboBox()
        self.element_order_combo.addItems(["使用当前阶数", "强制使用1阶", "强制使用2阶", "直接求解"])
        order_layout.addWidget(self.element_order_combo)
        solve_layout.addLayout(order_layout)
        
        # Solve button
        self.solve_btn = QPushButton("求解")
        self.solve_btn.clicked.connect(self.solve_fea)
        solve_layout.addWidget(self.solve_btn)
        
        # Log area
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        # Visualization controls
        visual_group = QGroupBox("可视化设置")
        visual_layout = QVBoxLayout(visual_group)
        
        # Surface selector
        surface_layout = QHBoxLayout()
        surface_layout.addWidget(QLabel("可视化表面:"))
        self.surface_visual_combo = QComboBox()
        surface_layout.addWidget(self.surface_visual_combo)
        visual_layout.addLayout(surface_layout)
        
        # Scale factor for displacement
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("位移放大系数:"))
        self.scale_factor = QDoubleSpinBox()
        self.scale_factor.setRange(0.1, 1000)
        self.scale_factor.setValue(1.0)
        self.scale_factor.setSingleStep(0.1)
        self.scale_factor.setDecimals(2)
        scale_layout.addWidget(self.scale_factor)
        visual_layout.addLayout(scale_layout)
        
        # Deformation type
        deform_layout = QHBoxLayout()
        deform_layout.addWidget(QLabel("变形类型:"))
        self.deform_type = QComboBox()
        self.deform_type.addItem("总变形", "total")
        self.deform_type.addItem("X方向", "x")
        self.deform_type.addItem("Y方向", "y")
        self.deform_type.addItem("Z方向", "z")
        deform_layout.addWidget(self.deform_type)
        visual_layout.addLayout(deform_layout)
        
        # Toggle deformation state - Replace dropdown with buttons
        deform_toggle_layout = QHBoxLayout()
        deform_toggle_layout.addWidget(QLabel("显示模式:"))
        
        # Create button group for deformation state
        self.deform_btn_group = QHBoxLayout()
        self.deformed_btn = QPushButton("变形后状态")
        self.deformed_btn.setCheckable(True)
        self.deformed_btn.setChecked(True)  # Default to deformed state
        self.deformed_btn.clicked.connect(lambda: self.set_deform_state("deformed"))
        
        self.undeformed_btn = QPushButton("变形前状态")
        self.undeformed_btn.setCheckable(True)
        self.undeformed_btn.clicked.connect(lambda: self.set_deform_state("undeformed"))
        
        self.deform_btn_group.addWidget(self.deformed_btn)
        self.deform_btn_group.addWidget(self.undeformed_btn)
        
        deform_toggle_layout.addLayout(self.deform_btn_group)
        visual_layout.addLayout(deform_toggle_layout)
        
        # Store current deformation state
        self.deform_state = "deformed"
        
        # Initially disable buttons until solving
        self.deformed_btn.setEnabled(False)
        self.undeformed_btn.setEnabled(False)
        
        # Export options (remove visualize button since it's replaced by the deform state buttons)
        export_layout = QHBoxLayout()
        self.save_img_btn = QPushButton("保存图像")
        self.save_img_btn.clicked.connect(self.save_visualization)
        self.export_stl_btn = QPushButton("导出STL")
        self.export_stl_btn.clicked.connect(self.export_to_stl)
        export_layout.addWidget(self.save_img_btn)
        export_layout.addWidget(self.export_stl_btn)
        visual_layout.addLayout(export_layout)
        
        # Create tabs for organizing UI
        tabs = QTabWidget()
        
        # Material properties tab
        material_tab = QWidget()
        material_layout = QVBoxLayout(material_tab)
        
        # Material properties group
        material_group = QGroupBox("材料属性设置")
        material_group_layout = QVBoxLayout(material_group)
        
        # Element selector
        element_selector_layout = QHBoxLayout()
        element_selector_layout.addWidget(QLabel("选择单元:"))
        self.element_selector_combo = QComboBox()
        self.element_selector_combo.currentIndexChanged.connect(self.update_material_properties)
        element_selector_layout.addWidget(self.element_selector_combo)
        material_group_layout.addLayout(element_selector_layout)
        
        # Material model selector
        material_model_layout = QHBoxLayout()
        material_model_layout.addWidget(QLabel("材料模型:"))
        self.material_model_combo = QComboBox()
        self.material_model_combo.addItem("Neo-Hookean", 1)
        self.material_model_combo.addItem("线弹性", 2)  # Linear elastic - added for future use
        self.material_model_combo.currentIndexChanged.connect(self.on_material_model_changed)
        material_model_layout.addWidget(self.material_model_combo)
        material_group_layout.addLayout(material_model_layout)
        
        # Create stacked widget for different material parameter inputs
        self.material_params_stack = QStackedWidget()
        
        # 1. Neo-Hookean parameters widget
        neo_hookean_widget = QWidget()
        neo_hookean_layout = QVBoxLayout(neo_hookean_widget)
        
        # mu (shear modulus)
        nh_mu_layout = QHBoxLayout()
        nh_mu_layout.addWidget(QLabel("剪切模量 μ:"))
        self.nh_mu_input = QDoubleSpinBox()
        self.nh_mu_input.setRange(0.00001, 1e9)
        self.nh_mu_input.setValue(10)
        self.nh_mu_input.setDecimals(12)  # Increased precision
        self.nh_mu_input.setSingleStep(1)
        nh_mu_layout.addWidget(self.nh_mu_input)
        neo_hookean_layout.addLayout(nh_mu_layout)
        
        # kappa (bulk modulus)
        nh_kappa_layout = QHBoxLayout()
        nh_kappa_layout.addWidget(QLabel("体积模量 κ:"))
        self.nh_kappa_input = QDoubleSpinBox()
        self.nh_kappa_input.setRange(0.00001, 1e9)
        self.nh_kappa_input.setValue(100)
        self.nh_kappa_input.setDecimals(12)  # Increased precision
        self.nh_kappa_input.setSingleStep(10)
        nh_kappa_layout.addWidget(self.nh_kappa_input)
        neo_hookean_layout.addLayout(nh_kappa_layout)
        
        # Add Neo-Hookean widget to stack
        self.material_params_stack.addWidget(neo_hookean_widget)
        
        # 2. Linear Elastic parameters widget
        linear_elastic_widget = QWidget()
        linear_elastic_layout = QVBoxLayout(linear_elastic_widget)
        
        # Young's modulus
        le_e_layout = QHBoxLayout()
        le_e_layout.addWidget(QLabel("弹性模量 E:"))
        self.le_e_input = QDoubleSpinBox()
        self.le_e_input.setRange(0.00001, 1e9)
        self.le_e_input.setValue(210000)
        self.le_e_input.setDecimals(12)  # Increased precision
        self.le_e_input.setSingleStep(1000)
        le_e_layout.addWidget(self.le_e_input)
        linear_elastic_layout.addLayout(le_e_layout)
        
        # Poisson's ratio
        le_nu_layout = QHBoxLayout()
        le_nu_layout.addWidget(QLabel("泊松比 ν:"))
        self.le_nu_input = QDoubleSpinBox()
        self.le_nu_input.setRange(0.00001, 0.499)  # Avoid 0.5 which is incompressible
        self.le_nu_input.setValue(0.3)
        self.le_nu_input.setDecimals(12)  # Increased precision
        self.le_nu_input.setSingleStep(0.01)
        le_nu_layout.addWidget(self.le_nu_input)
        linear_elastic_layout.addLayout(le_nu_layout)
        
        # Add Linear Elastic widget to stack
        self.material_params_stack.addWidget(linear_elastic_widget)
        
        # Add stacked widget to layout
        material_group_layout.addWidget(self.material_params_stack)
        
        # Density (common to all material models) - without unit
        density_layout = QHBoxLayout()
        density_layout.addWidget(QLabel("密度:"))
        self.density_input = QDoubleSpinBox()
        self.density_input.setRange(0.00001, 1e9)
        self.density_input.setValue(1000)
        self.density_input.setDecimals(12)  # Increased precision
        self.density_input.setSingleStep(100)
        density_layout.addWidget(self.density_input)
        material_group_layout.addLayout(density_layout)
        
        # Apply button
        self.apply_material_btn = QPushButton("应用材料属性")
        self.apply_material_btn.clicked.connect(self.apply_material_properties)
        material_group_layout.addWidget(self.apply_material_btn)
        
        # Current material properties display
        self.material_info_label = QLabel("当前未选择单元")
        material_group_layout.addWidget(self.material_info_label)
        
        material_layout.addWidget(material_group)
        
        # Add material tab as the first tab
        tabs.insertTab(0, material_tab, "材料属性")
        
        # Object manager tab
        obj_tab = QWidget()
        obj_layout = QVBoxLayout(obj_tab)
        # Expand the object manager to fill the tab
        obj_layout.addWidget(self.object_manager, stretch=1)
        tabs.addTab(obj_tab, "对象管理")
        
        # Solve tab
        solve_tab = QWidget()
        solve_tab_layout = QVBoxLayout(solve_tab)
        solve_tab_layout.addWidget(solve_group)
        solve_tab_layout.addWidget(visual_group)
        tabs.addTab(solve_tab, "求解与可视化")
        
        # Model tab
        model_tab = QWidget()
        model_tab_layout = QVBoxLayout(model_tab)
        model_tab_layout.addWidget(model_group)
        tabs.addTab(model_tab, "模型结构")
        
        # Add components to left layout
        left_layout.addWidget(file_group)
        left_layout.addWidget(tabs, stretch=1)
        left_layout.addWidget(log_group, stretch=0)
        
        # Right panel for visualization
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        self.visualization_widget = MayaviVisualization()
        right_layout.addWidget(self.visualization_widget)
        
        # Add panels to splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setSizes([400, 600])
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        
        # Disable UI elements until a file is loaded
        self.toggle_ui_enabled(False)
    
    def log(self, message):
        """Add message to log"""
        self.log_text.append(message)
        QApplication.processEvents()  # 确保UI更新
    
    def populate_elements(self):
        """Populate the element selector combobox with available elements"""
        self.element_selector_combo.clear()
        
        if not self.fe:
            return
            
        # Add elements to the combo box
        for elem_name, elem in self.fe.elems.items():
            self.element_selector_combo.addItem(f"{elem_name} ({elem.__class__.__name__})", elem_name)
        
        # If there are elements, select the first one and update material properties
        if self.element_selector_combo.count() > 0:
            self.update_material_properties(0)
            self.apply_material_btn.setEnabled(True)
            self.log("已加载单元列表")
        else:
            self.material_info_label.setText("未找到有效的单元")
            self.apply_material_btn.setEnabled(False)
    
    def update_material_properties(self, index):
        """Update material property inputs when an element is selected"""
        if index < 0 or not self.fe:
            return
            
        elem_name = self.element_selector_combo.itemData(index)
        elem = self.fe.elems.get(elem_name)
        
        if not elem:
            return
            
        # Get material information
        if hasattr(elem, 'materials'):
            try:
                # Check if it has Neo-Hookean parameters
                if hasattr(elem.materials, 'mu') and hasattr(elem.materials, 'kappa'):
                    # Get mu (shear modulus) and convert to MPa if it's in Pa
                    mu = float(elem.materials.mu.item())
                    if mu > 1e6:  # Likely in Pa, convert to MPa
                        mu = mu / 1e6
                    self.nh_mu_input.setValue(mu)
                    
                    # Get kappa (bulk modulus) and convert to MPa
                    kappa = float(elem.materials.kappa.item())
                    if kappa > 1e6:  # Likely in Pa, convert to MPa
                        kappa = kappa / 1e6
                    self.nh_kappa_input.setValue(kappa)
                    
                    # Get density if available
                    density = 1000.0  # Default value
                    if hasattr(elem.materials, 'rho'):
                        density = float(elem.materials.rho.item())
                    self.density_input.setValue(density)
                    
                    # Update info label
                    self.material_info_label.setText(
                        f"材料属性: {elem_name} (Neo-Hookean)\n"
                        f"剪切模量 μ: {mu:.3f} MPa\n"
                        f"体积模量 κ: {kappa:.3f} MPa\n"
                        f"密度: {density:.3f}"
                    )
                else:
                    self.material_info_label.setText(f"无法读取 {elem_name} 的材料属性")
                    
            except Exception as e:
                self.log(f"读取材料属性错误: {str(e)}")
                self.material_info_label.setText(f"读取 {elem_name} 的材料属性时出错")
        else:
            self.material_info_label.setText(f"单元 {elem_name} 没有关联的材料属性")
    
    def apply_material_properties(self):
        """Apply material properties to the selected element using initialize_materials"""
        if not self.fe:
            return
            
        idx = self.element_selector_combo.currentIndex()
        if idx < 0:
            self.log("未选择单元，无法应用材料属性")
            return
            
        elem_name = self.element_selector_combo.itemData(idx)
        elem = self.fe.elems.get(elem_name)
        
        if not elem:
            self.log(f"单元 {elem_name} 不存在")
            return
            
        try:
            # Get common values from UI
            density = self.density_input.value()
            
            # Get selected material model
            material_type = self.material_model_combo.currentData()
            material_name = self.material_model_combo.currentText()
            
            # Get material parameters based on material type
            if material_type == 1:  # Neo-Hookean
                # Get Neo-Hookean specific values directly - no unit conversion
                mu = self.nh_mu_input.value()
                kappa = self.nh_kappa_input.value()
                
                # Create materials parameter tensor for Neo-Hookean
                materials_params = torch.tensor([[mu, kappa]], dtype=torch.float32)
                
                # Use initialize_materials to create a new material object
                materials_now = FEA.elements.materials.initialize_materials(
                    materials_type=material_type,
                    materials_params=materials_params
                )
                
                # Set the material to the element
                elem.set_materials(materials_now)
                
                # Set density separately
                elem.set_density(density)
                
                # Log the update
                self.log(f"已更新 {elem_name} 的材料属性:\n"
                        f"材料模型: Neo-Hookean\n"
                        f"剪切模量 μ: {mu:.3f}\n"
                        f"体积模量 κ: {kappa:.3f}\n"
                        f"密度: {density}")
                
                # Update the info label
                self.material_info_label.setText(
                    f"材料属性: {elem_name} ({material_name})\n"
                    f"剪切模量 μ: {mu:.3f}\n"
                    f"体积模量 κ: {kappa:.3f}\n"
                    f"密度: {density:.3f}"
                )
                
            elif material_type == 2:  # Linear Elastic - placeholder for future implementation
                E = self.le_e_input.value()
                nu = self.le_nu_input.value()
                
                # Create materials parameter tensor for Linear Elastic
                materials_params = torch.tensor([[E, nu]], dtype=torch.float32)
                
                # Use initialize_materials to create a new material object
                materials_now = FEA.elements.materials.initialize_materials(
                    materials_type=material_type,
                    materials_params=materials_params
                )
                
                # Set the material to the element
                elem.set_materials(materials_now)
                
                # Set density separately
                elem.set_density(density)
                
                # Log the update
                self.log(f"已更新 {elem_name} 的材料属性:\n"
                        f"材料模型: 线弹性\n"
                        f"弹性模量 E: {E:.3f}\n"
                        f"泊松比 ν: {nu:.3f}\n"
                        f"密度: {density}")
                
                # Update the info label
                self.material_info_label.setText(
                    f"材料属性: {elem_name} ({material_name})\n"
                    f"弹性模量 E: {E:.3f}\n"
                    f"泊松比 ν: {nu:.3f}\n"
                    f"密度: {density:.3f}"
                )
                
            # Future material models can be added here as additional elif blocks
            
            else:
                self.log(f"不支持的材料模型类型: {material_type}")
                
        except Exception as e:
            import traceback
            self.log(f"应用材料属性错误: {str(e)}")
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"应用材料属性失败: {str(e)}")
    
    def toggle_ui_enabled(self, enabled):
        """Enable/disable UI elements based on model load status"""
        self.model_tree.setEnabled(enabled)
        self.object_manager.setEnabled(enabled)
        self.solve_btn.setEnabled(enabled)
        self.surface_visual_combo.setEnabled(enabled)
        self.scale_factor.setEnabled(enabled)
        self.deform_type.setEnabled(enabled)
        self.save_img_btn.setEnabled(enabled)
        self.export_stl_btn.setEnabled(enabled)
        self.element_order_combo.setEnabled(enabled)
        self.tol_error.setEnabled(enabled)
        
        # Material properties UI
        self.element_selector_combo.setEnabled(enabled)
        self.material_model_combo.setEnabled(enabled)
        self.nh_mu_input.setEnabled(enabled)
        self.nh_kappa_input.setEnabled(enabled)
        self.le_e_input.setEnabled(enabled)
        self.le_nu_input.setEnabled(enabled)
        self.density_input.setEnabled(enabled)
        self.apply_material_btn.setEnabled(enabled)
    
    def load_inp_file(self):
        """Load INP file for FEA"""
        options = QFileDialog.Options()
        self.inp_file, _ = QFileDialog.getOpenFileName(
            self, "打开INP文件", "", "INP Files (*.inp);;All Files (*)", options=options
        )
        
        if not self.inp_file:
            return
        
        # Display only the INP filename to prevent UI overflow and set tooltip for full path
        filename = os.path.basename(self.inp_file)
        self.file_path_label.setText(filename)
        self.file_path_label.setToolTip(self.inp_file)
        self.log(f"加载INP文件: {self.inp_file}")
        
        try:
            # Show progress dialog
            progress = QProgressDialog("加载INP文件中...", None, 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(10)
            progress.show()
            QApplication.processEvents()
            
            # Load INP file
            self.fem = FEA.FEA_INP()
            self.fem.Read_INP(self.inp_file)
            
            progress.setValue(50)
            QApplication.processEvents()
            
            # Create FE model
            self.fe = FEA.from_inp(self.fem)
            
            # Clear all existing constraints and loads (automatically)
            self.fe.constraints = {}
            self.fe.loads = {}
            self.log("已自动清除所有边界条件和载荷")
            
            progress.setValue(80)
            QApplication.processEvents()
            
            # Update UI
            self.update_model_tree()
            self.populate_surfaces()
            self.populate_elements()  # Populate elements in the new material property tab
            
            # Set models for object manager
            self.object_manager.set_models(self.fe, self.fem)
            self.object_manager.clear_all_objects()  # Clear the object manager UI as well
            
            progress.setValue(100)
            progress.close()
            
            # Enable UI elements
            self.toggle_ui_enabled(True)
            self.log("INP文件加载成功")
            
            # Display the geometric model immediately
            self.display_geometry()
            
        except Exception as e:
            self.log(f"加载INP文件错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"加载INP文件失败: {str(e)}")
            
    def display_geometry(self):
        """Display the geometric model (undeformed) after loading an INP file"""
        if not self.fe:
            return
            
        try:
            self.log("显示几何模型...")
            
            # Get undeformed coordinates
            undeformed = self.fe.nodes
            if torch.is_tensor(undeformed):
                undeformed = undeformed.detach().cpu().numpy().astype(np.float64)
            
            # Get surface elements - use all available surfaces
            try:
                extern_surf = self.fem.Find_Surface(['surface_0_All'])[1]
            except Exception as e:
                # Fallback: combine all surfaces from all parts
                try:
                    all_surfaces = []
                    for part_name, part in self.fem.part.items():
                        for surface_name, surface in part.surfaces_tri.items():
                            if isinstance(surface, (list, np.ndarray)) and len(surface) > 0:
                                all_surfaces.extend(surface)
                    if all_surfaces:
                        extern_surf = np.array(all_surfaces, dtype=int)
                    else:
                        # Last resort: use the first available surface
                        part_name = list(self.fem.part.keys())[0]
                        surface_name = list(self.fem.part[part_name].surfaces_tri.keys())[0]
                        extern_surf = self.fem.part[part_name].surfaces_tri[surface_name]
                except Exception as e2:
                    self.log(f"获取表面失败: {str(e2)}")
                    return
                    
            # Create a dummy scalar field for visualization
            scalar_data = np.zeros(undeformed.shape[0])
            
            # Display in visualization widget
            self.visualization_widget.visualize_fea(
                deformed_coords=undeformed,  # Use undeformed as "deformed" since we're showing the original geometry
                elements=extern_surf,
                scalar_data=scalar_data,
                log_func=self.log,
                scalar_label="几何模型"
            )
            
            self.log("几何模型显示成功")
            
        except Exception as e:
            self.log(f"显示几何模型错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
    
    def update_model_tree(self):
        """Update model tree with loaded model structure"""
        self.model_tree.clear()
        
        if not self.fe:
            return
        
        # 创建模型根节点
        model_root = QTreeWidgetItem(self.model_tree, ["模型", ""])
        
        # 添加有限元节点信息
        nodes_item = QTreeWidgetItem(model_root, ["节点", f"{self.fe.nodes.shape[0]} 节点"])
        
        # 添加单元信息
        elements_item = QTreeWidgetItem(model_root, ["单元", f"{len(self.fe.elems)} 组"])
        for elem_name, elem in self.fe.elems.items():
            elem_info = f"{elem.__class__.__name__}"
            QTreeWidgetItem(elements_item, [elem_name, elem_info])
        
        # 添加节点集
        node_sets_item = QTreeWidgetItem(model_root, ["节点集", f"{len(self.fe.node_sets)} 组"])
        for set_name, nodes in self.fe.node_sets.items():
            QTreeWidgetItem(node_sets_item, [set_name, f"{len(nodes)} 节点"])
        
        # 添加表面集
        surfaces_item = QTreeWidgetItem(model_root, ["表面", f"{len(self.fe.surface_sets)} 组"])
        for surface_name, elements in self.fe.surface_sets.items():
            QTreeWidgetItem(surfaces_item, [surface_name, f"{len(elements)} 单元"])
        
        # 添加单元集
        elem_sets_item = QTreeWidgetItem(model_root, ["单元集", f"{len(self.fe.element_sets)} 组"])
        for set_name, elems in self.fe.element_sets.items():
            QTreeWidgetItem(elem_sets_item, [set_name, f"{len(elems)} 单元"])
        
        # 添加参考点信息
        if hasattr(self.fe, 'reference_points') and self.fe.reference_points:
            rp_item = QTreeWidgetItem(model_root, ["参考点", f"{len(self.fe.reference_points)} 个"])
            for rp_name in self.fe.reference_points:
                QTreeWidgetItem(rp_item, [rp_name, ""])
        
        # 添加载荷信息
        if self.fe.loads:
            loads_item = QTreeWidgetItem(model_root, ["载荷", f"{len(self.fe.loads)} 个"])
            for load_name, load in self.fe.loads.items():
                load_info = f"{load.__class__.__name__}"
                QTreeWidgetItem(loads_item, [load_name, load_info])
        
        # 添加约束信息
        if self.fe.constraints:
            constraints_item = QTreeWidgetItem(model_root, ["约束", f"{len(self.fe.constraints)} 个"])
            for constraint_name, constraint in self.fe.constraints.items():
                constraint_info = f"{constraint.__class__.__name__}"
                QTreeWidgetItem(constraints_item, [constraint_name, constraint_info])
        
        # 展开树
        self.model_tree.expandToDepth(1)
    
    def populate_surfaces(self):
        """Populate surfaces combo box"""
        self.surface_visual_combo.clear()
        
        if not self.fe:
            return
        
        # 添加所有来自FEA_Main的表面集
        for name, surface in self.fe.surface_sets.items():
            self.surface_visual_combo.addItem(name, name)
        
        # 添加"所有表面"选项用于可视化
        if len(self.fe.surface_sets) > 0:
            self.surface_visual_combo.addItem("所有可用表面", "all")
    
    def on_object_added(self, obj_type, obj_instance):
        """Handle object added signal from object manager"""
        # Log addition and count for debugging
        self.log(f"已添加 {obj_type}")
        # Debug: show current count in list
        try:
            count = len(self.object_manager.objects.get(obj_type, []))
            self.log(f"[DEBUG] {obj_type} count: {count}")
            
            list_widget = self.object_manager._get_list_widget(obj_type)
            gui_count = list_widget.count() if list_widget else 'N/A'
            self.log(f"[DEBUG] GUI list count for {obj_type}: {gui_count}")
        except Exception as e:
            self.log(f"[DEBUG] 无法获取{obj_type}计数: {e}")
    
    def on_object_removed(self, obj_type, obj_name):
        """Handle object removed signal from object manager"""
        self.log(f"已删除 {obj_type}: {obj_name}")
        
        # Actually remove the object from the FEA model
        try:
            if obj_type == 'load' and self.fe:
                self.fe.delete_load(obj_name)
                self.log(f"成功从FEA模型中删除载荷: {obj_name}")
            elif obj_type == 'boundary_condition' and self.fe:
                self.fe.delete_constraint(obj_name)
                self.log(f"成功从FEA模型中删除边界条件: {obj_name}")
            elif obj_type == 'reference_point' and self.fe:
                self.fe.delete_reference_point(obj_name)
                self.log(f"成功从FEA模型中删除参考点: {obj_name}")
            elif obj_type == 'coupling' and self.fe:
                self.fe.delete_constraint(obj_name)
                self.log(f"成功从FEA模型中删除耦合约束: {obj_name}")
        except Exception as e:
            self.log(f"从FEA模型中删除{obj_type}时出错: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
    
    def on_object_modified(self, obj_type, obj_name, obj_instance):
        """Handle object modified signal from object manager"""
        # 可以根据需要在这里处理对象修改后的操作
        self.log(f"已修改 {obj_type}: {obj_name}")
    
    def process_mid_nodes(self):
        """Process mid-nodes for 2nd order elements"""
        try:
            mid_nodes_index = self.fe.elems['element-0'].get_2nd_order_point_index()
            self.fe.nodes[mid_nodes_index[:, 0]] = (self.fe.nodes[mid_nodes_index[:, 1]] + self.fe.nodes[mid_nodes_index[:, 2]]) / 2.0
            self.log("已处理二阶单元的中间节点")
            return True
        except Exception as e:
            self.log(f"注意: 无需处理中间节点或发生错误: {str(e)}")
            return False
    
    def solve_fea(self):
        """Solve FEA model"""
        if not self.fe:
            return
            
        try:
            self.log("开始FEA求解...")
            
            # Show progress dialog
            progress = QProgressDialog("FEA求解中...", None, 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()
            
            start_time = time.time()
            
            # Get tolerance
            tol_error = self.tol_error.value()
            
            # Always process mid-nodes for second-order elements
            progress.setValue(10)
            self.process_mid_nodes()
            
            # Handle element order based on selection
            order_selection = self.element_order_combo.currentIndex()
            
            progress.setValue(20)
            
            if order_selection == 0:  # Use current order
                if hasattr(self.fe.elems['element-0'], 'order'):
                    if self.fe.elems['element-0'].order == 1:
                        self.log("使用一阶单元求解")
                        progress.setValue(30)
                        self.fe.solve(tol_error=tol_error)
                        progress.setValue(80)
                    else:
                        self.log("使用二阶单元求解")
                        self.log("步骤1: 一阶求解")
                        self.fe.elems['element-0'].set_order(1)
                        progress.setValue(30)
                        self.fe.solve(tol_error=tol_error)
                        progress.setValue(50)
                        
                        self.log("步骤2: 二阶求解")
                        self.fe.elems['element-0'].set_order(2)
                        self.fe.refine_RGC()
                        progress.setValue(60)
                        self.fe.solve(RGC0=self.fe.RGC, tol_error=tol_error)
                        progress.setValue(80)
                else:
                    self.log("使用默认单元求解")
                    progress.setValue(30)
                    self.fe.solve(tol_error=tol_error)
                    progress.setValue(80)
            elif order_selection == 1:  # Force 1st order
                self.log("强制使用一阶单元求解")
                if hasattr(self.fe.elems['element-0'], 'set_order'):
                    self.fe.elems['element-0'].set_order(1)
                progress.setValue(30)
                self.fe.solve(tol_error=tol_error)
                progress.setValue(80)
            elif order_selection == 2:  # Force 2nd order
                self.log("强制使用二阶单元求解")
                if hasattr(self.fe.elems['element-0'], 'set_order'):
                    self.log("步骤1: 一阶求解")
                    self.fe.elems['element-0'].set_order(1)
                    progress.setValue(30)
                    self.fe.solve(tol_error=tol_error)
                    progress.setValue(50)
                    
                    self.log("步骤2: 二阶求解")
                    self.fe.elems['element-0'].set_order(2)
                    self.fe.refine_RGC()
                    progress.setValue(60)
                    self.fe.solve(RGC0=self.fe.RGC, tol_error=tol_error)
                    progress.setValue(80)
                else:
                    self.log("当前单元不支持二阶，使用默认单元求解")
                    progress.setValue(30)
                    self.fe.solve(tol_error=tol_error)
                    progress.setValue(80)
            elif order_selection == 3:  # Use 2nd order with mid-node processing
                self.log("直接求解二阶单元")
                progress.setValue(30)
                self.fe.solve(tol_error=tol_error)
                progress.setValue(80)
            
            end_time = time.time()
            self.log(f"FEA求解完成，耗时 {end_time - start_time:.2f} 秒")
            
            # Enable visualization buttons
            self.deformed_btn.setEnabled(True)
            self.undeformed_btn.setEnabled(True)
            self.save_img_btn.setEnabled(True)
            self.export_stl_btn.setEnabled(True)
            
            progress.setValue(100)
            progress.close()
            
        except Exception as e:
            import traceback
            self.log(f"FEA求解错误: {str(e)}")
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"FEA求解失败: {str(e)}")
    
    def visualize_results(self):
        """Visualize FEA results"""
        if not self.fe or not hasattr(self.fe, 'RGC'):
            self.log("没有FEA结果可以可视化")
            return
            
        try:
            # Show progress dialog
            progress = QProgressDialog("准备可视化...", None, 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()
            
            # Get displacement field
            U = self.fe.RGC[0]
            if torch.is_tensor(U):
                U = U.detach().cpu().numpy().astype(np.float64)
                
            progress.setValue(20)
            QApplication.processEvents()
            
            # Get undeformed coordinates
            undeformed = self.fe.nodes
            if torch.is_tensor(undeformed):
                undeformed = undeformed.detach().cpu().numpy().astype(np.float64)
            
            # Ensure dimensions match
            if undeformed.shape[1] != U.shape[1]:
                self.log(f"维度不匹配: undeformed {undeformed.shape}, U {U.shape}")
                if U.shape[1] == 3:
                    undeformed = undeformed[:, :3]
                else:
                    # Adjust displacement vector to match
                    U_fixed = np.zeros((U.shape[0], undeformed.shape[1]))
                    U_fixed[:, :min(U.shape[1], undeformed.shape[1])] = U[:, :min(U.shape[1], undeformed.shape[1])]
                    U = U_fixed
            
            # Apply scale factor to displacement
            scale = self.scale_factor.value()
            U_scaled = U * scale
            
            # Calculate deformed coordinates
            deformed = undeformed + U_scaled
            
            # Enable deformation state buttons after we have solved results
            self.deformed_btn.setEnabled(True)
            self.undeformed_btn.setEnabled(True)
            
            # Select which coordinates to display based on deform_state
            display_coords = deformed if self.deform_state == "deformed" else undeformed
            
            progress.setValue(40)
            QApplication.processEvents()
            
            # Get selected surface for visualization
            idx = self.surface_visual_combo.currentIndex()
            if idx < 0:
                self.log("未选择表面，无法可视化")
                progress.close()
                return
                
            surface_name = self.surface_visual_combo.itemData(idx)
            
            # Get surface elements
            extern_surf = None
            
            if surface_name == "all":
                # 组合所有表面，使用get_surface_triangles获取每个表面的三角面片
                try:
                    all_surfaces = []
                    for name in self.fe.surface_sets.keys():
                        try:
                            # 直接调用FEA_Main的get_surface_triangles函数获取三角面片
                            triangles = self.fe.get_surface_triangles(name)
                            all_surfaces.extend(triangles)
                            self.log(f"获取表面 {name} 的三角面片成功")
                        except Exception as e:
                            self.log(f"获取表面 {name} 的三角面片失败: {str(e)}")
                            
                    if all_surfaces:
                        extern_surf = all_surfaces
                        self.log(f"通过组合所有表面集获取到三角面片")
                    else:
                        raise ValueError("未找到有效的表面集")
                        
                except Exception as e:
                    self.log(f"组合所有表面失败: {str(e)}")
                    # 回退到第一个可用表面
                    if len(self.fe.surface_sets) > 0:
                        first_surface_name = next(iter(self.fe.surface_sets.keys()))
                        try:
                            extern_surf = self.fe.get_surface_triangles(first_surface_name)
                            self.log(f"回退到表面: {first_surface_name}")
                        except Exception as e2:
                            self.log(f"回退获取表面失败: {str(e2)}")
                            raise ValueError(f"无法获取表面信息: {str(e2)}")
                    else:
                        raise ValueError("无可用表面")
            else:
                # 直接使用选中的表面的get_surface_triangles函数
                if surface_name in self.fe.surface_sets:
                    try:
                        # 直接调用FEA_Main的get_surface_triangles函数获取三角面片
                        extern_surf = self.fe.get_surface_triangles(surface_name)
                        self.log(f"可视化表面: {surface_name}，使用get_surface_triangles函数获取三角面片")
                    except Exception as e:
                        self.log(f"获取表面 {surface_name} 的三角面片失败: {str(e)}")
                        raise ValueError(f"无法获取表面信息: {str(e)}")
                else:
                    raise ValueError(f"未找到表面: {surface_name}")
            
            progress.setValue(60)
            QApplication.processEvents()
            
            # 打印更详细的调试信息
            self.log(f"表面元素数据类型: {type(extern_surf)}")
            if isinstance(extern_surf, np.ndarray):
                self.log(f"表面元素形状: {extern_surf.shape}")
            elif isinstance(extern_surf, list):
                self.log(f"表面元素列表长度: {len(extern_surf)}")
                if len(extern_surf) > 0:
                    self.log(f"第一个元素类型: {type(extern_surf[0])}")
                
            # 确保数据有效
            if extern_surf is None or (hasattr(extern_surf, '__len__') and len(extern_surf) == 0):
                raise ValueError("无法获取有效的表面元素数据")
            
            # Calculate scalar data based on deformation type
            deform_type = self.deform_type.currentData()
            if deform_type == "total":
                scalar_data = np.sqrt(np.sum(U**2, axis=1))
                scalar_label = "总变形量"
            elif deform_type == "x":
                scalar_data = np.abs(U[:, 0])
                scalar_label = "X方向变形"
            elif deform_type == "y":
                scalar_data = np.abs(U[:, 1])
                scalar_label = "Y方向变形"
            elif deform_type == "z":
                scalar_data = np.abs(U[:, 2])
                scalar_label = "Z方向变形"
                
            # If showing undeformed state, still calculate scalar field but display undeformed coordinates
            if self.deform_state == "undeformed":
                scalar_label += " (变形前状态)"
            else:
                scalar_label += " (变形后状态)"
            
            progress.setValue(80)
            QApplication.processEvents()
            
            # Visualize - use display_coords which is either deformed or undeformed based on toggle
            success = self.visualization_widget.visualize_fea(
                deformed_coords=display_coords,
                elements=extern_surf,
                scalar_data=scalar_data,
                log_func=self.log,
                scalar_label=scalar_label
            )
            
            if success:
                self.log(f"可视化更新完成 - {scalar_label}")
            
            progress.setValue(100)
            progress.close()
            
        except Exception as e:
            import traceback
            self.log(f"可视化错误: {str(e)}")
            self.log(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"可视化失败: {str(e)}")
            
            if 'progress' in locals() and progress:
                progress.close()
    
    def save_visualization(self):
        """Save current visualization as an image"""
        if not hasattr(self, 'visualization_widget') or not self.visualization_widget:
            QMessageBox.warning(self, "保存图像", "没有可用的可视化窗口")
            return
            
        try:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self, "保存图像", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)", options=options
            )
            
            if file_name:
                # Add file extension if not provided
                if not (file_name.lower().endswith('.png') or file_name.lower().endswith('.jpg')):
                    file_name += '.png'
                
                self.log(f"正在保存图像到: {file_name}")
                success = self.visualization_widget.save_screenshot(file_name)
                
                if success:
                    self.log("图像保存成功")
                    QMessageBox.information(self, "保存图像", f"图像已保存到:\n{file_name}")
                else:
                    raise Exception("保存图像失败")
        except Exception as e:
            self.log(f"保存图像错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"保存图像失败: {str(e)}")
    
    def export_to_stl(self):
        """Export current visualization model to STL file"""
        if not hasattr(self, 'visualization_widget') or not self.visualization_widget:
            QMessageBox.warning(self, "导出STL", "没有可用的可视化窗口")
            return
            
        try:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self, "导出STL", "", "STL Files (*.stl);;All Files (*)", options=options
            )
            
            if file_name:
                # Add file extension if not provided
                if not file_name.lower().endswith('.stl'):
                    file_name += '.stl'
                
                self.log(f"正在导出STL到: {file_name}")
                success = self.visualization_widget.export_stl(file_name)
                
                if success:
                    self.log("STL导出成功")
                    QMessageBox.information(self, "导出STL", f"模型已导出到:\n{file_name}")
                else:
                    raise Exception("导出STL失败")
        except Exception as e:
            self.log(f"导出STL错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"导出STL失败: {str(e)}")
    
    def set_deform_state(self, state):
        """Set the deformation state and update button states"""
        # Update the internal state
        self.deform_state = state
        
        # Update button appearance
        self.deformed_btn.setChecked(state == "deformed")
        self.undeformed_btn.setChecked(state == "undeformed")
        
        # Re-visualize if we already have results
        if hasattr(self.fe, 'RGC'):
            self.log(f"切换到{('变形后' if state == 'deformed' else '变形前')}状态")
            self.visualize_results()
    
    def on_material_model_changed(self, index):
        """Handle material model selection change"""
        if index < 0:
            return
            
        material_type = self.material_model_combo.itemData(index)
        material_name = self.material_model_combo.currentText()
        
        # Show the appropriate parameter input panel based on material type
        if material_type == 1:  # Neo-Hookean
            self.material_params_stack.setCurrentIndex(0)
            self.log(f"已选择材料模型: {material_name} (使用剪切模量μ和体积模量κ)")
        elif material_type == 2:  # Linear Elastic
            self.material_params_stack.setCurrentIndex(1)
            self.log(f"已选择材料模型: {material_name} (使用弹性模量E和泊松比ν)")
        
        # Update the element material if one is selected
        elem_idx = self.element_selector_combo.currentIndex()
        if elem_idx >= 0:
            self.update_material_properties(elem_idx)