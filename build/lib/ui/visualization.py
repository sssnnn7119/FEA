"""
Visualization module for FEA results using Mayavi
"""

import os
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QMessageBox, QApplication
from PyQt5.QtCore import QTimer
import torch

# Import Mayavi modules
try:
    from mayavi import mlab
    from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
    from traits.api import HasTraits, Instance
    from traitsui.api import View, Item
    from tvtk.api import tvtk
    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False
    print("Warning: Mayavi not available, visualization features will be disabled.")


if MAYAVI_AVAILABLE:
    class MayaviQWidget(QWidget):
        """Qt widget to hold the Mayavi visualization"""
        
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            
            # Mayavi visualization
            self.visualization = MayaviVisualizer()
            self.ui = self.visualization.edit_traits(
                parent=self, kind='subpanel').control
            layout.addWidget(self.ui)
            self.layout = layout
        
        def get_scene(self):
            return self.visualization.scene

        def get_mlab(self):
            return self.visualization.scene.mlab
    
    class MayaviVisualizer(HasTraits):
        """Mayavi scene wrapper"""
        
        scene = Instance(MlabSceneModel, ())
        
        view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                       height=400, width=600, show_label=False),
                  resizable=True)
        
        def __init__(self):
            super().__init__()
            # Initialize with background only
            # Don't access scene yet, defer until Qt main thread is ready
            self._scene_initialized = False
        
        def _initialize_scene(self):
            """Initialize the scene in the main thread when ready"""
            if not self._scene_initialized:
                self.scene.background = (1, 1, 1)  # white background
                # Set text color to black for better visibility against white background
                self.scene.foreground = (0, 0, 0)  # black text
                self.scene.mlab.clf()
                self._scene_initialized = True
                
        def start_scene(self):
            """Start scene in the Qt main thread context"""
            # Use QTimer to ensure this runs in the main thread
            QTimer.singleShot(0, self._initialize_scene)


class MayaviVisualization(QWidget):
    """Main visualization widget that contains Mayavi"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        if MAYAVI_AVAILABLE:
            try:
                self.mayavi_widget = MayaviQWidget(self)
                layout.addWidget(self.mayavi_widget)
                self.has_mayavi = True
                
                # Initialize the scene in the Qt main thread to avoid QBasicTimer warning
                QTimer.singleShot(0, self._initialize_scene)
                
                # Store current visualization data
                self.current_data = {
                    'deformed_coords': None,
                    'elements': None,
                    'scalar_data': None
                }
            except Exception as e:
                print(f"Error initializing Mayavi: {e}")
                layout.addWidget(QLabel("Error initializing Mayavi visualization."))
                self.has_mayavi = False
        else:
            layout.addWidget(QLabel("Mayavi not available. Please install mayavi to enable visualization."))
            self.has_mayavi = False
    
    def _initialize_scene(self):
        """Initialize the scene in the main thread"""
        if self.has_mayavi:
            try:
                self.mayavi_widget.visualization.start_scene()
            except Exception as e:
                print(f"Error initializing Mayavi scene: {e}")
                
    def _prepare_triangulation(self, elements):
        """
        Prepare triangulation for visualization.
        Converts different element types to triangles for surface rendering.
        Only processes face connectivity (surface triangles), not volume elements.
        
        Args:
            elements: Surface element connectivity (triangles, quads, etc.)
            
        Returns:
            Triangulated elements as numpy array
        """
        if not isinstance(elements, (list, np.ndarray, tuple)):
            return np.array([], dtype=int)
            
        # Handle special case of list of torch tensors (from get_surface_triangles)
        if isinstance(elements, list) and elements and torch.is_tensor(elements[0]):
            all_triangles = []
            for tensor in elements:
                # Convert tensor to numpy array
                triangles_array = tensor.detach().cpu().numpy()
                all_triangles.append(triangles_array)
            
            if all_triangles:
                return np.vstack(all_triangles)
            else:
                return np.array([], dtype=int)
        
        # Handle list of arrays
        if isinstance(elements, list):
            if not elements:
                return np.array([], dtype=int)
                
            # Convert list of faces to numpy array
            triangles = []
            for elem in elements:
                if len(elem) == 3:  # Already a triangle
                    triangles.append(list(elem))
                elif len(elem) == 4:  # Quad to 2 triangles
                    triangles.append([elem[0], elem[1], elem[2]])
                    triangles.append([elem[0], elem[2], elem[3]])
                
            if triangles:
                return np.array(triangles, dtype=int)
            else:
                return np.array([], dtype=int)
                
        # If elements is already a numpy array
        elif isinstance(elements, np.ndarray):
            if len(elements) == 0:
                return np.array([], dtype=int)
                
            # If it's a 2D array of quads (4 vertices)
            if len(elements.shape) == 2 and elements.shape[1] == 4:
                triangles = []
                for i in range(elements.shape[0]):
                    # Convert quad to triangles
                    triangles.append([elements[i, 0], elements[i, 1], elements[i, 2]])
                    triangles.append([elements[i, 0], elements[i, 2], elements[i, 3]])
                return np.array(triangles, dtype=int)
            
            # Already triangles or other face elements
            return elements
        
        return np.array([], dtype=int)
    
    def _validate_data(self, coords, elements, scalar_data=None):
        """Validate input data for visualization"""
        # Validate coordinates
        if not isinstance(coords, np.ndarray) or len(coords) == 0:
            return False, "No node coordinates provided"
            
        # Make sure coordinates have 3 dimensions (x, y, z)
        if coords.shape[1] < 3:
            return False, f"Coordinates must have 3 dimensions, got {coords.shape[1]}"
            
        # Validate elements after triangulation
        if len(elements) == 0:
            return False, "No elements provided for visualization"
            
        # Check if all indices are valid
        max_node_idx = coords.shape[0] - 1
        all_indices = elements.flatten()
        if np.any(all_indices < 0) or np.any(all_indices > max_node_idx):
            # Filter out invalid elements
            valid_elements = []
            for elem in elements:
                if np.all(np.array(elem) <= max_node_idx) and np.all(np.array(elem) >= 0):
                    valid_elements.append(elem)
            
            if not valid_elements:
                return False, f"All element indices are out of bounds (max index: {max_node_idx})"
                
            elements = np.array(valid_elements, dtype=int)
            
        # If scalar data provided, validate it
        if scalar_data is not None:
            if len(scalar_data) != coords.shape[0]:
                return False, f"Scalar data length ({len(scalar_data)}) doesn't match number of nodes ({coords.shape[0]})"
        
        return True, elements
    
    def visualize_fea(self, deformed_coords, elements, scalar_data=None, log_func=print, scalar_label="Displacement Magnitude"):
        """
        Visualize FEA results
        
        Args:
            deformed_coords: Deformed nodal coordinates (n_nodes, 3)
            elements: Element connectivity for visualization (n_elems, n_nodes_per_elem)
            scalar_data: Scalar data for coloring (n_nodes,)
            log_func: Function to use for logging (defaults to print)
            scalar_label: Label for the scalar data colorbar (defaults to "Displacement Magnitude")
        """
        if not MAYAVI_AVAILABLE or not self.has_mayavi:
            log_func("Mayavi visualization not available")
            QMessageBox.warning(self, "Visualization Error", "Mayavi visualization not available")
            return False
            
        try:
            # Convert inputs to numpy arrays with proper data types
            deformed_coords = np.asarray(deformed_coords, dtype=float)
            
            # Process elements to ensure they're all triangles
            tri_elements = self._prepare_triangulation(elements)
            
            # Validate data
            is_valid, result = self._validate_data(deformed_coords, tri_elements, scalar_data)
            
            if not is_valid:
                log_func(f"Visualization data validation failed: {result}")
                QMessageBox.warning(self, "Visualization Error", f"Data validation failed: {result}")
                return False
                
            # If result contains processed elements, update tri_elements
            if isinstance(result, np.ndarray):
                tri_elements = result
            
            # Store current visualization data for later use (e.g., export)
            self.current_data['deformed_coords'] = deformed_coords
            self.current_data['elements'] = tri_elements
            self.current_data['scalar_data'] = scalar_data
            
            # Clear previous visualization
            if hasattr(self, 'mayavi_widget') and self.mayavi_widget:
                self.mayavi_widget.get_scene().mlab.clf()
            
            # Transpose coordinates for Mayavi (expects x, y, z arrays)
            x = deformed_coords[:, 0].astype(float)
            y = deformed_coords[:, 1].astype(float)
            z = deformed_coords[:, 2].astype(float)
            
            # Create the visualization
            if scalar_data is not None:
                # Process scalar data
                scalar_data = np.asarray(scalar_data, dtype=float)
                
                # Handle NaN and Inf values
                scalar_data = np.nan_to_num(scalar_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Create the mesh with scalar data
                mesh = self.mayavi_widget.get_mlab().triangular_mesh(
                    x, y, z, tri_elements, scalars=scalar_data, colormap='viridis')
                    
                # Add a colorbar with black text
                cb = self.mayavi_widget.get_mlab().colorbar(mesh, orientation='vertical', title=scalar_label)
                # Ensure colorbar text is visible against white background
                if hasattr(cb, 'title_text_property'):
                    cb.title_text_property.color = (0.0, 0.0, 0.0)  # black
                if hasattr(cb, 'label_text_property'):
                    cb.label_text_property.color = (0.0, 0.0, 0.0)  # black
            else:
                # Create the mesh without scalar data
                mesh = self.mayavi_widget.get_mlab().triangular_mesh(
                    x, y, z, tri_elements, color=(0.8, 0.8, 0.8))
            
            # Add axes with black text
            ax = self.mayavi_widget.get_mlab().axes()
            # Set axes text color to black for better visibility
            if hasattr(ax, 'axes'):
                if hasattr(ax.axes, 'x_label_property'):
                    ax.axes.x_label_property.color = (0.0, 0.0, 0.0)
                if hasattr(ax.axes, 'y_label_property'):
                    ax.axes.y_label_property.color = (0.0, 0.0, 0.0)
                if hasattr(ax.axes, 'z_label_property'):
                    ax.axes.z_label_property.color = (0.0, 0.0, 0.0)
                if hasattr(ax.axes, 'x_axis_property'):
                    ax.axes.x_axis_property.color = (0.0, 0.0, 0.0)
                if hasattr(ax.axes, 'y_axis_property'):
                    ax.axes.y_axis_property.color = (0.0, 0.0, 0.0)
                if hasattr(ax.axes, 'z_axis_property'):
                    ax.axes.z_axis_property.color = (0.0, 0.0, 0.0)
            
            log_func("Visualization completed successfully")
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"Error in visualization: {str(e)}"
            log_func(error_msg)
            log_func(traceback.format_exc())
            QMessageBox.critical(self, "Visualization Error", error_msg)
            return False
    
    def save_screenshot(self, filename):
        """
        Save current visualization as an image
        
        Args:
            filename: Output filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not MAYAVI_AVAILABLE or not self.has_mayavi:
            print("Mayavi visualization not available")
            return False
            
        try:
            # Make sure we have a mayavi scene
            if not hasattr(self, 'mayavi_widget') or not self.mayavi_widget:
                print("No visualization widget available")
                return False
                
            # Save the figure
            self.mayavi_widget.get_mlab().savefig(filename)
            print(f"Screenshot saved to {filename}")
            return True
        except Exception as e:
            import traceback
            print(f"Error saving screenshot: {e}")
            print(traceback.format_exc())
            return False
    
    def export_stl(self, filename):
        """
        Export visualization mesh to STL file
        
        Args:
            filename: Output STL filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not MAYAVI_AVAILABLE or not self.has_mayavi:
            print("Mayavi visualization not available")
            return False
            
        try:
            # Make sure we have a visualization
            if not hasattr(self, 'mayavi_widget') or not self.mayavi_widget:
                print("No visualization widget available")
                return False
                
            # Get the current scene
            scene = self.mayavi_widget.get_scene()
            mlab = self.mayavi_widget.get_mlab()
            
            if not scene or not mlab:
                print("No scene available")
                return False
            
            # Get the active dataset from the current scene or use stored data
            if hasattr(scene, 'renderer') and scene.renderer.actors:
                # Get from current scene
                dataset = None
                for actor in scene.renderer.actors:
                    if hasattr(actor, 'mapper') and hasattr(actor.mapper, 'input'):
                        dataset = actor.mapper.input
                        break
                
                if not dataset:
                    raise Exception("Could not get dataset from scene")
            else:
                # Create from stored data if we have it
                if (self.current_data['deformed_coords'] is not None and 
                    self.current_data['elements'] is not None):
                    
                    coords = self.current_data['deformed_coords']
                    elements = self.current_data['elements']
                    
                    # Create a new mesh
                    x = coords[:, 0].astype(float)
                    y = coords[:, 1].astype(float)
                    z = coords[:, 2].astype(float)
                    
                    # Create a temporary mesh to get a dataset
                    temp_mesh = mlab.triangular_mesh(x, y, z, elements)
                    dataset = temp_mesh.mlab_source.dataset
                else:
                    raise Exception("No visualization data available")
            
            # Create a writer and write the STL file
            writer = tvtk.STLWriter()
            writer.file_name = filename
            writer.set_input_data(dataset)
            writer.write()
            
            print(f"STL file exported to {filename}")
            return True
            
        except Exception as e:
            import traceback
            print(f"Error exporting STL: {e}")
            print(traceback.format_exc())
            return False