import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QTreeView, QFileSystemModel, QTabWidget, QLabel,
    QFileDialog, QLineEdit, QMessageBox, QFrame, QSplitter, QTextEdit, QSizePolicy,
    QDesktopWidget # For screen geometry
)
from PyQt5.QtCore import Qt, QDir, QModelIndex
from PyQt5.QtGui import QFont, QIntValidator # For font and integer validation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec # For more flexible subplot layouts

# --- Data Loading and Plotting ---
class DataPlotter:
    def __init__(self):
        # Increased figsize for better visibility
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = None # Generic axis for this plotter instance
        self.cbar = None # To store the colorbar object for 2D plots

    def create_axes(self):
        """Clears figure and creates a single new axis."""
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.figure.tight_layout() # Apply tight_layout after creating axes
        self.cbar = None # Reset colorbar reference

    def plot_image(self, data, x_coords, y_coords, title="Image", xlabel="X-axis", ylabel="Y-axis", cbar_label="Value"):
        self.create_axes() # Ensure single axis for image
        # Changed colormap to 'turbo_r' as requested
        im = self.ax.imshow(data, aspect='auto', origin='lower',
                                extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
                                cmap='turbo_r') # Apply turbo_r colormap
        self.cbar = self.figure.colorbar(im, ax=self.ax)
        self.cbar.set_label(cbar_label)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.figure.tight_layout() # Apply tight_layout after plotting
        self.canvas.draw()

    def plot_line(self, x_data, y_data, title="Line Plot", xlabel="X-axis", ylabel="Y-axis", y2_data=None, y2_label=None):
        """
        Plots a 1D line plot. Can optionally plot a second Y-axis using twinx.
        """
        self.create_axes() # Ensure single axis for line plot
        self.ax.plot(x_data, y_data, label=ylabel) # Add label for legend
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True, linestyle='--', alpha=0.6) # Add dashed and transparent grid to 1D plots

        if y2_data is not None:
            ax2 = self.ax.twinx()
            ax2.plot(x_data, y2_data, 'r-', label=y2_label) # Plot y2 in red, solid line
            ax2.set_ylabel(y2_label)
            # Combine legends from both axes
            lines, labels = self.ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            self.ax.legend(lines + lines2, labels + labels2, loc='best')
        else:
            self.ax.legend(loc='best') # Only show legend for single plot if no y2_data

        self.ax.autoscale_view() # Autoscale for 1D plots
        self.figure.tight_layout() # Apply tight_layout after plotting
        self.canvas.draw()

    def clear_plot(self):
        self.figure.clear()
        self.canvas.draw()
        self.ax = None
        self.cbar = None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Data Dashboard")

        # Set initial window size based on screen geometry
        screen = QApplication.desktop().screenGeometry()
        self.setGeometry(
            int(screen.width() * 0.1),  # X position (10% from left)
            int(screen.height() * 0.1), # Y position (10% from top)
            int(screen.width() * 0.8),  # Width (80% of screen width)
            int(screen.height() * 0.8)  # Height (80% of screen height)
        )

        self.data_2d_loaded = None # To store loaded 2D data matrix
        self.x_coords_2d = None # X coordinates for 2D data
        self.y_coords_2d = None # Y coordinates for 2D data
        self.current_2d_file_path = None # Store path to re-plot with new titles

        self.data_1d_loaded_x = None # To store loaded 1D X data
        self.data_1d_loaded_y1 = None # To store loaded 1D Y1 data
        self.data_1d_loaded_y2 = None # To store loaded 1D Y2 data (optional)
        self.current_1d_file_path = None # Store path to re-plot with new titles

        self.current_x_cut_idx = 0
        self.current_y_cut_idx = 0

        # Store lambda handlers to allow proper disconnect/reconnect
        self._y_cut_text_changed_handler = None
        self._x_cut_text_changed_handler = None

        self.current_theme = "light" # Default theme is light

        self.init_ui()
        self.apply_stylesheet(self.current_theme) # Apply initial stylesheet

    def apply_stylesheet(self, theme):
        # Define light theme stylesheet
        light_stylesheet = """
            QMainWindow {
                background-color: #f0f0f0; /* Light background */
                color: #333333; /* Dark text */
            }
            QWidget {
                background-color: #ffffff; /* White for panels */
                color: #333333;
                font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
                font-size: 10pt;
            }
            QLabel {
                color: #333333;
                padding: 2px;
            }
            QLabel#titleLabel { /* Specific style for titles */
                font-size: 12pt;
                font-weight: bold;
                color: #0078d7; /* Standard blue for titles */
            }
            QPushButton {
                background-color: #e0e0e0;
                color: #333333;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 8px 15px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
                border-color: #bbbbbb;
            }
            QPushButton:pressed {
                background-color: #c0c0c0;
            }
            QLineEdit, QTextEdit {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 5px;
            }
            QTreeView {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #cccccc;
                border-radius: 5px;
                padding: 5px;
                selection-background-color: #aaddff; /* Light blue for selection */
                selection-color: #333333;
            }
            QTabWidget::pane { /* The tab content frame */
                border: 1px solid #cccccc;
                background-color: #f0f0f0;
                border-radius: 5px;
            }
            QTabBar::tab {
                background: #e0e0e0;
                color: #333333;
                border: 1px solid #e0e0e0;
                border-bottom-color: #f0f0f0; /* Same as pane color */
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 15px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #f0f0f0; /* Same as pane color */
                border-color: #cccccc;
                border-bottom-color: #f0f0f0; /* Same as pane color */
            }
            QTabBar::tab:hover {
                background: #d0d0d0;
            }
            QFrame {
                border: 1px solid #cccccc;
                border-radius: 7px;
                background-color: #e8e8e8; /* Slightly different shade for frames */
                margin: 5px;
                padding: 5px;
            }
        """

        # Define dark theme stylesheet (current style)
        dark_stylesheet = """
            QMainWindow {
                background-color: #2e2e2e; /* Dark background */
                color: #e0e0e0; /* Light text */
            }
            QWidget {
                background-color: #3c3c3c; /* Slightly lighter dark for panels */
                color: #e0e0e0;
                font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
                font-size: 10pt;
            }
            QLabel {
                color: #e0e0e0;
                padding: 2px;
            }
            QLabel#titleLabel { /* Specific style for titles */
                font-size: 12pt;
                font-weight: bold;
                color: #87ceeb; /* Sky blue for titles */
            }
            QPushButton {
                background-color: #555555;
                color: #ffffff;
                border: 1px solid #666666;
                border-radius: 5px;
                padding: 8px 15px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #6a6a6a;
                border-color: #888888;
            }
            QPushButton:pressed {
                background-color: #4a4a4a;
            }
            QLineEdit, QTextEdit {
                background-color: #4a4a4a;
                color: #e0e0e0;
                border: 1px solid #666666;
                border-radius: 5px;
                padding: 5px;
            }
            QTreeView {
                background-color: #4a4a4a;
                color: #e0e0e0;
                border: 1px solid #666666;
                border-radius: 5px;
                padding: 5px;
                selection-background-color: #0078d7; /* Windows blue for selection */
                selection-color: #ffffff;
            }
            QTabWidget::pane { /* The tab content frame */
                border: 1px solid #555555;
                background-color: #3c3c3c;
                border-radius: 5px;
            }
            QTabBar::tab {
                background: #555555;
                color: #e0e0e0;
                border: 1px solid #555555;
                border-bottom-color: #3c3c3c; /* Same as pane color */
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 15px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #3c3c3c; /* Same as pane color */
                border-color: #555555;
                border-bottom-color: #3c3c3c; /* Same as pane color */
            }
            QTabBar::tab:hover {
                background: #6a6a6a;
            }
            QFrame {
                border: 1px solid #555555;
                border-radius: 7px;
                background-color: #424242; /* Slightly different shade for frames */
                margin: 5px;
                padding: 5px;
            }
        """
        if theme == "light":
            self.setStyleSheet(light_stylesheet)
        else:
            self.setStyleSheet(dark_stylesheet)

        # Set a global font for better consistency
        font = QFont("Segoe UI", 10)
        QApplication.setFont(font)

    def toggle_theme(self):
        if self.current_theme == "light":
            self.current_theme = "dark"
        else:
            self.current_theme = "light"
        self.apply_stylesheet(self.current_theme)


    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Overall vertical layout for the central widget
        overall_v_layout = QVBoxLayout(central_widget)

        # Top bar for theme button
        top_bar_layout = QHBoxLayout()
        top_bar_layout.addStretch() # Push button to the right
        self.theme_toggle_button = QPushButton("Toggle Theme")
        self.theme_toggle_button.clicked.connect(self.toggle_theme)
        top_bar_layout.addWidget(self.theme_toggle_button)
        overall_v_layout.addLayout(top_bar_layout) # Add top bar to the overall vertical layout

        # Main content horizontal layout (formerly main_layout)
        main_content_h_layout = QHBoxLayout() 

        # --- Left Panel: Data Selection ---
        left_panel_splitter = QSplitter(Qt.Vertical) # Split left panel for 2D and 1D data
        # Reduced width of the data loader file system panel
        left_panel_splitter.setFixedWidth(400) # Changed from 150 to 100

        # --- 2D Data Selection Section ---
        _2d_selection_widget = QWidget()
        _2d_selection_layout = QVBoxLayout(_2d_selection_widget)
        _2d_selection_layout.addWidget(QLabel("<h2>Load Folder: 2D Plot</h2>", objectName="titleLabel"))

        _2d_root_folder_layout = QHBoxLayout()
        self.root_folder_2d_path_display = QLineEdit("No 2D folder selected")
        self.root_folder_2d_path_display.setReadOnly(True)
        self.browse_root_2d_button = QPushButton("Browse 2D Root")
        self.browse_root_2d_button.clicked.connect(lambda: self.select_root_folder(is_2d=True))
        _2d_root_folder_layout.addWidget(self.root_folder_2d_path_display)
        _2d_root_folder_layout.addWidget(self.browse_root_2d_button)
        _2d_selection_layout.addLayout(_2d_root_folder_layout)

        self.folder_model_2d = QFileSystemModel()
        self.folder_model_2d.setFilter(QDir.Dirs | QDir.Files | QDir.NoDotAndDotDot)
        self.tree_view_2d = QTreeView()
        self.tree_view_2d.setModel(self.folder_model_2d)
        self.tree_view_2d.setHeaderHidden(True)
        for i in range(1, self.folder_model_2d.columnCount()):
            self.tree_view_2d.hideColumn(i)
        self.tree_view_2d.clicked.connect(lambda index: self.on_tree_view_clicked(index, is_2d=True))
        _2d_selection_layout.addWidget(self.tree_view_2d)
        _2d_selection_layout.addStretch()

        # --- 1D Data Selection Section ---
        _1d_selection_widget = QWidget()
        _1d_selection_layout = QVBoxLayout(_1d_selection_widget)
        _1d_selection_layout.addWidget(QLabel("<h2>Load Folder: 1D Plot </h2>", objectName="titleLabel"))

        _1d_root_folder_layout = QHBoxLayout()
        self.root_folder_1d_path_display = QLineEdit("No 1D folder selected")
        self.root_folder_1d_path_display.setReadOnly(True)
        self.browse_root_1d_button = QPushButton("Browse 1D Root")
        self.browse_root_1d_button.clicked.connect(lambda: self.select_root_folder(is_2d=False))
        _1d_root_folder_layout.addWidget(self.root_folder_1d_path_display)
        _1d_root_folder_layout.addWidget(self.browse_root_1d_button)
        _1d_selection_layout.addLayout(_1d_root_folder_layout)

        self.folder_model_1d = QFileSystemModel()
        self.folder_model_1d.setFilter(QDir.Dirs | QDir.Files | QDir.NoDotAndDotDot)
        self.tree_view_1d = QTreeView()
        self.tree_view_1d.setModel(self.folder_model_1d)
        self.tree_view_1d.setHeaderHidden(True)
        for i in range(1, self.folder_model_1d.columnCount()):
            self.tree_view_1d.hideColumn(i)
        self.tree_view_1d.clicked.connect(lambda index: self.on_tree_view_clicked(index, is_2d=False))
        _1d_selection_layout.addWidget(self.tree_view_1d)
        _1d_selection_layout.addStretch()

        left_panel_splitter.addWidget(_2d_selection_widget)
        left_panel_splitter.addWidget(_1d_selection_widget)

        # --- Right Panel: Plots and Controls ---
        right_panel = QVBoxLayout()
        self.tab_widget = QTabWidget()

        # --- 2D Data Plot Tab (Modified Layout) ---
        self.plot_2d_widget = QWidget()
        self.plot_2d_grid_layout = QGridLayout(self.plot_2d_widget) # Use QGridLayout for 2D plots and controls

        # Initialize separate DataPlotter instances for each 2D plot
        self.data_plotter_2d_main = DataPlotter()
        self.data_plotter_2d_x_cut = DataPlotter()
        self.data_plotter_2d_y_cut = DataPlotter()

        # Set size policy for canvases to expand
        self.data_plotter_2d_main.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.data_plotter_2d_x_cut.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.data_plotter_2d_y_cut.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        # Place plot canvases in the grid (Row 0)
        self.plot_2d_grid_layout.addWidget(self.data_plotter_2d_main.canvas, 0, 0)
        self.plot_2d_grid_layout.addWidget(self.data_plotter_2d_x_cut.canvas, 0, 1)
        self.plot_2d_grid_layout.addWidget(self.data_plotter_2d_y_cut.canvas, 0, 2)

        # Set column stretch factors to make plots same size
        self.plot_2d_grid_layout.setColumnStretch(0, 1)
        self.plot_2d_grid_layout.setColumnStretch(1, 1)
        self.plot_2d_grid_layout.setColumnStretch(2, 1)

        # Add Matplotlib Toolbars below each plot (Row 1)
        self.toolbar_2d_main = NavigationToolbar(self.data_plotter_2d_main.canvas, self)
        self.toolbar_2d_x_cut = NavigationToolbar(self.data_plotter_2d_x_cut.canvas, self)
        self.toolbar_2d_y_cut = NavigationToolbar(self.data_plotter_2d_y_cut.canvas, self)
        self.plot_2d_grid_layout.addWidget(self.toolbar_2d_main, 1, 0)
        self.plot_2d_grid_layout.addWidget(self.toolbar_2d_x_cut, 1, 1)
        self.plot_2d_grid_layout.addWidget(self.toolbar_2d_y_cut, 1, 2)

        # 2D Plot Controls (Row 2, Col 0 - Labels and Cut Navigation)
        _2d_controls_frame = QFrame()
        _2d_controls_layout = QVBoxLayout(_2d_controls_frame)
        _2d_controls_layout.addWidget(QLabel("<h3>2D Plot Labels & Cut Controls:</h3>", objectName="titleLabel"))

        # Group Title and Axis Labels
        _2d_label_group_layout = QVBoxLayout()
        self.plot_2d_title_input = QLineEdit("2D Image Plot")
        self.plot_2d_xaxis_input = QLineEdit("X-axis")
        self.plot_2d_yaxis_input = QLineEdit("Y-axis")
        self.plot_2d_cbar_input = QLineEdit("Intensity")

        # Connect editingFinished signals for plot labels
        _2d_label_group_layout.addLayout(self._create_label_input_no_button("Plot Title:", self.plot_2d_title_input))
        self.plot_2d_title_input.editingFinished.connect(lambda: self.set_2d_plot_title(self.plot_2d_title_input.text()))

        _2d_label_group_layout.addLayout(self._create_label_input_no_button("X-axis Label:", self.plot_2d_xaxis_input))
        self.plot_2d_xaxis_input.editingFinished.connect(lambda: self.set_2d_xaxis_label(self.plot_2d_xaxis_input.text()))

        _2d_label_group_layout.addLayout(self._create_label_input_no_button("Y-axis Label:", self.plot_2d_yaxis_input))
        self.plot_2d_yaxis_input.editingFinished.connect(lambda: self.set_2d_yaxis_label(self.plot_2d_yaxis_input.text()))

        _2d_label_group_layout.addLayout(self._create_label_input_no_button("Colorbar Label:", self.plot_2d_cbar_input))
        self.plot_2d_cbar_input.editingFinished.connect(lambda: self.set_2d_cbar_label(self.plot_2d_cbar_input.text()))

        _2d_controls_layout.addLayout(_2d_label_group_layout)

        # Direct Index Input and Navigation Buttons for Cuts
        _2d_controls_layout.addWidget(QLabel("<h4>Direct Cut Index Input:</h4>"))

        # Y-Index for X-Cut
        y_cut_index_layout = QHBoxLayout()
        y_cut_index_layout.addWidget(QLabel("Y-Index for X-Cut:"))
        self.y_cut_index_input = QLineEdit(str(self.current_y_cut_idx))
        self.y_cut_index_input.setValidator(QIntValidator(0, 99999)) # Example range, adjust as needed
        # Store lambda handler and connect (textChanged for immediate feedback on numerical input)
        self._y_cut_text_changed_handler = lambda text: self.set_2d_cut_indices(text, 'y')
        self.y_cut_index_input.textChanged.connect(self._y_cut_text_changed_handler)
        y_cut_index_layout.addWidget(self.y_cut_index_input)
        self.prev_y_cut_button = QPushButton("Prev") # Renamed for clarity
        self.prev_y_cut_button.clicked.connect(lambda: self.navigate_2d_cut('x', -1))
        y_cut_index_layout.addWidget(self.prev_y_cut_button)
        self.next_y_cut_button = QPushButton("Next") # Renamed for clarity
        self.next_y_cut_button.clicked.connect(lambda: self.navigate_2d_cut('x', 1))
        y_cut_index_layout.addWidget(self.next_y_cut_button)
        _2d_controls_layout.addLayout(y_cut_index_layout)

        # X-Index for Y-Cut
        x_cut_index_layout = QHBoxLayout()
        x_cut_index_layout.addWidget(QLabel("X-Index for Y-Cut:"))
        self.x_cut_index_input = QLineEdit(str(self.current_x_cut_idx))
        self.x_cut_index_input.setValidator(QIntValidator(0, 99999)) # Example range, adjust as needed
        # Store lambda handler and connect (textChanged for immediate feedback on numerical input)
        self._x_cut_text_changed_handler = lambda text: self.set_2d_cut_indices(text, 'x')
        self.x_cut_index_input.textChanged.connect(self._x_cut_text_changed_handler)
        x_cut_index_layout.addWidget(self.x_cut_index_input)
        self.prev_x_cut_button = QPushButton("Prev") # Renamed for clarity
        self.prev_x_cut_button.clicked.connect(lambda: self.navigate_2d_cut('y', -1))
        x_cut_index_layout.addWidget(self.prev_x_cut_button)
        self.next_x_cut_button = QPushButton("Next") # Renamed for clarity
        self.next_x_cut_button.clicked.connect(lambda: self.navigate_2d_cut('y', 1))
        x_cut_index_layout.addWidget(self.next_x_cut_button)
        _2d_controls_layout.addLayout(x_cut_index_layout)


        self.plot_2d_grid_layout.addWidget(_2d_controls_frame, 2, 0) # Place controls below main 2D plot

        # X-Cut Data Display (Row 2, Col 1)
        self.x_cut_data_display = QTextEdit()
        self.x_cut_data_display.setReadOnly(True)
        self.x_cut_data_display.setPlaceholderText("X-Cut Data will appear here (Value vs X)")
        self.plot_2d_grid_layout.addWidget(self.x_cut_data_display, 2, 1) # Removed setMaximumHeight
        self.x_cut_data_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        # Y-Cut Data Display (Row 2, Col 2)
        self.y_cut_data_display = QTextEdit()
        self.y_cut_data_display.setReadOnly(True)
        self.y_cut_data_display.setPlaceholderText("Y-Cut Data will appear here (Value vs Y)")
        self.plot_2d_grid_layout.addWidget(self.y_cut_data_display, 2, 2) # Removed setMaximumHeight
        self.y_cut_data_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        self.tab_widget.addTab(self.plot_2d_widget, "2D Data Plot")

        # --- 1D Data Plot Tab ---
        self.plot_1d_widget = QWidget()
        # Use QHBoxLayout for 1D plot and its data display side-by-side
        self.plot_1d_main_layout = QHBoxLayout(self.plot_1d_widget)

        # Left side of 1D tab: Plot and Toolbar
        _1d_plot_and_toolbar_layout = QVBoxLayout()
        self.data_plotter_1d = DataPlotter() # Create 1D specific plotter
        _1d_plot_and_toolbar_layout.addWidget(self.data_plotter_1d.canvas)

        # Matplotlib Toolbar for 1D plot
        self.toolbar_1d = NavigationToolbar(self.data_plotter_1d.canvas, self)
        _1d_plot_and_toolbar_layout.addWidget(self.toolbar_1d)

        # 1D Plot Title and Axis Labels Controls
        _1d_label_controls_frame = QFrame()
        _1d_label_controls_layout = QVBoxLayout(_1d_label_controls_frame)
        _1d_label_controls_layout.addWidget(QLabel("<h3>1D Plot Labels:</h3>", objectName="titleLabel"))

        self.plot_1d_title_input = QLineEdit("1D Line Plot")
        self.plot_1d_xaxis_input = QLineEdit("X-axis")
        self.plot_1d_yaxis1_input = QLineEdit("Y1-axis") # Renamed from plot_1d_yaxis_input
        self.plot_1d_yaxis2_input = QLineEdit("Y2-axis") # New input for second Y-axis

        # Connect editingFinished signals for plot labels
        _1d_label_controls_layout.addLayout(self._create_label_input_no_button("Plot Title:", self.plot_1d_title_input))
        self.plot_1d_title_input.editingFinished.connect(lambda: self.set_1d_plot_title(self.plot_1d_title_input.text()))

        _1d_label_controls_layout.addLayout(self._create_label_input_no_button("X-axis Label:", self.plot_1d_xaxis_input))
        self.plot_1d_xaxis_input.editingFinished.connect(lambda: self.set_1d_xaxis_label(self.plot_1d_xaxis_input.text()))

        _1d_label_controls_layout.addLayout(self._create_label_input_no_button("Y1-axis Label:", self.plot_1d_yaxis1_input))
        self.plot_1d_yaxis1_input.editingFinished.connect(lambda: self.set_1d_yaxis1_label(self.plot_1d_yaxis1_input.text()))

        _1d_label_controls_layout.addLayout(self._create_label_input_no_button("Y2-axis Label:", self.plot_1d_yaxis2_input))
        self.plot_1d_yaxis2_input.editingFinished.connect(lambda: self.set_1d_yaxis2_label(self.plot_1d_yaxis2_input.text()))

        _1d_plot_and_toolbar_layout.addWidget(_1d_label_controls_frame)
        _1d_plot_and_toolbar_layout.addStretch() # Push controls to top

        self.plot_1d_main_layout.addLayout(_1d_plot_and_toolbar_layout, 2) # Give plot more space

        # Right side of 1D tab: Data Display
        _1d_data_display_layout = QVBoxLayout()
        _1d_data_display_layout.addWidget(QLabel("<h3>1D Data Values:</h3>", objectName="titleLabel"))
        self.data_1d_display = QTextEdit()
        self.data_1d_display.setReadOnly(True)
        self.data_1d_display.setPlaceholderText("1D Data will appear here (X vs Y)")
        self.data_1d_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Allow it to expand
        _1d_data_display_layout.addWidget(self.data_1d_display)
        _1d_data_display_layout.addStretch()

        self.plot_1d_main_layout.addLayout(_1d_data_display_layout, 1) # Give data display less space than plot

        self.tab_widget.addTab(self.plot_1d_widget, "1D Data Plot")

        right_panel.addWidget(self.tab_widget)

        main_content_h_layout.addWidget(left_panel_splitter)
        main_content_h_layout.addLayout(right_panel, 1) # Right panel expands
        
        overall_v_layout.addLayout(main_content_h_layout) # Add main content to overall vertical layout


    def _create_label_input_no_button(self, label_text, line_edit_widget):
        """Helper to create a label and line edit without an apply button."""
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel(label_text))
        h_layout.addWidget(line_edit_widget)
        return h_layout

    def select_root_folder(self, is_2d: bool):
        """Allows the user to select the root directory for 2D or 1D data."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Data Root Folder")
        if folder_path:
            if is_2d:
                self.root_folder_2d_path_display.setText(folder_path)
                self.folder_model_2d.setRootPath(folder_path)
                self.tree_view_2d.setRootIndex(self.folder_model_2d.index(folder_path))
                self.data_plotter_2d_main.clear_plot()
                self.data_plotter_2d_x_cut.clear_plot()
                self.data_plotter_2d_y_cut.clear_plot()
                self.data_2d_loaded = None # Reset loaded 2D data
                self.x_coords_2d = None
                self.y_coords_2d = None
                self.current_2d_file_path = None
                self.x_cut_data_display.clear()
                self.y_cut_data_display.clear()
                # Reset cut indices
                self.current_x_cut_idx = 0
                self.current_y_cut_idx = 0
                
                # Disconnect temporarily before setting text to avoid triggering update_2d_cuts prematurely
                if self._y_cut_text_changed_handler:
                    try:
                        self.y_cut_index_input.textChanged.disconnect(self._y_cut_text_changed_handler)
                    except TypeError:
                        pass
                if self._x_cut_text_changed_handler:
                    try:
                        self.x_cut_index_input.textChanged.disconnect(self._x_cut_text_changed_handler)
                    except TypeError:
                        pass

                self.x_cut_index_input.setText(str(self.current_x_cut_idx))
                self.y_cut_index_input.setText(str(self.current_y_cut_idx))

                # Reconnect after setting text
                if self._y_cut_text_changed_handler:
                    self.y_cut_index_input.textChanged.connect(self._y_cut_text_changed_handler)
                if self._x_cut_text_changed_handler:
                    self.x_cut_index_input.textChanged.connect(self._x_cut_text_changed_handler)

            else:
                self.root_folder_1d_path_display.setText(folder_path)
                self.folder_model_1d.setRootPath(folder_path)
                self.tree_view_1d.setRootIndex(self.folder_model_1d.index(folder_path))
                self.data_plotter_1d.clear_plot()
                self.data_1d_loaded_x = None
                self.data_1d_loaded_y1 = None # Reset Y1
                self.data_1d_loaded_y2 = None # Reset Y2
                self.current_1d_file_path = None
                self.data_1d_display.clear()


    def on_tree_view_clicked(self, index: QModelIndex, is_2d: bool):
        """Handles clicks on the file system tree views."""
        model = self.folder_model_2d if is_2d else self.folder_model_1d # Corrected: self.folder_1d to self.folder_model_1d
        path = model.filePath(index)

        if model.isDir(index):
            print(f"Directory selected ({'2D' if is_2d else '1D'}): {path}")
        else:
            print(f"File selected ({'2D' if is_2d else '1D'}): {path}")
            if is_2d:
                self._load_and_plot_2d(path)
            else:
                self._load_and_plot_1d(path)

    def _load_and_plot_2d(self, file_path):
        try:
            self.current_2d_file_path = file_path # Store current file path

            # Always clear the plots before loading new 2D data
            self.data_plotter_2d_main.clear_plot()
            self.data_plotter_2d_x_cut.clear_plot()
            self.data_plotter_2d_y_cut.clear_plot()
            self.x_cut_data_display.clear()
            self.y_cut_data_display.clear()

            # Determine delimiter
            if file_path.lower().endswith(".csv"):
                delimiter = ','
            elif file_path.lower().endswith(".txt"):
                delimiter = r'\s+'
            else:
                QMessageBox.warning(self, "Unsupported 2D File Type",
                                    f"File '{os.path.basename(file_path)}' is not a supported 2D type (.csv, .txt).")
                return

            # Attempt to load with first column as Y-axis and header as X-axis
            try:
                # Read with first row as header and first column as index
                data_df = pd.read_csv(file_path, sep=delimiter, header=0, index_col=0)

                # Extract coordinates and convert to float, coercing errors
                self.x_coords_2d = pd.to_numeric(data_df.columns, errors='coerce').values
                self.y_coords_2d = pd.to_numeric(data_df.index, errors='coerce').values
                self.data_2d_loaded = data_df.values

                # Check for NaN values in coordinates after conversion
                if np.isnan(self.x_coords_2d).any() or np.isnan(self.y_coords_2d).any():
                    QMessageBox.warning(self, "Coordinate Conversion Warning",
                                        f"Some non-numeric values were found in X or Y coordinates of '{os.path.basename(file_path)}' "
                                        "and converted to NaN. These might affect plotting accuracy.")

                if self.data_2d_loaded.ndim != 2:
                    raise ValueError("Data matrix is not 2-dimensional after parsing.")

                # Removed: QMessageBox.information(self, "2D Data Loaded", ...)

            except Exception as parse_error:
                # Fallback if the structured loading fails
                QMessageBox.warning(self, "2D Data Format Warning",
                                    f"Could not parse '{os.path.basename(file_path)}' with first column as Y-axis and header as X-axis. "
                                    f"Reason: {parse_error}. \n\n"
                                    f"Attempting fallback: loading as raw matrix with numerical indices.")
                data_df = pd.read_csv(file_path, header=None, sep=delimiter)
                self.data_2d_loaded = data_df.values
                self.y_coords_2d = np.arange(self.data_2d_loaded.shape[0])
                self.x_coords_2d = np.arange(self.data_2d_loaded.shape[1])
                # Removed: QMessageBox.warning(self, "2D Data Loaded (Fallback)", ...)


            if self.data_2d_loaded.ndim != 2:
                QMessageBox.warning(self, "2D Data Format Error",
                                    f"Loaded data from '{os.path.basename(file_path)}' is not 2-dimensional. Expected a matrix.")
                return

            # Reset cut indices
            self.current_x_cut_idx = 0
            self.current_y_cut_idx = 0
            
            # Disconnect temporarily before setting text to avoid triggering update_2d_cuts prematurely
            if self._y_cut_text_changed_handler:
                try:
                    self.y_cut_index_input.textChanged.disconnect(self._y_cut_text_changed_handler)
                except TypeError:
                    pass
            if self._x_cut_text_changed_handler:
                try:
                    self.x_cut_index_input.textChanged.disconnect(self._x_cut_text_changed_handler)
                except TypeError:
                    pass

            self.x_cut_index_input.setText(str(self.current_x_cut_idx))
            self.y_cut_index_input.setText(str(self.current_y_cut_idx))

            # Reconnect after setting text
            if self._y_cut_text_changed_handler:
                self.y_cut_index_input.textChanged.connect(self._y_cut_text_changed_handler)
            if self._x_cut_text_changed_handler:
                self.x_cut_index_input.textChanged.connect(self._x_cut_text_changed_handler)


            # Plot main 2D image with default titles
            self.data_plotter_2d_main.plot_image(self.data_2d_loaded,
                                                 self.x_coords_2d,
                                                 self.y_coords_2d,
                                                 title=os.path.basename(file_path),
                                                 xlabel=self.plot_2d_xaxis_input.text(),
                                                 ylabel=self.plot_2d_yaxis_input.text(),
                                                 cbar_label=self.plot_2d_cbar_input.text())

            # Initial plot of cuts and update data displays
            self.update_2d_cuts()
            self.tab_widget.setCurrentWidget(self.plot_2d_widget) # Switch to 2D tab

        except Exception as e:
            QMessageBox.critical(self, "2D Data Load Error",
                                 f"Failed to load 2D data from '{os.path.basename(file_path)}':\n{e}")
            print(f"Error loading 2D data from {file_path}: {e}")
            self.data_2d_loaded = None # Ensure data is marked as not loaded on error
            self.data_plotter_2d_main.clear_plot()
            self.data_plotter_2d_x_cut.clear_plot()
            self.data_plotter_2d_y_cut.clear_plot()
            self.x_cut_data_display.clear()
            self.y_cut_data_display.clear()


    def _load_and_plot_1d(self, file_path):
        try:
            self.current_1d_file_path = file_path # Store current file path
            self.data_plotter_1d.clear_plot() # Clear existing 1D plot
            self.data_1d_display.clear() # Clear existing 1D data display

            delimiter = ',' if file_path.lower().endswith(".csv") else r'\s+'

            df = None
            header_names = []
            try:
                # Attempt to read with header
                df = pd.read_csv(file_path, sep=delimiter, header=0)
                header_names = df.columns.tolist()
                # Removed: QMessageBox.information(self, "1D Data Load Info", ...)
            except Exception as e:
                # If reading with header fails, try without header
                QMessageBox.warning(self, "1D Data Header Warning",
                                    f"Could not read '{os.path.basename(file_path)}' with header. Attempting without header. Reason: {e}")
                df = pd.read_csv(file_path, sep=delimiter, header=None)
                # Generate default header names if no header was found
                header_names = [f"Column {i+1}" for i in range(df.shape[1])]

            # Determine number of columns and assign data
            if df.shape[1] >= 3:
                self.data_1d_loaded_x = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
                self.data_1d_loaded_y1 = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
                self.data_1d_loaded_y2 = pd.to_numeric(df.iloc[:, 3], errors='coerce').values
                # Use actual header names for the plot labels if they are meaningful, otherwise fall back to input fields
                # For plot labels, stick to input fields for user control, as requested in previous steps.
                # The request is about the *displayed data header*.
            elif df.shape[1] >= 2:
                self.data_1d_loaded_x = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
                self.data_1d_loaded_y1 = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
                self.data_1d_loaded_y2 = None # No second Y-axis
                # Adjust header_names if only 2 columns were read but original header had more.
                # This ensures the displayed data has correct column count.
                if len(header_names) > 2:
                    header_names = header_names[:2]
            else:
                raise ValueError("Not enough columns found in 1D data. Expected at least two for X and Y.")


            # Filter out NaN values before plotting (apply to all relevant data)
            valid_indices = ~np.isnan(self.data_1d_loaded_x) & ~np.isnan(self.data_1d_loaded_y1)
            if self.data_1d_loaded_y2 is not None:
                valid_indices = valid_indices & ~np.isnan(self.data_1d_loaded_y2)

            x_to_plot = self.data_1d_loaded_x[valid_indices]
            y1_to_plot = self.data_1d_loaded_y1[valid_indices]
            y2_to_plot = self.data_1d_loaded_y2[valid_indices] if self.data_1d_loaded_y2 is not None else None


            if len(x_to_plot) > 0:
                self.data_plotter_1d.plot_line(x_to_plot, y1_to_plot,
                                                  title=os.path.basename(file_path),
                                                  xlabel=self.plot_1d_xaxis_input.text(),
                                                  ylabel=self.plot_1d_yaxis1_input.text(), # Use y1 label
                                                  y2_data=y2_to_plot,
                                                  y2_label=self.plot_1d_yaxis2_input.text() if y2_to_plot is not None else None)
                self.tab_widget.setCurrentWidget(self.plot_1d_widget) # Switch to 1D tab

                # Display 1D data in QTextEdit
                data_dict = {}
                # Use actual header names for display
                if len(header_names) >= 1:
                    data_dict[header_names[0]] = x_to_plot
                else:
                    data_dict[self.plot_1d_xaxis_input.text()] = x_to_plot

                if len(header_names) >= 2:
                    data_dict[header_names[1]] = y1_to_plot
                else:
                    data_dict[self.plot_1d_yaxis1_input.text()] = y1_to_plot

                if y2_to_plot is not None:
                    if len(header_names) >= 3:
                        data_dict[header_names[2]] = y2_to_plot
                    else:
                        data_dict[self.plot_1d_yaxis2_input.text()] = y2_to_plot

                data_1d_df = pd.DataFrame(data_dict)
                self.data_1d_display.setText(data_1d_df.to_string(index=False, float_format="%.2f"))

            else:
                QMessageBox.warning(self, "No Plottable Data",
                                    f"No valid numeric data points found in '{os.path.basename(file_path)}' to plot.")
                self.data_plotter_1d.clear_plot()
                self.data_1d_display.clear()
            
            # Ensure Y2 label input is enabled/disabled based on data availability
            self.plot_1d_yaxis2_input.setEnabled(self.data_1d_loaded_y2 is not None)
            self.plot_1d_yaxis2_input.setStyleSheet("color: #e0e0e0;" if self.data_1d_loaded_y2 is not None else "color: #888888;")


        except Exception as e:
            QMessageBox.critical(self, "1D Data Load Error",
                                 f"Failed to load 1D data from '{os.path.basename(file_path)}':\n{e}")
            print(f"Error loading 1D data from {file_path}: {e}")
            self.data_plotter_1d.clear_plot()
            self.data_1d_loaded_x = None
            self.data_1d_loaded_y1 = None
            self.data_1d_loaded_y2 = None
            self.data_1d_display.clear()
            self.plot_1d_yaxis2_input.setEnabled(False) # Disable Y2 input on error
            self.plot_1d_yaxis2_input.setStyleSheet("color: #888888;")


    def update_2d_cuts(self):
        """Updates the 2D plot's cuts based on current indices and displays data."""
        if self.data_2d_loaded is not None and self.x_coords_2d is not None and self.y_coords_2d is not None:
            # Ensure indices are within bounds
            self.current_y_cut_idx = max(0, min(self.current_y_cut_idx, self.data_2d_loaded.shape[0] - 1))
            self.current_x_cut_idx = max(0, min(self.current_x_cut_idx, self.data_2d_loaded.shape[1] - 1))

            # Disconnect temporarily before setting text to avoid triggering update_2d_cuts recursively
            # Ensure handlers are not None before attempting to disconnect
            if self._y_cut_text_changed_handler:
                try:
                    self.y_cut_index_input.textChanged.disconnect(self._y_cut_text_changed_handler)
                except TypeError: # Catch if it's already disconnected or not connected
                    pass
            if self._x_cut_text_changed_handler:
                try:
                    self.x_cut_index_input.textChanged.disconnect(self._x_cut_text_changed_handler)
                except TypeError:
                    pass

            # Update input fields to reflect current indices
            self.y_cut_index_input.setText(str(self.current_y_cut_idx))
            self.x_cut_index_input.setText(str(self.current_x_cut_idx))

            # Reconnect after setting text
            if self._y_cut_text_changed_handler:
                self.y_cut_index_input.textChanged.connect(self._y_cut_text_changed_handler)
            if self._x_cut_text_changed_handler:
                self.x_cut_index_input.textChanged.connect(self._x_cut_text_changed_handler)


            # Plot X-cut (Value vs X-coordinate)
            x_cut_data_y = self.data_2d_loaded[self.current_y_cut_idx, :]
            self.data_plotter_2d_x_cut.plot_line(self.x_coords_2d, x_cut_data_y,
                                                 title=f"X-Cut at Y-coord {self.y_coords_2d[self.current_y_cut_idx]:.2f}",
                                                 xlabel=self.plot_2d_xaxis_input.text(), # Use main plot's X-axis label
                                                 ylabel=self.plot_2d_cbar_input.text()) # Use main plot's Colorbar label for value axis

            # Display X-cut data
            x_cut_df = pd.DataFrame({
                self.plot_2d_xaxis_input.text(): self.x_coords_2d,
                self.plot_2d_cbar_input.text(): x_cut_data_y
            })
            self.x_cut_data_display.setText(
                f"Y-Coordinate: {self.y_coords_2d[self.current_y_cut_idx]:.2f}\n" +
                x_cut_df.to_string(index=False, float_format="%.2f")
            )

            # Plot Y-cut (Value vs Y-coordinate - ROTATED as requested)
            y_cut_data_x = self.data_2d_loaded[:, self.current_x_cut_idx]
            self.data_plotter_2d_y_cut.plot_line(self.y_coords_2d, y_cut_data_x, # Y-coordinates on X-axis, Values on Y-axis
                                                 title=f"Y-Cut at X-coord {self.x_coords_2d[self.current_x_cut_idx]:.2f}",
                                                 xlabel=self.plot_2d_yaxis_input.text(), # Use main plot's Y-axis label
                                                 ylabel=self.plot_2d_cbar_input.text()) # Use main plot's Colorbar label for value axis

            # Display Y-cut data
            y_cut_df = pd.DataFrame({
                self.plot_2d_yaxis_input.text(): self.y_coords_2d,
                self.plot_2d_cbar_input.text(): y_cut_data_x
            })
            self.y_cut_data_display.setText(
                f"X-Coordinate: {self.x_coords_2d[self.current_x_cut_idx]:.2f}\n" +
                y_cut_df.to_string(index=False, float_format="%.2f")
            )
        else:
            self.data_plotter_2d_x_cut.clear_plot()
            self.data_plotter_2d_y_cut.clear_plot()
            self.x_cut_data_display.clear()
            self.y_cut_data_display.clear()
            
            # Disconnect temporarily before setting text to avoid triggering update_2d_cuts recursively
            if self._y_cut_text_changed_handler:
                try:
                    self.y_cut_index_input.textChanged.disconnect(self._y_cut_text_changed_handler)
                except TypeError:
                    pass
            if self._x_cut_text_changed_handler:
                try:
                    self.x_cut_index_input.textChanged.disconnect(self._x_cut_text_changed_handler)
                except TypeError:
                    pass

            self.x_cut_index_input.setText("0")
            self.y_cut_index_input.setText("0")

            # Reconnect after setting text
            if self._y_cut_text_changed_handler:
                self.y_cut_index_input.textChanged.connect(self._y_cut_text_changed_handler)
            if self._x_cut_text_changed_handler:
                self.x_cut_index_input.textChanged.connect(self._x_cut_text_changed_handler)
            print("No 2D data loaded to update cuts.")

    def navigate_2d_cut(self, cut_type: str, direction: int):
        """Navigates to the next/previous cut."""
        if self.data_2d_loaded is None:
            # Removed: QMessageBox.information(self, "No Data", "Please load 2D data first to navigate cuts.")
            return

        # Disconnect temporarily to avoid triggering set_2d_cut_indices from setText
        if self._y_cut_text_changed_handler:
            try:
                self.y_cut_index_input.textChanged.disconnect(self._y_cut_text_changed_handler)
            except TypeError:
                pass
        if self._x_cut_text_changed_handler:
            try:
                self.x_cut_index_input.textChanged.disconnect(self._x_cut_text_changed_handler)
            except TypeError:
                pass

        if cut_type == 'x': # Controls Y-index for X-cut plot
            max_idx = self.data_2d_loaded.shape[0] - 1
            self.current_y_cut_idx = max(0, min(max_idx, self.current_y_cut_idx + direction))
        elif cut_type == 'y': # Controls X-index for Y-cut plot
            max_idx = self.data_2d_loaded.shape[1] - 1
            self.current_x_cut_idx = max(0, min(max_idx, self.current_x_cut_idx + direction))

        # Update text fields and then reconnect
        self.y_cut_index_input.setText(str(self.current_y_cut_idx))
        self.x_cut_index_input.setText(str(self.current_x_cut_idx))

        if self._y_cut_text_changed_handler:
            self.y_cut_index_input.textChanged.connect(self._y_cut_text_changed_handler)
        if self._x_cut_text_changed_handler:
            self.x_cut_index_input.textChanged.connect(self._x_cut_text_changed_handler)

        self.update_2d_cuts()

    def set_2d_cut_indices(self, text, axis_type):
        """Sets the 2D cut indices from the QLineEdit inputs (triggered by textChanged)."""
        if self.data_2d_loaded is None:
            return

        if not text: # Handle empty string input
            return

        try:
            new_index = int(text)
            if axis_type == 'y': # This is for Y-index (controls X-cut)
                max_idx = self.data_2d_loaded.shape[0] - 1
                if 0 <= new_index <= max_idx:
                    if self.current_y_cut_idx != new_index: # Only update if different
                        self.current_y_cut_idx = new_index
                        self.update_2d_cuts()
                else:
                    # Revert to current valid index if out of bounds
                    if self._y_cut_text_changed_handler:
                        self.y_cut_index_input.textChanged.disconnect(self._y_cut_text_changed_handler)
                    self.y_cut_index_input.setText(str(self.current_y_cut_idx))
                    if self._y_cut_text_changed_handler:
                        self.y_cut_index_input.textChanged.connect(self._y_cut_text_changed_handler)
                    QMessageBox.warning(self, "Invalid Y-Index",
                                        f"Y-Index must be between 0 and {max_idx}.")
            elif axis_type == 'x': # This is for X-index (controls Y-cut)
                max_idx = self.data_2d_loaded.shape[1] - 1
                if 0 <= new_index <= max_idx:
                    if self.current_x_cut_idx != new_index: # Only update if different
                        self.current_x_cut_idx = new_index
                        self.update_2d_cuts()
                else:
                    # Revert to current valid index if out of bounds
                    if self._x_cut_text_changed_handler:
                        self.x_cut_index_input.textChanged.disconnect(self._x_cut_text_changed_handler)
                    self.x_cut_index_input.setText(str(self.current_x_cut_idx))
                    if self._x_cut_text_changed_handler:
                        self.x_cut_index_input.textChanged.connect(self._x_cut_text_changed_handler)
                    QMessageBox.warning(self, "Invalid X-Index",
                                        f"X-Index must be between 0 and {max_idx}.")
        except ValueError:
            # Revert to current valid index if non-integer input
            if axis_type == 'y':
                if self._y_cut_text_changed_handler:
                    self.y_cut_index_input.textChanged.disconnect(self._y_cut_text_changed_handler)
                self.y_cut_index_input.setText(str(self.current_y_cut_idx))
                if self._y_cut_text_changed_handler:
                    self.y_cut_index_input.textChanged.connect(self._y_cut_text_changed_handler)
            elif axis_type == 'x':
                if self._x_cut_text_changed_handler:
                    self.x_cut_index_input.textChanged.disconnect(self._x_cut_text_changed_handler)
                self.x_cut_index_input.setText(str(self.current_x_cut_idx))
                if self._x_cut_text_changed_handler:
                    self.x_cut_index_input.textChanged.connect(self._x_cut_text_changed_handler)
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid integer.")


    # --- Methods to apply plot labels ---
    # These methods are now directly connected to editingFinished signals.
    def set_2d_plot_title(self, title_text):
        if self.data_2d_loaded is not None:
            self.data_plotter_2d_main.plot_image(self.data_2d_loaded, self.x_coords_2d, self.y_coords_2d,
                                             title=title_text,
                                             xlabel=self.plot_2d_xaxis_input.text(),
                                             ylabel=self.plot_2d_yaxis_input.text(),
                                             cbar_label=self.plot_2d_cbar_input.text())
        # No QMessageBox here, as it would pop up on every keystroke.
        # The plot will simply update with whatever text is entered.

    def set_2d_xaxis_label(self, label_text):
        if self.data_2d_loaded is not None:
            self.data_plotter_2d_main.plot_image(self.data_2d_loaded, self.x_coords_2d, self.y_coords_2d,
                                             title=self.plot_2d_title_input.text(),
                                             xlabel=label_text,
                                             ylabel=self.plot_2d_yaxis_input.text(),
                                             cbar_label=self.plot_2d_cbar_input.text())
            self.update_2d_cuts() # Update cuts as their X-axis label might change

    def set_2d_yaxis_label(self, label_text):
        if self.data_2d_loaded is not None:
            self.data_plotter_2d_main.plot_image(self.data_2d_loaded, self.x_coords_2d, self.y_coords_2d,
                                             title=self.plot_2d_title_input.text(),
                                             xlabel=self.plot_2d_xaxis_input.text(),
                                             ylabel=label_text,
                                             cbar_label=self.plot_2d_cbar_input.text())
            self.update_2d_cuts() # Update cuts as their Y-axis label might change

    def set_2d_cbar_label(self, label_text):
        if self.data_2d_loaded is not None:
            self.data_plotter_2d_main.plot_image(self.data_2d_loaded, self.x_coords_2d, self.y_coords_2d,
                                             title=self.plot_2d_title_input.text(),
                                             xlabel=self.plot_2d_xaxis_input.text(),
                                             ylabel=self.plot_2d_yaxis_input.text(),
                                             cbar_label=label_text)
            self.update_2d_cuts() # Update cuts as their value axis label might change

    def set_1d_plot_title(self, title_text):
        if self.data_1d_loaded_x is not None and self.data_1d_loaded_y1 is not None:
            self.data_plotter_1d.plot_line(self.data_1d_loaded_x, self.data_1d_loaded_y1,
                                              title=title_text,
                                              xlabel=self.plot_1d_xaxis_input.text(),
                                              ylabel=self.plot_1d_yaxis1_input.text(),
                                              y2_data=self.data_1d_loaded_y2,
                                              y2_label=self.plot_1d_yaxis2_input.text() if self.data_1d_loaded_y2 is not None else None)

    def set_1d_xaxis_label(self, label_text):
        if self.data_1d_loaded_x is not None and self.data_1d_loaded_y1 is not None:
            self.data_plotter_1d.plot_line(self.data_1d_loaded_x, self.data_1d_loaded_y1,
                                              title=self.plot_1d_title_input.text(),
                                              xlabel=label_text,
                                              ylabel=self.plot_1d_yaxis1_input.text(),
                                              y2_data=self.data_1d_loaded_y2,
                                              y2_label=self.plot_1d_yaxis2_input.text() if self.data_1d_loaded_y2 is not None else None)

    def set_1d_yaxis1_label(self, label_text):
        if self.data_1d_loaded_x is not None and self.data_1d_loaded_y1 is not None:
            self.data_plotter_1d.plot_line(self.data_1d_loaded_x, self.data_1d_loaded_y1,
                                              title=self.plot_1d_title_input.text(),
                                              xlabel=self.plot_1d_xaxis_input.text(),
                                              ylabel=label_text,
                                              y2_data=self.data_1d_loaded_y2,
                                              y2_label=self.plot_1d_yaxis2_input.text() if self.data_1d_loaded_y2 is not None else None)

    def set_1d_yaxis2_label(self, label_text):
        if self.data_1d_loaded_x is not None and self.data_1d_loaded_y1 is not None and self.data_1d_loaded_y2 is not None:
            self.data_plotter_1d.plot_line(self.data_1d_loaded_x, self.data_1d_loaded_y1,
                                              title=self.plot_1d_title_input.text(),
                                              xlabel=self.plot_1d_xaxis_input.text(),
                                              ylabel=self.plot_1d_yaxis1_input.text(),
                                              y2_data=self.data_1d_loaded_y2,
                                              y2_label=label_text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
