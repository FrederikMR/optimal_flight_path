#%% Modules

import os

import jax.numpy as jnp

import sys
import re
import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QMessageBox, QCheckBox, QGridLayout
)
from PyQt5.QtGui import QPixmap, QPalette, QBrush
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QDialog

from flight_model.download_jet_streams import (
    DownloadJetStreams
    )

from flight_model.interpolation import (
    BilinearInterpolation
    )

from flight_model.manifold import (
    WGS84Earth
    )

from flight_model.plotting import (
    plot_earth_chart,
    plot_3d_earth,
    plot_earth_region_chart,
    plot_jet_streams,
    )

#%% Helper functions

def decimal_to_dms(lat, lon):
    def to_dms(deg, is_lat=True):
        degrees = int(abs(deg))
        minutes_float = (abs(deg) - degrees) * 60
        minutes = int(minutes_float)
        seconds = round((minutes_float - minutes) * 60)
        
        if is_lat:
            hemisphere = 'N' if deg >= 0 else 'S'
        else:
            hemisphere = 'E' if deg >= 0 else 'W'

        return f"{degrees}°{minutes}′{seconds}″{hemisphere}"

    return to_dms(lat, is_lat=True) + " " + to_dms(lon, is_lat=False)


def dms_to_decimal(dms_str):
    pattern = r'(\d+)°(\d+)′(\d+)″([NSEW])'
    matches = re.findall(pattern, dms_str)

    if len(matches) != 2:
        raise ValueError("Input must contain both latitude and longitude.")

    def convert(part):
        deg, min_, sec, direction = part
        decimal = int(deg) + int(min_) / 60 + int(sec) / 3600
        if direction in ['S', 'W']:
            decimal *= -1
        return decimal

    lat = convert(matches[0])
    lon = convert(matches[1])
    return lat, lon


#%% GUI

class FlightPathFinder(QWidget):
    def __init__(self):
        super().__init__()

        self.advanced_settings = {
            "flight_speed": 800,
            "flight_height": 11,
            "num_waypoints": 5,
            "jet_stream_pressure": 250,
            "half_axes": "6378.137, 6378.137, 6356.752",
            "grid_points": 100,
            "max_iter": 1000,
            "tolerance": 1e-2,
            "data_folder": "../jet_stream_data/",
            "figure_folder": "../output_figures/"
        }

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Flight Path Finder")
        self.resize(1400, 850)
        self.set_background_image("flight_ui_background_image.png")
    
        # Main horizontal layout
        main_layout = QHBoxLayout()
    
        # ----- LEFT FORM (Main Input Form) -----
        self.form_container = QWidget()
        self.form_container.setMinimumWidth(600)
        self.form_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.form_container.setStyleSheet("""
            background-color: rgba(255, 255, 255, 200);
            padding: 20px;
            border-radius: 8px;
        """)
    
        form_layout = QVBoxLayout()
        form_layout.setSpacing(15)
    
        title = QLabel("<b>Enter Flight Details</b>")
        title.setStyleSheet("font-size: 16px;")
        form_layout.addWidget(title)
    
        grid = QGridLayout()
        grid.setSpacing(10)
    
        self.start_coord_input = QLineEdit()
        self.start_coord_input.setPlaceholderText('e.g. 40°42′46″N 74°00′22″W')
    
        self.end_coord_input = QLineEdit()
        self.end_coord_input.setPlaceholderText('e.g. 34°03′08″N 118°14′37″W')
    
        self.start_city_input = QLineEdit()
        self.start_city_input.setPlaceholderText('e.g. New York')
    
        self.end_city_input = QLineEdit()
        self.end_city_input.setPlaceholderText('e.g. Los Angeles')
    
        def labeled_input(label_text, input_widget):
            container = QWidget()
            vbox = QVBoxLayout()
            vbox.setContentsMargins(0, 0, 0, 0)
            vbox.addWidget(QLabel(label_text))
            vbox.addWidget(input_widget)
            container.setLayout(vbox)
            return container
    
        grid.addWidget(labeled_input("Start coordinate:", self.start_coord_input), 0, 0)
        grid.addWidget(labeled_input("End coordinate:", self.end_coord_input), 0, 1)
        grid.addWidget(labeled_input("Start city:", self.start_city_input), 1, 0)
        grid.addWidget(labeled_input("End city:", self.end_city_input), 1, 1)
    
        form_layout.addLayout(grid)
    
        self.datetime_input = QLineEdit()
        self.datetime_input.setPlaceholderText("HH:mm DD/MM/YYYY (e.g. 12:00 12/08/2025)")
        form_layout.addWidget(self.datetime_input)
    
        self.jet_stream_checkbox = QCheckBox("Use jet stream")
        form_layout.addWidget(self.jet_stream_checkbox)
    
        self.advanced_button = QPushButton("Show Advanced Settings")
        self.advanced_button.clicked.connect(self.toggle_advanced_settings)
        form_layout.addWidget(self.advanced_button)
    
        self.submit_button = QPushButton("Find Path")
        self.submit_button.clicked.connect(self.process_inputs)
        form_layout.addWidget(self.submit_button)
        
        # Create buttons
        self.show_3d_button = QPushButton("Show 3D Earth")
        self.show_chart_button = QPushButton("Show Earth Chart")
        self.show_region_button = QPushButton("Show Regional Chart")
        self.flight_details_button = QPushButton("Flight Details")
        
        # Disable all initially
        self.show_3d_button.setEnabled(False)
        self.show_chart_button.setEnabled(False)
        self.show_region_button.setEnabled(False)
        self.flight_details_button.setEnabled(False)
        
        # Connect to their functions
        self.show_3d_button.clicked.connect(lambda: self.show_figure("earth_3d.png"))
        self.show_chart_button.clicked.connect(lambda: self.show_figure("earth_chart.png"))
        self.show_region_button.clicked.connect(lambda: self.show_figure("earth_region_chart.png"))
        self.flight_details_button.clicked.connect(self.show_flight_details)
        
        buttons_layout = QGridLayout()
        buttons_layout.addWidget(self.show_3d_button, 0, 0)
        buttons_layout.addWidget(self.show_chart_button, 0, 1)
        buttons_layout.addWidget(self.show_region_button, 1, 0)
        buttons_layout.addWidget(self.flight_details_button, 1, 1)
        
        form_layout.addLayout(buttons_layout)
    
        self.form_container.setLayout(form_layout)
    
        # ----- ADVANCED SETTINGS PANEL (Right) -----
        self.advanced_container = QWidget()
        self.advanced_container.setFixedWidth(500)
        self.advanced_container.setStyleSheet("""
            background-color: rgba(255, 255, 255, 220);
            border-radius: 8px;
            padding: 20px;
        """)
        self.advanced_container.setVisible(False)
    
        adv_layout = QGridLayout()
    
        self.speed_input = QLineEdit(str(self.advanced_settings["flight_speed"]))
        self.height_input = QLineEdit(str(self.advanced_settings["flight_height"]))
        self.num_waypoints_input = QLineEdit(str(self.advanced_settings["num_waypoints"]))
        self.half_axes_input = QLineEdit(self.advanced_settings["half_axes"])
        self.grid_input = QLineEdit(str(self.advanced_settings["grid_points"]))
        self.max_iter_input = QLineEdit(str(self.advanced_settings["max_iter"]))
        self.tol_input = QLineEdit(str(self.advanced_settings["tolerance"]))
        self.data_folder_input = QLineEdit(self.advanced_settings["data_folder"])
        self.figure_folder_input = QLineEdit(self.advanced_settings["figure_folder"])
    
        adv_layout.addWidget(QLabel("Flight speed (km/h):"), 0, 0)
        adv_layout.addWidget(self.speed_input, 0, 1)
        adv_layout.addWidget(QLabel("Flight height (km):"), 1, 0)
        adv_layout.addWidget(self.height_input, 1, 1)
        adv_layout.addWidget(QLabel("Flight height (km):"), 1, 0)
        adv_layout.addWidget(self.height_input, 1, 1)
        
        adv_layout.addWidget(QLabel("Number of waypoints:"), 2, 0)
        adv_layout.addWidget(self.num_waypoints_input, 2, 1)
        
        self.jet_stream_pressure_input = QLineEdit(str(self.advanced_settings["jet_stream_pressure"]))
        adv_layout.addWidget(QLabel("Jet stream pressure level (hPa):"), 3, 0)
        adv_layout.addWidget(self.jet_stream_pressure_input, 3, 1)
        
        # Shift all other widgets down one row
        adv_layout.addWidget(QLabel("Half axes (km):"), 4, 0)
        adv_layout.addWidget(self.half_axes_input, 4, 1)
        adv_layout.addWidget(QLabel("Grid points:"), 5, 0)
        adv_layout.addWidget(self.grid_input, 5, 1)
        adv_layout.addWidget(QLabel("Max iterations:"), 6, 0)
        adv_layout.addWidget(self.max_iter_input, 6, 1)
        adv_layout.addWidget(QLabel("Tolerance:"), 7, 0)
        adv_layout.addWidget(self.tol_input, 7, 1)
        adv_layout.addWidget(QLabel("Data folder:"), 8, 0)
        adv_layout.addWidget(self.data_folder_input, 8, 1)
        adv_layout.addWidget(QLabel("Figure folder:"), 9, 0)
        adv_layout.addWidget(self.figure_folder_input, 9, 1)
    
        self.advanced_container.setLayout(adv_layout)
    
        # ----- Final Layout Adjustment -----
        # Create a left-aligned vertical layout for the form
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.form_container)
        left_layout.addStretch()
    
        main_layout.addLayout(left_layout)
        main_layout.addStretch()
        main_layout.addWidget(self.advanced_container)
    
        self.setLayout(main_layout)
        
    def show_figure(self, filename):
        fig_path = os.path.join(self.figure_folder_input.text().strip(), filename)
        if not os.path.isfile(fig_path):
            QMessageBox.warning(self, "File Not Found", f"Could not find: {fig_path}")
            return
    
        dialog = QDialog(self)
        dialog.setWindowTitle(filename)
        layout = QVBoxLayout()
    
        label = QLabel()
        pixmap = QPixmap(fig_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Image Error", f"Could not load image: {fig_path}")
            return
    
        label.setPixmap(pixmap.scaledToWidth(800))  # Resize to fit nicely
        layout.addWidget(label)
        dialog.setLayout(layout)
        dialog.exec_()

    def set_background_image(self, image_path):
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print(f"Warning: Image '{image_path}' not found.")
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(pixmap.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)))
        self.setAutoFillBackground(True)
        self.setPalette(palette)

    def toggle_advanced_settings(self):
        visible = self.advanced_container.isVisible()
        self.advanced_container.setVisible(not visible)
        self.advanced_button.setText("Hide Advanced Settings" if not visible else "Show Advanced Settings")
        
    def show_flight_details(self):
        if hasattr(self, 'flight_details_text'):
            QMessageBox.information(self, "Flight Info", self.flight_details_text)
        else:
            QMessageBox.information(self, "Flight Info", "No flight information available yet.")

    def process_inputs(self):
        start_coord = self.start_coord_input.text().strip()
        end_coord = self.end_coord_input.text().strip()
        start_city = self.start_city_input.text().strip()
        end_city = self.end_city_input.text().strip()
        use_jet_stream = self.jet_stream_checkbox.isChecked()
        num_waypoints = int(self.num_waypoints_input.text())
        jet_stream_pressure = int(self.jet_stream_pressure_input.text())

        try:
            start_lat, start_lon = dms_to_decimal(start_coord)
            end_lat, end_lon = dms_to_decimal(end_coord)
        except ValueError as e:
            QMessageBox.critical(self, "Invalid Input", f"Coordinate error: {e}")
            return

        try:
            flight_speed = float(self.speed_input.text())
            flight_height = float(self.height_input.text())
            grid_points = int(self.grid_input.text())
            max_iter = int(self.max_iter_input.text())
            tolerance = float(self.tol_input.text())
            data_folder = self.data_folder_input.text().strip()
            figure_folder = self.figure_folder_input.text().strip()
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers for advanced settings.")
            return
        
        try:
            half_axes_str = self.half_axes_input.text().strip()
            half_axes = tuple(float(x.strip()) for x in half_axes_str.split(','))
            if len(half_axes) != 3:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Half axes must be 3 comma-separated numbers (e.g. 6378.137, 6378.137, 6356.752).")
            return

        dt_str = self.datetime_input.text().strip()
        if not dt_str:
            QMessageBox.warning(self, "Invalid Input", "Please enter date and time.")
            return

        try:
            dt = datetime.datetime.strptime(dt_str, "%H:%M %d/%m/%Y")
        except ValueError:  
            QMessageBox.warning(self, "Invalid Input", "Date/time must be in format HH:mm DD/MM/YYYY")
            return
        
        if use_jet_stream:
            try:
                dt = datetime.datetime.strptime(dt_str, "%H:%M %d/%m/%Y")
                
                time_part = dt.strftime("%H:%M")
                day_part = dt.strftime("%d")
                month_part = dt.strftime("%m")
                year_part = dt.strftime("%Y")
            
                # Now you can use:
                # time_part → '12:00'
                # day       → '08'
                # month     → '06'
                # year      → '2025'
            
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Date/time must be in format HH:MM DD/MM/YYYY")
                return
            try:
                data_loader = DownloadJetStreams(time=time_part,
                                                 day=day_part,
                                                 month=month_part,
                                                 year=year_part,
                                                 pressure_level=250,
                                                 area = (90, -180, -90, 180),
                                                 save_path=data_folder,
                                                 )
                
                lons, lats, u, v = data_loader()
                
                lons, lats, u, v = jnp.array(lons), jnp.array(lats), jnp.array(u), jnp.array(v)
                speed = jnp.sqrt(u**2+v**2)
                
                interpolation_speed = BilinearInterpolation(lons, lats, speed)
                interpolation_u = BilinearInterpolation(lons, lats, u)
                interpolation_v = BilinearInterpolation(lons, lats, v)
                
                lon_fine, lat_fine, speed_fine = interpolation_speed.interpolate_field(grid_size=(800,400))
                _, _, u_fine = interpolation_u.interpolate_field(grid_size=(800,400))
                _, _, v_fine = interpolation_v.interpolate_field(grid_size=(800,400))
    
                force_field = lambda t,x: jnp.array([interpolation_u(x[0], x[1]),
                                                     interpolation_v(x[0], x[1])])
            except:
                force_field = None
        else:
            force_field = None
            
        if not os.path.exists(figure_folder):
            os.makedirs(figure_folder)
            
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        
        man = WGS84Earth(force_field=force_field,
                         height=flight_height,
                         half_axes=half_axes,
                         flight_speed=flight_speed,
                         grid_points=grid_points,
                         max_iter=max_iter,
                         tol=tolerance,
                         )
        
        z1 = jnp.stack([start_lon, start_lat])
        z2 = jnp.stack([end_lon, end_lat])
        
        
        travel_time, flight_path = man.geodesic(z1, z2)
        
        plot_earth_chart(z1, z2, flight_path, figure_path=figure_folder)
        plot_3d_earth(z1, z2, flight_path, figure_path=figure_folder,
                      start_city=start_city, end_city=end_city)
        plot_earth_region_chart(z1, z2, flight_path, figure_path=figure_folder)
        if force_field is not None:
            plot_jet_streams(lon_fine, lat_fine, u_fine, v_fine, speed_fine, flight_path,
                             figure_path=figure_folder)
        else:
            plot_earth_region_chart(z1, z2, flight_path, figure_path=figure_folder)
        
        if num_waypoints < 2:
            QMessageBox.warning(self, "Invalid Input", "Number of waypoints must be at least 2.")
            return
        
        # Compute indices for exactly num_waypoints points, spaced across the full path
        path_len = flight_path.shape[0]
        indices = jnp.linspace(0, path_len - 1, num=num_waypoints, dtype=int)
        
        waypoints = flight_path[indices]

        # Convert each waypoint (lat, lon) from decimal to DMS string
        waypoints_dms = []
        for point in waypoints:
            lat, lon = float(point[0]), float(point[1])
            waypoints_dms.append(decimal_to_dms(lat, lon))

        eta = dt + datetime.timedelta(hours=float(travel_time))

        # Format the ETA string in same format as user input
        eta_str = eta.strftime("%H:%M %d/%m/%Y")
        
        # Convert hours to timedelta
        travel_time_str = datetime.timedelta(hours=float(travel_time))
        
        # Format it as HH:MM:SS
        travel_time_str = str(travel_time_str)  # returns something like "5:45:14.400000"
        
        # Optionally, strip microseconds for clean display
        travel_time_str = str(travel_time_str).split(".")[0]  # "5:45:14"
        
        self.show_3d_button.setEnabled(True)
        self.show_chart_button.setEnabled(True)
        self.show_region_button.setEnabled(True)
        self.flight_details_button.setEnabled(True)
        
        # Save the text for later
        waypoints_text = "\n".join(waypoints_dms)
        self.flight_details_text = (
            f"Start Coordinate: {start_coord} ({start_lat:.6f}, {start_lon:.6f})\n"
            f"End Coordinate: {end_coord} ({end_lat:.6f}, {end_lon:.6f})\n"
            f"Start City: {start_city}\n"
            f"End City: {end_city}\n"
            f"Date and Time: {dt.strftime('%H:%M %d/%m/%Y')}\n"
            f"Use Jet Stream: {'Yes' if force_field is not None else 'No'}\n\n"
            f"Flight Speed: {flight_speed} km/h\n"
            f"Flight Height: {flight_height} km\n"
            f"Number of Waypoints: {num_waypoints}\n"
            f"Jet Stream Pressure Level: {jet_stream_pressure} hPa\n"
            f"Half Axes: {half_axes}\n"
            f"Grid Points: {grid_points}\n"
            f"Max Iter: {max_iter}\n"
            f"Tolerance: {tolerance:.1e}\n"
            f"Data Folder: {data_folder}\n"
            f"Figure Folder: {figure_folder}\n\n"
            f"Travel time: {travel_time_str}\n"
            f"ETA (departure time): {eta_str}\n\n"
            f"Waypoints:\n{waypoints_text}"
        )
        
        QMessageBox.information(self, "Flight Info", self.flight_details_text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FlightPathFinder()
    window.show()
    sys.exit(app.exec_())
