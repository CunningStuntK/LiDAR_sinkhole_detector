# =====================================================================================================================
# Filename:     COLDS.py
# Written by:   Keith Cusson                Date: Aug 2025
# Description:  This script contains the primary application for the Cusson Open-source LiDAR Depression Scanner
# License:      MIT License (c) 2025 Keith Cusson
# =====================================================================================================================
# ---------------------------------------------------------------------------------------------------------------------
# Import packages
# ---------------------------------------------------------------------------------------------------------------------
from COLDS_utils import COLDS_gui as gui, COLDS_sink as Sink
import gc
import geopandas as gpd
from multiprocessing import Manager, Queue, pool, Process, set_start_method
import numpy as np
import os
from osgeo import gdal, ogr, osr
import pandas as pd
from pathlib import Path
import pdal
import pyproj
from qgis.core import *
import re
from scipy import ndimage
from shapely import Polygon, box
import subprocess
from time import time, sleep

# ---------------------------------------------------------------------------------------------------------------------
# Initialize QGIS
# ---------------------------------------------------------------------------------------------------------------------
# The QGIS prefix path is set in the custom variables defined in the batch file that calls this script.
QgsApplication.setPrefixPath(prefixPath=os.getenv('QGIS_PREFIX_PATH'),
                             useDefaultPaths=True)

# Starting the application creates a PyQt5 QApplication object that can be used to run this application's GUI
qgs: QgsApplication = QgsApplication([], False)
qgs.initQgis()                                              # Initialize QGIS

qgs.setStyle('Fusion')                                      # Set the application style

qgs_exe = f'{os.environ.get("OSGEO4W_ROOT", "")}/bin/qgis-ltr-bin.exe'
# =====================================================================================================================
# CLASSES
# =====================================================================================================================
class Colds(gui.MainWindow):
    """
    Subclass of the COLDS_gui.MainWindow that provides additional functionality relating to QGIS integration.
    """

    # Gui window constructor
    def __init__(
            self,
            app: QgsApplication = None):
        # Initalize the methods of the main window
        super().__init__(app)
        self.qgs_proj: QgsProject | None = None
        self.progress_queue: Queue = Queue()

    def update_data(
            self,
            qgs_proj: QgsProject = None,
            data: gui.ProjectData = None
    ):
        """
        Updates the window's project data. To be used as a slot for a threaded emitter.

        :param qgs_proj: QGIS Project instance to be updated. [Default: None]
        :type qgs_proj:  qgis.core.QgsProject

        :param data:     Project Data object to be updated.
        :type data:      gui.ProjectData

        :return:         No return, updates the window variables in place.
        """
        if qgs_proj is not None:
            self.qgs_proj = qgs_proj

        if data is not None:
            self.data = data

    def update_gdf_pc_tiles(
            self,
            gdf: gpd.GeoDataFrame
    ):
        """
        Update the data for the tile point clouds in the point cloud file metadata geodataframe with a given dataframe,
        and save the results to the layer GeoPackage.

        :param gdf: Geodataframe containing the updated tile cloud metadata.
        :type gdf:  geopandas.GeoDataFrame

        :return:    None. Updates gdf_pc_md in place
        """
        # Update the table in place
        self.data.gdf_pc_md.update(gdf)

        # Determine the path for the vector GeoPackage
        out_path: Path = Path(self.qgs_proj.fileName()).with_suffix('.gpkg')

        # Collect the epsg from the project
        proj_wkt = combine_epsg_codes(self.qgs_proj.crs().authid()[5:],
                                      self.qgs_proj.verticalCrs().authid()[5:],)

        # Save to file
        self.data.gdf_pc_md.to_file(str(out_path),
                                    layer='Point_Clouds',
                                    crs=proj_wkt,
                                    engine='fiona')

    # ------------------------------------------------------------------------------------------------------------------
    # Project Menu
    def create_project(
            self,
            name: str,
            proj_path: str = None
    ):
        """
        This method creates a new project file

        :param name:      Name of the project.
        :type name:       str

        :param proj_path: Path where the project file will be stored.
        :type proj_path:  str

        :return:          None
        """
        if proj_path is None:
            return

        # Create a new QGIS project instance.
        qgs_proj: QgsProject = QgsProject.instance()

        # Set the project identication values
        qgs_proj.setTitle(name)
        qgs_proj.setPresetHomePath(proj_path)
        qgs_proj.setCustomVariables(self.data.export_project_state())

        # Add the link to this application to the project links.
        colds_link: QgsAbstractMetadataBase.Link = QgsProjectMetadata.Link(
            name='Cusson Open-source LiDAR Depression Scanner',
            type='GIT',
            url='https://github.com/CunningStuntK/LiDAR_sinkhole_detector'
        )
        colds_link.description = 'Source code for the program that generated this file.'
        qgs_proj.metadata().addLink(colds_link)

        # Create the folder structure for the project
        folders = ['las_files',
                   'rasters/tiles']
        for f in folders:
            folder = Path(proj_path) / f
            folder.mkdir(parents=True,
                         exist_ok=True)

        # Save the project to file
        out_path: Path = Path(proj_path) / f'{valid_filename(name)}.qgz'
        qgs_proj.setFileName(str(out_path))
        qgs_proj.write()

        # Save the project handle to the application, and prepare the Main Window for use.
        self.qgs_proj = qgs_proj
        self.menu_bar.toggle_add_menu(True)
        self.setWindowTitle(f'{self.program_name} - {self.qgs_proj.title()}')
        self.menu_bar.update_recent(str(out_path))

    def open_project(
            self,
            filename: str = None
    ) -> None:
        """
        This method opens an existing project from a file.

        :param filename: Path to project file to be opened.
        :type filename:  str

        :return: None return
        """
        if filename is None:
            return
        self.qgs_proj: QgsProject = QgsProject.instance()
        result = self.qgs_proj.read(filename=filename)
        if result:
            self.qgs_proj.pathResolver()
        else:
            self.dlg_error('File Open',
                           'Invalid project file, please try again.')
            self.qgs_proj = None
            return
        self.setWindowTitle(f'{self.program_name} - {self.qgs_proj.title()}')

        self.menu_bar.update_recent(filename)

        # Check if the project geopackage exists
        gpkg_path: Path = Path(filename).with_suffix('.gpkg')
        if gpkg_path.exists():
            self.open_project_geopackage(str(gpkg_path))

        # Read the current project state
        self.data.update_proj_state(self.qgs_proj.customVariables())

        self.menu_bar.toggle_add_menu(self.data.state == 0)
        self.wgt_buttons.enable_step_btn(self.data.export_project_state())
        if self.data.state > 0:
            self.wgt_table.enable_selection(0)
            if len(self.data.gdf_vec_md) > 0:
                self.wgt_table.enable_selection(1)
        if self.data.state > 1:
            self.wgt_table.enable_selection(2)

    def open_project_geopackage(
            self,
            filename: str
    ):
        """
        This method opens up an existing project geopackage, and sets the project variables as applicable

        :return: None, sets application project data based on file.
        """
        df_lyrs: pd.DataFrame = gpd.list_layers(filename)

        if 'AOI' in df_lyrs.values:
            self.data.gdf_aoi = gpd.read_file(filename,
                                              layer='AOI')

        if 'Water_Features' in df_lyrs.values:
            self.data.gdf_water = gpd.read_file(filename,
                                                layer='Water_Features')

        if 'Point_Clouds' in df_lyrs.values:
            self.data.gdf_pc_md = gpd.read_file(filename,
                                                layer='Point_Clouds')

        if 'Vector_Metadata' in df_lyrs.values:
            self.data.gdf_vec_md = gpd.read_file(filename,
                                                 layer='Vector_Metadata')

        # TODO: Add statement for depression data

    # ------------------------------------------------------------------------------------------------------------------
    # Add Menu
    def add_cloud(
            self,
            cloud_type: str,
            filename: list[str]
    ):
        """
        This method adds any point clouds loaded by selecting point clouds from the Add menu

        :param cloud_type: Cloud type label for loaded files.
        :type cloud_type:  str

        :param filename:   Path(s) to the list of point cloud(s) to be uploaded
        :type filename:    list[str]

        :return:           None
        """
        # Retrieve the metadata from the file(s) as a list of dictionaries
        md_list = []
        for las_file in filename:
            # Prior to reading the file, determine if it has already been added to the geodataframe.
            if las_file in self.data.gdf_pc_md.values:
                self.dlg_warning('File Exists',
                                 f'This file has already been loaded:\n{las_file}')
                continue
            md = read_input_las_metadata(las_file, cloud_type)
            md_list.append(md)

        if len(md_list) < 1:
            return

        # Generate a geodataframe from the remaining files, and concatenate it with the existing frame.
        gdf_las: gpd.GeoDataFrame = gpd.GeoDataFrame(
            data=md_list,
            geometry=gpd.GeoSeries.from_wkt([md['extent'] for md in md_list]))

        gdf_las.drop('extent',
                     axis=1,
                     inplace=True)

        self.data.gdf_pc_md = pd.concat([self.data.gdf_pc_md, gdf_las],
                                        ignore_index=True)
        self.wgt_message.print_message(f'{len(gdf_las)} point cloud(s) added to the project.')
        self.wgt_table.btn_accept.setVisible(True)

        # Ensure the data type has been activated in the combo box
        self.wgt_table.enable_selection(0)

        # If no selection has been made in the combo box, select Input Cloud
        if self.wgt_table.combo_data_type.currentIndex() == -1:
            self.wgt_table.combo_data_type.setCurrentIndex(0)

        # If Input Cloud is the current selection, update the table.
        if self.wgt_table.combo_data_type.currentIndex() == 0:
            self.change_type()

    def add_vector(
            self,
            lyr_type: str,
            filename: list[str]
    ) -> None:
        """
        This function provides all the functionality for adding vector files, including adding them to the vector
        metadata DataFrame, enabling the selector combo box, and updating the statistics table area.

        :param lyr_type: String representing the type of vector layer loaded.
        :type lyr_type:  str

        :param filename: List of paths to files to be added to the project.
        :type filename:  list[str]

        :return:         None
        """
        # Create a list to hold all loaded dataframes.
        vec_df_list: list[pd.DataFrame] = [self.data.gdf_vec_md]

        # Load the vector file(s) selected by the user as geodataframes.
        for vec_file in filename:
            file_name = Path(vec_file).name
            vec_layers: pd.DataFrame = gpd.list_layers(vec_file)
            # Add only the layers within the file that the user selects.
            if len(vec_layers) > 1:
                selected_layers = self.dlg_list(
                    title='Select Layers',
                    message=f'Which layers would you like to load for file:\n{file_name}',
                    vals=vec_layers.name.tolist()
                )
                if selected_layers is None:
                    continue
                # Remove all unselected layers from the dataframe.
                vec_layers = vec_layers[vec_layers['name'].isin(selected_layers)]

            # Add the additional metadata required for later processing.
            vec_layers['filename'] = vec_file
            vec_layers['layer_type'] = lyr_type

            # Determine if any of the selected layers already exist, and notify the user there can be no duplicates.
            if vec_file in self.data.gdf_vec_md.values:
                # If the file has already been loaded, check to see if any of its layers have been loaded using a merge.
                df_all: pd.DataFrame = vec_layers.merge(self.data.gdf_vec_md[['name', 'filename']],
                                                        on=['name', 'filename'],
                                                        how='left',
                                                        indicator=True)
                # If the layer is already in the metadata dataframe, notifu the user and do not add it.
                if 'both' in df_all['_merge'].values:
                    dup_lyrs = df_all.loc[df_all['_merge'] == 'both', 'name'].tolist()
                    warn_msg = f'The following layers in **{file_name}** have already been added:\n'
                    for lyr in dup_lyrs:
                        warn_msg += f'  + {lyr}\n'
                    self.dlg_warning('Duplicate Layer',
                                     warn_msg)

                    vec_layers = vec_layers[df_all['_merge'] != 'both']

            # If no layers have been identified, move to the next file
            if len(vec_layers) == 0:
                continue

            # Retrieve the spatial reference for each layer
            ds_vec: gdal.Dataset = gdal.OpenEx(vec_file)
            vec_layers['srs'] = vec_layers.apply(
                lambda x: ds_vec.GetLayer(x['name']).GetSpatialRef(),
                axis=1
            )

            # Compute the EPSG code(s) for layer(s) that have a spatial reference.
            filt: np.ndarray[bool] = vec_layers.srs.isnull()
            vec_layers.loc[~filt, ['Horizontal EPSG', 'Vertical EPSG']] = vec_layers[~filt].apply(
                lambda x: epsg_codes_from_wkt(x.srs.ExportToWkt()),
                axis=1,
                result_type='expand'
            )

            # Convert instances where no spatial reference was detected to empty strings
            vec_layers.loc[filt, ['Horizontal EPSG', 'Vertical EPSG']] = ''

            # Store the compound wkt description of the spatial reference for future processing.
            vec_layers.loc[:, 'wkt'] = vec_layers.apply(
                lambda x: combine_epsg_codes(x['Horizontal EPSG'], x['Vertical EPSG']),
                axis=1
            )

            vec_layers['geometry'] = None
            vec_layers = gpd.GeoDataFrame(data=vec_layers,
                                          geometry='geometry')
            vec_layers.drop(columns=['srs'],
                            inplace=True)

            vec_df_list.append(vec_layers)

        # If the vector list only contains the original dataframe, do not make any changes
        if len(vec_df_list) == 1:
            return

        # Add the data to the vector metadata DataFrame.
        self.data.gdf_vec_md = pd.concat(vec_df_list,
                                         ignore_index=True)
        self.wgt_message.print_message(f'{len(vec_df_list) - 1} {lyr_type} vector layer(s) added to the project.')

        # Ensure the data type has been activated in the combo box
        self.wgt_table.enable_selection(1)

        # If no selection has been made in the combo box, select Vector Data
        if self.wgt_table.combo_data_type.currentIndex() == -1:
            self.wgt_table.combo_data_type.setCurrentIndex(1)

        # If Vector Data is the current selection, update the table.
        if self.wgt_table.combo_data_type.currentIndex() == 1:
            self.change_type()

    def add_raster(self):
        """
        Function that adds rasters to the QGIS project, and saves the project to file.

        :return: None. Saves QGIS project to file
        """
        # Retrieve the folder containing the rasters to be added to the project.
        rast_dir: Path = Path(self.qgs_proj.homePath()) / 'rasters'
        root: QgsLayerTree = self.qgs_proj.layerTreeRoot()

        # Iterate through each raster layer, and add them to their appropriate group.
        for rast in rast_dir.glob('*.tif'):
            res_group: QgsLayerTreeGroup = root.findGroup(rast.stem)

            # Create the raster layer and add it to the map.
            qlyr: QgsRasterLayer = QgsRasterLayer(str(rast),
                                                  rast.stem)
            qlyr_map: QgsMapLayer = self.qgs_proj.addMapLayer(qlyr,
                                                              False)
            res_group.addLayer(qlyr_map)

        # Save the project to file.
        self.qgs_proj.write()

    # ------------------------------------------------------------------------------------------------------------------
    # Table View
    def change_type(self):
        """
        This function updates the display table if a change in the display selector is detected

        :return: None return, updates the table.
        """
        # Determine the state of the selector
        match self.wgt_table.combo_data_type.currentIndex():
            # Input Cloud
            case 0:
                # Filter the point cloud dataframe for input clouds
                filt: pd.Series = self.data.gdf_pc_md.cloud_type == 'Input Cloud'
                columns = ['filename',
                           'name',
                           'Size (bytes)',
                           '# Points',
                           'min x',
                           'max x',
                           'min y',
                           'max y',
                           'Classified',
                           'Horizontal EPSG',
                           'Vertical EPSG',
                           'Read Time (s)']

                # Create the dataframe to be set as the model.
                df_model = self.data.gdf_pc_md.loc[filt, columns]

            # Vector File
            case 1:
                columns = ['filename',
                           'name',
                           'layer_type',
                           'geometry_type',
                           'Horizontal EPSG',
                           'Vertical EPSG']
                df_model = self.data.gdf_vec_md.loc[:, columns]

            # Tile Cloud
            case 2:
                # Filter the point cloud data for tile clouds
                filt: pd.Series = self.data.gdf_pc_md.cloud_type == 'Tile Cloud'
                columns = ['filename',
                           'name',
                           'Size (bytes)',
                           '# Points',
                           'min x',
                           'max x',
                           'min y',
                           'max y',
                           'Classified',
                           'Horizontal EPSG',
                           'Vertical EPSG',
                           'Read Time (s)']

                # Create the dataframe to be set as the model.
                df_model = self.data.gdf_pc_md.loc[filt, columns]

            case _:
                return
        self.wgt_table.table.clearSelection()
        self.wgt_table.model.update_data(df_model)
        self.wgt_table.table.resizeColumnsToContents()

    def remove_file(
            self,
            index: int
    ):
        """
        Method that removes an input file selected from the stats table.

        :param index: Row index for the file to be removed.
        :type index:  int

        :return:      None. Removes the file from the appropriate project dataframe and refreshes the stats table view.
        """
        # Determine what type of file you intend to remove
        if self.wgt_table.combo_data_type.currentIndex() == 0:
            self.wgt_message.print_message(
                f'Removed point cloud: {self.data.gdf_pc_md.loc[index, "filename"]}'
            )
            self.data.gdf_pc_md.drop([index],
                                     inplace=True)
            self.data.gdf_pc_md.reset_index(inplace=True,
                                            drop=True)
            if len(self.data.gdf_pc_md) == 0:
                self.wgt_table.btn_accept.setVisible(False)
        elif self.wgt_table.combo_data_type.currentIndex() == 1:
            self.wgt_message.print_message(
                f'Removed {self.data.gdf_vec_md.loc[index, "layer_type"]} vector layer: '
                f'{self.data.gdf_vec_md.loc[index, "filename"]}'
            )
            self.data.gdf_vec_md.drop([index],
                                      inplace=True)
            self.data.gdf_pc_md.reset_index(inplace=True,
                                            drop=True)
        else:
            return

        self.change_type()

    # ------------------------------------------------------------------------------------------------------------------
    # Input File finalization
    def finalize_start(self):
        """
        Method that finalizes input files, and adds them to project.

        :return: None. Updates ProjectData and qgs_proj, saves data to file. Final processing is done in a thread.
        """
        # Get confirmation that all required files have been added.
        msg = f'Are these all the files required for the project?'
        if len(self.data.gdf_vec_md) > 0:
            df_vec: pd.DataFrame = self.data.gdf_vec_md.drop(columns=['geometry'])
        else:
            df_vec: pd.DataFrame = pd.DataFrame()
        response = self.dlg_input_tree('Finalize inputs',
                                       'Are these all the files required for the project?',
                                       df_vec,
                                       self.data.gdf_pc_md.drop(columns=['geometry']))

        # If the user responds no, uncheck the button, and exit the function.
        if not response:
            self.wgt_table.btn_accept.setChecked(False)
            return

        # Determine the Horizontal and Vertical CRS for the project
        df_epsg: pd.DataFrame = pd.concat([self.data.gdf_pc_md, self.data.gdf_vec_md],
                                          ignore_index=True)
        if len(df_epsg.drop_duplicates(subset=['Horizontal EPSG', 'Vertical EPSG'])) != 1:
            df_epsg.loc[~(df_epsg['Horizontal EPSG'] == ''), 'H_Desc'] = (
                df_epsg[~(df_epsg['Horizontal EPSG'] == '')].apply(
                    lambda x: pyproj.CRS.from_epsg(x['Horizontal EPSG']).name,
                    axis=1
                ))
            df_epsg.loc[~(df_epsg['Vertical EPSG'] == ''), 'V_Desc'] = (
                df_epsg[~(df_epsg['Vertical EPSG'] == '')].apply(
                    lambda x: pyproj.CRS.from_epsg(x['Vertical EPSG']).name,
                    axis=1
                ))
            df_epsg.loc[df_epsg.cloud_type == 'Input Cloud', 'layer_type'] = 'Point Cloud'

            epsg_list: list[str] = self.dlg_epsg_table('Select Coordinate System',
                                                       'Given the following coordinate reference systems from the '
                                                       'input files, select a horizontal and vertical coordinate '
                                                       'system for the project.<br>'
                                                       'Ideally, select the relevant coordinate systems from the point '
                                                       'cloud inputs to generate the most accurate results.',
                                                       df_epsg[['filename',
                                                                'H_Desc',
                                                                'V_Desc',
                                                                'layer_type',
                                                                'name',
                                                                'Horizontal EPSG',
                                                                'Vertical EPSG']])
        else:
            columns = ['Horizontal EPSG', 'Vertical EPSG']
            epsg_list: np.ndarray = df_epsg.drop_duplicates(subset=columns)[columns].to_numpy()[0]
            epsg_list: list[str] = epsg_list.tolist()

        # Prepare the application for displaying progress bars and running the thread
        self.wgt_progress.show_bars(2)
        self.stack.setCurrentIndex(1)
        self.wgt_progress.update_desc('Finalizing Inputs')
        self.wgt_table.btn_accept.setVisible(False)
        self.menu_bar.toggle_add_menu(False)

        # Create the EPSG for the project from the provided inputs
        if not epsg_list[0] == '':
            crs_h: QgsCoordinateReferenceSystem = QgsCoordinateReferenceSystem(f'EPSG:{epsg_list[0]}')
            self.qgs_proj.setCrs(crs_h)
            self.wgt_message.print_message(f'Horizontal Spatial Reference System set to EPSG:{epsg_list[0]}')
        if not epsg_list[1] == '':
            crs_v: QgsCoordinateReferenceSystem = QgsCoordinateReferenceSystem(f'EPSG:{epsg_list[1]}')
            self.qgs_proj.setVerticalCrs(crs_v)
            self.wgt_message.print_message(f'Vertical Spatial Reference System set to EPSG:{epsg_list[1]}')
        proj_wkt = combine_epsg_codes(epsg_list[0], epsg_list[1])

        # Update the progress bar.
        self.wgt_progress.update_bar(10)

        # Update the project state
        self.data.state = 1
        process: Process = Process(target=finalize_inputs,
                                   args=(Path(self.qgs_proj.fileName()),
                                         self.data,
                                         proj_wkt,
                                         self.progress_queue))

        self.worker_signals.result.connect(self.update_data)
        self.worker_signals.finished.connect(self.finalize_end)
        self.worker_signals.finished.connect(self.timer.stop)
        self.worker_signals.finished.connect(process.join)
        self.timer.start(200)
        process.start()

    def finalize_end(self):
        """
        Method that adds the input layers as layers to the QGIS project once the project's geopackage has been created.

        :return: None. Updates qgs_proj in place
        """
        # Add the layers to the QGIS project
        self.wgt_progress.update_desc('Saving layers to QGIS Project', 1)
        self.wgt_progress.reset_bar(3, 1)
        out_file: Path = Path(self.qgs_proj.fileName()).with_suffix('.gpkg')

        # Add water features to the QGIS project if they exist
        if len(self.data.gdf_water) > 0:
            qlyr = QgsVectorLayer(f'{str(out_file)}|layername=Water_Features',
                                  'Water Features',
                                  'ogr')
            qlyr_wf: QgsVectorLayer = self.qgs_proj.addMapLayer(qlyr)

            # Set the styling for the Water Feature layer
            qstyle: QgsSymbol = QgsStyle.defaultStyle().symbol('topo water')
            qrend: QgsFeatureRenderer = QgsSingleSymbolRenderer(qstyle)
            qlyr_wf.setRenderer(qrend)
            qlyr_wf.triggerRepaint()

            # Print a completion message
            self.wgt_message.print_message('Water features layer added to project')

        # Update the progress bars
        self.wgt_progress.update_bar(1, 1)
        self.wgt_progress.update_bar(90, 0)

        # Add the point cloud extents to the QGIS project file
        qlyr = QgsVectorLayer(f'{str(out_file)}|layername=Point_Clouds',
                              'Point Cloud Extents',
                              'ogr')
        qlyr_pc: QgsVectorLayer = self.qgs_proj.addMapLayer(qlyr)

        # Set the styling for the point cloud extents
        qstyle: QgsSymbol = QgsStyle.defaultStyle().symbol('outline green')
        qstyle.symbolLayer(0).setWidth(0.5)
        qrend: QgsFeatureRenderer = QgsSingleSymbolRenderer(qstyle)
        qlyr_pc.setRenderer(qrend)
        qlyr_pc.triggerRepaint()
        qlyr_pc.setOpacity(0.3)

        # Report function progress
        self.wgt_progress.update_bar(2, 1)
        self.wgt_progress.update_bar(95, 0)
        self.wgt_message.print_message('Point cloud extents layer added to project')

        # Add the area of interest layer to the QGIS project file
        qlyr: QgsVectorLayer = QgsVectorLayer(f'{str(out_file)}|layername=AOI',
                                              'AOI',
                                              'ogr')
        qlyr_aoi: QgsVectorLayer = self.qgs_proj.addMapLayer(qlyr)

        # Set the styling for the AOI layer
        qstyle: QgsSymbol = QgsStyle.defaultStyle().symbol('outline red')
        qstyle.symbolLayer(0).setWidth(0.5)
        qrend: QgsFeatureRenderer = QgsSingleSymbolRenderer(qstyle)
        qlyr_aoi.setRenderer(qrend)
        qlyr_aoi.triggerRepaint()

        # Report function process
        self.wgt_progress.update_bar(3, 1)
        self.wgt_progress.update_bar(99, 0)
        self.wgt_message.print_message('Area of interest layer added to project')

        # Change the state of the program, set the default view, and save the project file to disk.
        self.qgs_proj.setCustomVariables(self.data.export_project_state())
        view_settings: QgsProjectViewSettings = self.qgs_proj.viewSettings()
        extent: QgsReferencedRectangle = QgsReferencedRectangle(rectangle=qlyr_aoi.extent().buffered(50),
                                                                crs=self.qgs_proj.crs())
        view_settings.setDefaultViewExtent(extent)
        self.qgs_proj.write()

        # Reset window use
        self.stack.setCurrentIndex(0)
        self.set_buttons()

        # Disconnect signals connected for the finalize_start script
        self.worker_signals.result.disconnect()
        self.worker_signals.finished.disconnect()

    # ------------------------------------------------------------------------------------------------------------------
    # Action Buttons
    def click_tile(self):
        """
        Method to merge input point clouds and split them into equal sized tiles.

        :return: None. Performs the described actions, and saves the new point clouds to the las_tiles folder.
        """
        # Approximate the number of points in the area of interest by computing the percentage overlap between each
        # point cloud with the area of interest and using that to approximate the number of points from that point
        # cloud that are in the area of interest. Increase the result by 25% to account for point clouds that do not
        # cover their full bounding box extents
        aoi: Polygon = self.data.gdf_aoi.geometry.union_all()
        coverage: pd.Series = self.data.gdf_pc_md.intersection(aoi).area / self.data.gdf_pc_md.area

        approx_points = (coverage * self.data.gdf_pc_md['# Points']).sum() * 1.25

        tiles = 2048 * approx_points / gdal.GetUsablePhysicalRAM()

        # A tiling scheme needs to be derived to subdivide the area of interest. The ideal tiling scheme will create
        # equal size squares to cover the entirety of the bounding box. To compensate for boundary issues, there should
        # be a 5 metre overlap on each side of the box

        # Divide the area of the area of interest by the number of tiles to determine the side length for each tile.
        max_tile_area = aoi.area / tiles
        max_tile_length = np.floor(np.sqrt(max_tile_area)) - 10  # Subtracting 10 accounts for the buffer on both sides

        # Ensure the tile length is no larger than the size of the largest side of the bounding box of the area of
        # interest
        minx, miny, maxx, maxy = aoi.bounds
        max_tile_length = min(max_tile_length, max(maxx - minx, maxy - miny))

        # Create a grid of tiles with the max tile length, and overlay it onto the area of interest.
        tile_grid = [int(np.ceil((maxx - minx) / max_tile_length)),
                     int(np.ceil((maxy - miny) / max_tile_length))]

        # Warn the user that the process can be incredibly time-consuming, and confirm they wish to start.
        response = self.dlg_confirm('Merging/Tiling Point clouds',
                                    f'This process will convert the {len(self.data.gdf_pc_md)} loaded input '
                                    f'cloud(s) into up to {tile_grid[0] * tile_grid[1]:,} tile(s). This process may '
                                    f'take several hours depending on computer performance. If it is interrupted, the '
                                    f'process will need to be restarted. Would you like to continue?')

        if response == 65536:
            self.wgt_buttons.button_tile.setChecked(False)
            return

        # Run the merge/split thread.
        self.wgt_message.print_message('Point cloud merge & tile process started.')
        # Prepare the application for displaying progress bars and running the thread
        self.wgt_progress.show_bars(2)
        self.stack.setCurrentIndex(1)
        self.wgt_progress.update_desc('Merging and splitting point clouds')
        self.wgt_progress.reset_bar(len(self.data.gdf_pc_md) * 2)
        self.wgt_progress.set_perc(0)
        self.wgt_progress.update_desc('Creating grid', 1)
        self.wgt_progress.reset_bar(1, 1)
        self.wgt_buttons.disable_all()

        # Update the project state
        self.data.state = 2

        # Retrieve the project's spatial reference data
        epsg_list = [self.qgs_proj.crs().authid()[5:],
                     self.qgs_proj.verticalCrs().authid()[5:]]

        # Run the thread
        process: Process = Process(target=tile_and_merge,
                                   args=(self.data.gdf_pc_md,
                                         aoi,
                                         max_tile_length,
                                         Path(self.qgs_proj.fileName()),
                                         epsg_list,
                                         self.no_cpus,
                                         self.progress_queue))

        self.worker_signals.error.connect(print)
        self.worker_signals.error.connect(breakpoint)
        self.worker_signals.result.connect(self.data.update_gdf_pc_md)
        self.worker_signals.finished.connect(self.process_cleanup)
        self.worker_signals.finished.connect(process.join)
        self.timer.start(50)
        process.start()

    def process_cleanup(self):
        """
        This method reverts the display to where it should be after a process has finished.

        :return: None
        """
        # Set the gui state and stop the buttons
        self.timer.stop()
        self.stack.setCurrentIndex(0)
        self.change_type()
        self.wgt_table.enable_selection(2)
        self.set_buttons()

        # Update the QGIS project, and save it to file.
        self.qgs_proj.setCustomVariables(self.data.export_project_state())
        self.qgs_proj.write()

        # Disconnect signals connected for the click_tile script
        if self.worker_signals.receivers(self.worker_signals.result) > 0:
            self.worker_signals.result.disconnect()
        self.worker_signals.finished.disconnect()

    def click_dem(self):
        """
        This method generates the DEMs for a project from the point clouds, classifying them along the way if required.

        :return: None
        """
        # Determine which point clouds need to be classified
        gdf_tiles: gpd.GeoDataFrame = self.data.gdf_pc_md[self.data.gdf_pc_md['cloud_type'] == 'Tile Cloud']
        classified: pd.Series = gdf_tiles['Classified'] == 'True'

        # If any tiles don't contain classified points, they should all be classified using the same method to guarantee
        # a uniform classification
        if classified.all():
            # If all the point cloud tiles have been classified, ask the user if they want to reclassify them
            res: int = self.dlg_confirm(
                title='Reclassify points',
                message='All point clouds have already had ground points identified, would you like to reclassify them '
                        'using Cloth Surface Filtering?'
            )
            class_flag: bool = res == 16384

        else:
            class_flag: bool = True

        # Ask the user what resolution DEM rasters are required.
        dem_res: list[str] = self.dlg_resolution()

        # If the user has cancelled on raster selection, exit the function
        if len(dem_res) == 0:
            self.wgt_buttons.button_dem.setChecked(False)
            return

        # Create the folders that will contain the tiles for a particular resolution
        raster_tile_folder: Path = Path(self.qgs_proj.homePath()) / 'rasters' / 'tiles'

        for resolution in dem_res:
            (raster_tile_folder / resolution).mkdir(parents=True,
                                                    exist_ok=True)

        # Run the classification and DEM generation in a separate process to prevent GUI hang-ups.
        self.wgt_message.print_message(f'Commencing DEM generation')

        # Display 2 progress bars
        self.wgt_progress.show_bars(2)
        self.stack.setCurrentIndex(1)

        # Set the top display bar with one step for each tile, and one additional step for each resolution to be merged
        self.wgt_progress.update_desc('Generating Tile DEMs')
        self.wgt_progress.reset_bar(len(gdf_tiles) + len(dem_res))
        self.wgt_progress.set_perc(0)

        self.wgt_buttons.disable_all()

        # Update the project state
        self.data.state = 3

        # Run the thread
        process: Process = Process(target=generate_dem,
                                   args=(gdf_tiles,
                                         self.qgs_proj.homePath(),
                                         dem_res,
                                         class_flag,
                                         self.no_cpus,
                                         self.progress_queue))

        self.worker_signals.error.connect(print)
        self.worker_signals.error.connect(breakpoint)
        self.worker_signals.result.connect(self.update_gdf_pc_tiles)
        self.worker_signals.finished.connect(self.add_raster)
        self.worker_signals.finished.connect(process.join)
        self.worker_signals.finished.connect(self.process_cleanup)
        self.timer.start(50)
        process.start()

        # Retrieve the QGIS project tree root
        root: QgsLayerTree = self.qgs_proj.layerTreeRoot()

        # Add a group for each selected raster resolution
        for resolution in dem_res:
            root.addGroup(f'DEM_{resolution}')

    def click_sinkholes(self):
        """
        Method to run the identify sinkholes process

        :return: None. Runs the process
        """
        # Run the sinkhole identification function in a separate process to prevent GUI hang-ups.
        self.wgt_message.print_message(f'Commencing Sinkhole identification')

        # Display 2 progress bars
        self.wgt_progress.show_bars(2)
        self.stack.setCurrentIndex(1)

        # Set the top display bar with one step for each resolution
        self.wgt_progress.update_desc('Indentifying Sinkholes')
        self.wgt_progress.reset_bar(100)

        self.wgt_buttons.disable_all()

        # Update the project state
        self.data.state = 4

        # Run the thread
        process: Process = Process(target=Sink.identify_sinkholes,
                                   args=(self.qgs_proj.fileName(),
                                         self.progress_queue))

        self.worker_signals.error.connect(print)
        self.worker_signals.error.connect(breakpoint)
        self.worker_signals.result.connect(self.data.update_gdf_depressions)
        self.worker_signals.finished.connect(self.add_sinkholes)
        self.worker_signals.finished.connect(process.join)
        self.worker_signals.finished.connect(self.process_cleanup)
        self.timer.start(50)
        process.start()

    def add_sinkholes(self):
        """
        Method that adds identified depressions to the QGIS project after identify_sinkholes has completed.

        :return: None
        """
        # Retrieve the layers available in the project's geopackage, and select the ones that contain depressions
        path_vec: Path = Path(self.qgs_proj.fileName()).with_suffix('.gpkg')
        df_lyrs: pd.DataFrame = gpd.list_layers(path_vec)
        df_lyrs = df_lyrs[df_lyrs['name'].str.contains('Depressions', regex=False)]

        # Retrieve the QGIS project tree root.
        root: QgsLayerTree = self.qgs_proj.layerTreeRoot()

        # All polygons will use the same basic default symbol, that will be colourized based on the sinkhole score.
        symbol_properties = {'color': 'red',
                             'outline_color': 'black',
                             'outline_width': '0.25',
                             'outline_style': 'solid'}
        fill_symbol: QgsFillSymbol = QgsFillSymbol.createSimple(symbol_properties)

        # Create the renderer for the depression layers
        qrend: QgsGraduatedSymbolRenderer = QgsGraduatedSymbolRenderer()
        qrend.setSourceSymbol(fill_symbol)

        # Add the ranges for the depressions
        qrend.addClassLowerUpper(0.0, 0.5)
        qrend.addClassLowerUpper(0.5, 0.6)
        qrend.addClassLowerUpper(0.6, 0.8)
        qrend.addClassLowerUpper(0.8, 1.0)
        qrend.setClassAttribute('Score')

        # Update the colour scheme for the depressions
        qrend.updateColorRamp(QgsStyle().defaultStyle().colorRamp('RdYlGn'))

        qrend.updateRangeLabel(0, 'Unlikely (<0.5)')
        qrend.updateRangeLabel(1, 'Low (0.5 - 0.6)')
        qrend.updateRangeLabel(2, 'Medium (0.6 - 0.8)')
        qrend.updateRangeLabel(3, 'High (>0.8)')

        # Iterate through the layers, and add them to the QGIS project.
        for i, lyr in df_lyrs.iterrows():
            # Find the appropriate group for the current set of sinkhole data.
            res_group: QgsLayerTreeGroup = root.findGroup(lyr["name"].replace('Depressions', 'DEM'))

            # Create a QGIS layer from the vector file, and add it to the group
            qlyr: QgsVectorLayer = QgsVectorLayer(f'{str(path_vec)}|layername={lyr["name"]}',
                                                  lyr["name"],
                                                  'ogr')
            qlyr_map: QgsVectorLayer = self.qgs_proj.addMapLayer(qlyr,
                                                                 False)
            qlyr_tree: QgsLayerTreeLayer = res_group.insertLayer(0, qlyr_map)

            # Set the renderer on the layer
            qlyr_map.setRenderer(qrend)
            qlyr_map.triggerRepaint()
            qlyr_map.setOpacity(0.6)

        # After all layers have been added and rendered, save the project file.
        self.qgs_proj.write()

    def click_view(self):
        """
        Function that opens the project in the QGIS Desktop application. Function assumes that the default application
        for opening QGIS project files is the QGIS desktop application.

        :return: None. Opens the project file in QGIS
        """
        # Save the project to file
        self.qgs_proj.write()

        # Open the file with QGIS.
        if Path(qgs_exe).is_file():
            self.wgt_message.print_message('Opening project file in QGIS')
            subprocess.Popen([qgs_exe, self.qgs_proj.fileName()], shell=True)
        else:
            self.wgt_message.print_message('Could not find QGIS executable. Open the project file in QGIS directly.')


# =====================================================================================================================
# FUNCTIONS
# =====================================================================================================================
# Utility Functions
def valid_filename(txt: str) -> str:
    """
    This function takes an input string and removes spaces and invalid characters to make it safe to be used as a
    filename.

    :param txt: Text to be modified

    :return:    String containing the valid filename
    """
    txt = re.sub(r'[/\\:*?"<>|]', '_', txt)
    txt = txt.strip()
    txt = txt.strip('.')
    txt = txt.replace(' ', '_')
    return txt


def epsg_codes_from_wkt(wkt: str) -> dict[str, str]:
    """
    This function returns the EPSG code from a Well-Known Text representation of a Coordinate Reference System. It is
    separated into horizontal and vertical components if applicable.

    :param wkt: Well-Known Text to be decoded.
    :type wkt:  str

    :return:    String representation of the EPSG code.
    """
    # Create the epsg dictionary that will be used to return the result.
    epsg = {'Horizontal EPSG': '',
            'Vertical EPSG': ''}

    # Determine if the well known text is in the correct format
    if wkt is None:
        return epsg

    if not pyproj.crs.is_wkt(wkt):
        return epsg

    crs: pyproj.CRS = pyproj.CRS.from_wkt(wkt)
    crs_list: list[pyproj.CRS] = crs.sub_crs_list
    if not crs.is_compound:
        crs_list.append(crs)

    for comp in crs_list:
        comp: pyproj.CRS
        if comp.is_vertical:
            epsg['Vertical EPSG'] = str(comp.to_epsg(25))
        else:
            epsg['Horizontal EPSG'] = str(comp.to_epsg(25))

    return epsg


def combine_epsg_codes(
        horizontal: str = '',
        vertical: str = ''
) -> str:
    """
    Given the EPSG codes for horizontal and vertical coordinate systems, return the well known text for a combined
    coordinate system. If only one coordinate system is provided, return that coordinate reference system.

    :param horizontal: EPSG for the horizontal coordinate reference system
    :type horizontal:  str

    :param vertical:   EPSG for the vertical coordinate reference system
    :type vertical:    str

    :return:           String well known text representation of the output coordinate reference system.
    """
    # Create the input coordinate reference systems
    if horizontal != '':
        crs_h: pyproj.CRS = pyproj.CRS.from_epsg(horizontal)
    else:
        crs_h = None

    if vertical != '':
        crs_v: pyproj.CRS = pyproj.CRS.from_epsg(vertical)
    else:
        crs_v = None

    # If both coordinate reference systems have been provided, combine them.
    if crs_h and crs_v:
        out_name: str = f'{crs_h.name} + {crs_v.name}'
        out: pyproj.CRS = pyproj.crs.CompoundCRS(name=out_name,
                                                 components=[crs_h, crs_v])
    # If only one CRS has been provided, set it to be the output CRS
    elif crs_h and not crs_v:
        out = crs_h
    elif crs_v and not crs_h:
        out = crs_v
    # If no coordinate reference systems have been provided, return an empty string.
    else:
        return ''

    # Return the well known text of the output CRS
    return out.to_wkt()


def neighbours(arr: np.ndarray) -> np.ndarray:
    """
    Function to imitate the ArcGIS Pro Spatial Analyst Filter tool when set to low band pass. This tool performs a
    multidimensional convolution on the raster that computes the average of the surrounding 3x3 nearest neighbours to
    each pixel. To replicate this result, the convolve function in the scipy ndimage python package is used.

    :param arr:   Array representation of a raster image.
    :type arr:    np.ndarray

    :return:      None return, results of operation are placed on the queue
    """
    # Create a kernel to be used for convolution operations that applies an equal weight to all cells.
    kernel: np.ndarray = np.ones((3, 3))

    # Using the convolve tool on a raster where NODATA cells have been given a value of 0 to compute the sum of
    # pixel cell and its 8 nearest neighbours.
    arr_sum: np.ndarray = ndimage.convolve(np.where(arr == -9999., 0, arr),
                                           kernel,
                                           mode='constant',
                                           cval=0.0)

    # Using the convolve tool on a raster where NODATA values have been given a value of 0 and all other cells have
    # been given a value of 1 to determine the number of nearest neighbours for average calculation.
    arr_nn: np.ndarray = ndimage.convolve(np.where(arr == -9999., 0., 1.),
                                          kernel,
                                          mode='constant',
                                          cval=0.0)

    # To avoid dividing by 0, change 0 values to NaN
    arr_nn[arr_nn == 0.] = np.nan

    # Dividing the sum array by the nearest neighbours array provides the average value of each cell and its
    # surrounding values. Using the original raster array, replace no data areas in the resultant raster.
    arr_sum = arr_sum / arr_nn
    arr_sum = np.where(arr == -9999., -9999., arr_sum)

    # Return the result
    return arr_sum


# ----------------------------------------------------------------------------------------------------------------------
# Multiprocessing Functions
def read_input_las_metadata(
        file_name: str,
        cloud_type: str
) -> dict:
    """
    This function reads in a LAS/LAZ file to retrieve the file's metadata. Reading a single point speeds up read time.

    Sample Dictionary::

        {
            name:             'file.las',
            filename:         'C:/path/to/file.las',
            Size (bytes):     12458734
            cloud_type:       'Input Cloud',
            # Points:         123456789,
            Classified:      'Unknown',
            min x:            415000.0,
            max x:            415999.9,
            min y:            4980000.0,
            max y:            4980999.9
            Horizontal EPSG: '2961',
            Vertical EPSG:   '6647',
            wkt:             GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",...,
            read_time:       0.27,
            extent:          'POLYGON ((415000.0 4980000.0,
                                        415999.9 4980000.0,
                                        415999.9 4980999.9,
                                        415000.0 4980999.9,
                                        415000.0 4980000.0))',
        }

    :param file_name:     Path to the LAS/LAZ file to be read.
    :type file_name:      str

    :param cloud_type:    Type of cloud metadata to be stored. Either Input Cloud or Tile Cloud.
    :type cloud_type:     str

    :return:              dict containing relevant metadata for the file read in.
    """
    # Create a pipeline to read the las file.
    pl: pdal.Pipeline = pdal.Reader(filename=file_name,
                                    count=1).pipeline()

    # Execute the pipeline and record how long it takes to read it.
    t_start = time()
    _ = pl.execute()
    md: dict = pl.metadata['metadata']['readers.las']

    out_dict = {
        'name': Path(file_name).name,
        'filename': file_name,
        'Size (bytes)': Path(file_name).stat().st_size,
        'cloud_type': cloud_type,
        '# Points': md['count'],
        'Classified': 'Unknown',
        'min x': float(md['minx']),
        'max x': float(md['maxx']),
        'min y': float(md['miny']),
        'max y': float(md['maxy'])
    }

    # Determine the SRS of the input las file.
    if 'compoundwkt' in md['srs']:
        out_dict |= epsg_codes_from_wkt(md['srs']['compoundwkt'])
        out_dict['wkt'] = md['srs']['compoundwkt']
    else:
        if 'horizontal' in md['srs']:
            out_dict['Horizontal EPSG'] = epsg_codes_from_wkt(md['srs']['horizontal'])['Horizontal EPSG']
            out_dict['wkt'] = md['srs']['horizontal']
        else:
            out_dict['Horizontal EPSG'] = ''
        if 'vertical' in md['srs']:
            out_dict['Vertical EPSG'] = epsg_codes_from_wkt(md['srs']['vertical'])['Vertical EPSG']
        else:
            out_dict['Vertical EPSG'] = ''

    out_dict['wkt'] = combine_epsg_codes(out_dict['Horizontal EPSG'],
                                         out_dict['Vertical EPSG'])

    out_dict['Read Time (s)'] = time() - t_start
    out_dict['extent'] = (f'POLYGON (({md["minx"]} {md["miny"]}, '
                          f'{md["maxx"]} {md["miny"]}, '
                          f'{md["maxx"]} {md["maxy"]}, '
                          f'{md["minx"]} {md["maxy"]}, '
                          f'{md["minx"]} {md["miny"]}))')

    # Return the metadata dictionary and the relevant data from the points array.
    return out_dict


def read_point_cloud_chunk(
        filename: str,
        start_point: int,
        no_points: int,
        wkt_in: str,
        wkt_out: str
) -> np.ndarray:
    """
    Function that reads a selection of points from a las/laz point cloud file. Intended to be used within a
    multiprocessing pool. This script will also reproject points to a different spatial reference system if required.

    :param filename:    Path to the point cloud to be read.
    :type filename:     str

    :param start_point: Index of the first point in the file to be read.
    :type start_point:  int

    :param no_points:   Number of points to be read from the file.
    :type no_points:    int

    :param wkt_in:      Well-known text representation of the spatial reference system for the input file.
    :type wkt_in:       str

    :param wkt_out:     Well-known text representation of the desired output spatial reference system
    :type wkt_out:      str

    :return:            Numpy structured array of the selected points
    """
    # Construct the pipeline to read in a chunk of points.
    pl: pdal.Pipeline = pdal.Reader(filename=filename,
                                    start=start_point,
                                    count=no_points).pipeline()

    # If the input spatial reference system does not match the output spatial reference system, reproject the data.
    if wkt_in != wkt_out:
        pl |= pdal.Filter.reproject(out_srs=wkt_out)

    # Run the pipeline, and return the points array.
    _ = pl.execute()

    return pl.arrays[0]


def read_las_points_in_chunks(
        filename: str,
        no_points: int,
        chunk_size: int,
        wkt_in: str,
        wkt_out: str,
        bar_pos: int,
        no_cpus: int,
        queue: Queue
) -> np.ndarray:
    """
    Function that reads a las/laz point cloud file in chunks using a multiprocessing pool, and reports its results back
    to a multiprocessing queue.

    :param filename:   Path to las/laz point cloud file to be read.
    :type filename:    str

    :param no_points:  Number of points in the file to be read.
    :type no_points:   int

    :param chunk_size: Number of points in each chunk to be read.
    :type no_points:   int

    :param wkt_in:     Well-known text representation of the spatial reference system for the input file.
    :type wkt_in:      str

    :param wkt_out:    Well-known text representation of the desired output spatial reference system
    :type wkt_out:     str

    :param bar_pos:    Position of the progress bar in the progress bar widget to be updated.
    :type bar_pos:     int

    :param no_cpus:    Number of CPUs to be used for pool processing
    :type no_cpus:     int

    :param queue:      Multiprocessing queue in which to leave progress updates.
    :type queue:       multiprocessing.Queue

    :return:           Numpy structured array of the points in the file.
    """
    # Create a pool of chunks
    pool_queue = [i for i in range(0, no_points, chunk_size)]
    n_chunks = len(pool_queue)

    queue.put({
        'pbar_size': (n_chunks, bar_pos),
        'disp_perc': bar_pos,
        'desc': (f'Opening file {Path(filename).name}', bar_pos)
    })

    # Create the pool
    with pool.Pool(no_cpus) as p, Manager() as manager:
        # Create the counter that will be used to report progress
        chunk_counter = manager.Value('i', 0)

        # Create a list of the queued results
        queued_results: list[pool.AsyncResult] = [
            p.apply_async(
                read_point_cloud_chunk,
                args=(filename, start, chunk_size, wkt_in, wkt_out),
                callback=pool_progress(chunk_counter, bar_pos, queue)) for start in pool_queue]

        # Wait for all results to be completed
        for r in queued_results:
            r.wait()

        # Collect the results, and return them as a concatenated array
        results: list[np.ndarray] = [r.get()[['X', 'Y', 'Z', 'Classification']] for r in queued_results]

        return np.concatenate(results)


def classify_ground_pts(
        pts_arr: np.ndarray,
        queue: Queue
):
    """
    Using the Cloth Surface Filtering algorithm in PDAL, classify points as being ground/not-ground and return the
    results of the operation to the provided queue. This function is intended to be used as a multiprocessing Process.
    Details about cloth surface filtering can be found here: Remote sensing, 8(6):501, 2016.

    :param pts_arr: Point cloud points to be classified.
    :type pts_arr:  numpy.ndarray

    :param queue:   Queue to place the results when complete.
    :type queue:    multiprocessing.Queue

    :return:        None. Results of the operation are placed on the queue.
    """
    # Construct the pipeline that will be used to classify ground points and execute it.
    pl: pdal.Pipeline = pdal.Filter.csf(resolution=0.5,
                                        threshold=0.5).pipeline(pts_arr)
    _ = pl.execute()

    queue.put(pl.arrays[0][['X', 'Y', 'Z', 'Classification']])


def create_dem_tile(
        in_pts: np.ndarray,
        out_path: str,
        resolution: float,
        tile_box: Polygon,
        srs_wkt: str,
        queue: Queue
) -> None:
    """
    Function that writes a collection of points to raster. Intended to be run in a Multiprocessing process.

    :param in_pts:     Structured array of points to be written to file.
    :type in_pts:      numpy.ndarray

    :param out_path:   Path to the destination file to be written.
    :type out_path:    str

    :param resolution: Raster cell size in metres.
    :type resolution:  float

    :param tile_box:   Shape of the raster to be generated.
    :type tile_box:    shapely.Polygon

    :param srs_wkt:    Well-known text representation of the spatial reference system for the output raster.
    :type srs_wkt:     str

    :param queue:      Queue to place the results when complete.
    :type queue:       multiprocessing.Queue

    :return:           None. Time to complete the operation is placed on the queue.
    """

    # Record the time it takes to write the file
    start_time = time()

    # Compute the position of the southwest corner of the tile
    origin_x, origin_y, maxx, maxy = tile_box.bounds

    width = int(np.floor((maxx - origin_x) / resolution))
    height = int(np.floor((maxy - origin_y) / resolution))

    # Create a pipeline that will save the points to a DEM.
    pl: pdal.Pipeline = pdal.Writer.gdal(filename=out_path,
                                         resolution=resolution,
                                         output_type='mean',
                                         power=2.0,
                                         origin_x=origin_x,
                                         origin_y=origin_y,
                                         width=width,
                                         height=height,
                                         default_srs=srs_wkt).pipeline(in_pts)

    # Execute the pipeline
    _ = pl.execute()

    # Report completion to the queue.
    queue.put(time() - start_time)


def write_las_process(
        pts_out: np.ndarray,
        filename: str,
        wkt_out: str,
        queue: Queue
):
    """
    Function that writes a collection of points to a las/laz file. Intended to be run in a Multiprocessing Process

    :param pts_out:  Points to be written to file.
    :type pts_out:   numpy.ndarray

    :param filename: Path to the file to be written.
    :type filename:  str

    :param wkt_out:  Well-known text representation of the spatial reference system for the output file.
    :type wkt_out:   str

    :param queue:    Multiprocessing queue in which to post the results of the write operation.
    :type queue:     multiprocessing.Queue

    :return:         None. Write time in seconds is placed on the queue.
    """
    # Record the time it takes to write the file
    start_time = time()
    # Write the points to file, re-projecting the data if necessary
    pl: pdal.Pipeline = pdal.Writer(filename=filename,
                                    a_srs=wkt_out,
                                    compression=True).pipeline(pts_out)

    # Execute the pipeline.
    _ = pl.execute_streaming()

    # Report completion to the queue.
    queue.put(time() - start_time)


def pool_progress(
        counter: Manager,
        bar_pos: int,
        queue: Queue
):
    """
    Convenience function for multiprocessing progress updating.

    :param counter: Multiprocessing manager containing a value.
    :type counter:  multiprocessing manager.

    :param bar_pos: Position of the progress bar to be updated.
    :type bar_pos:  int

    :param queue:   Multiprocessing queue to emit progress signals.
    :type queue:    multiprocessing.Queue

    :return:        None, updates the counter value in place, and emits appropriate signals.
    """
    counter.value += 1

    queue.put({'progress': (counter.value, bar_pos)})


# ----------------------------------------------------------------------------------------------------------------------
# Parallel functions
def finalize_inputs(
        proj_file: Path,
        data: gui.ProjectData,
        proj_wkt: str,
        queue: Queue
) -> None:
    """
    Function intended to be used in a multiprocessing process to create the geopackage that will be used to store vector
    data for the project. Creates the file and adds the layers Point_Clouds, AOI, and Water_features.

    :param proj_file: Path to the QGIS project file
    :type proj_file:  pathlib.Path

    :param data:     Project data object containing the data to be processed.
    :type data:      COLDS_util.COLDS_gui.ProjectData

    :param proj_wkt: Well-known text description of the project's spatial reference system.
    :type proj_wkt:  str

    :param queue:    Queue to be used to pass results back to main thread.
    :type queue:     multiprocessing.Queue

    :return:         None. Submits one-tuple of a ProjectData object to the queue upon completion.
    """

    # Set the Spatial Reference System for the point cloud metadata geodataframe.
    pc_wkt_list = data.gdf_pc_md.drop_duplicates(subset=['wkt'])['wkt'].to_numpy()

    # Create the path for the output geopackage
    out_file: Path = proj_file.with_suffix('.gpkg')

    # Reset the partial progress bar for the point cloud extents processing
    queue.put({
        'desc': ('Creating point cloud extents', 1),
        'pbar_size': (len(data.gdf_pc_md), 1)
    })
    processed = 0
    pc_gdf_list: list[gpd.GeoDataFrame] = []

    # Iterate through the different wkt descriptions of the spatial reference systems in the point cloud dataframe,
    # and set the SRS for the elements that match it.
    for wkt in pc_wkt_list:
        pc_gdf_list.append(data.gdf_pc_md[data.gdf_pc_md.wkt == wkt].copy())
        pc_gdf_list[-1].set_crs(crs=wkt)

        # If a subset of the dataframe has an SRS that does not match the project's SRS, reproject the data to the
        # correct SRS
        if wkt != proj_wkt:
            pc_gdf_list[-1].to_crs(crs=proj_wkt,
                                   inplace=True)

        # Emit the progress signals
        processed += len(pc_gdf_list[-1])
        queue.put({'progress': (processed, 1)})
        queue.put({'progress': (10 + int(20 * processed / (len(data.gdf_pc_md) + 1)), 0)})

    # Concatenate the results, and emit a signal indicating this step is done
    data.gdf_pc_md = pd.concat(pc_gdf_list, axis=0)
    queue.put({
        'progress': (30, 0),
        'message': 'Point cloud extents created'
    })

    # Save the point cloud extents to a geopackage that will hold all project layers.
    queue.put({'desc': ('Opening area of interest layers', 1)})
    data.gdf_pc_md.to_file(str(out_file),
                           layer='Point_Clouds',
                           crs=proj_wkt,
                           engine='fiona')
    queue.put({
        'msg': 'Point cloud extents saved to GeoPackage',
        'progress': (35, 0)
    })

    # Generate the Area of interest for the project.
    if 'AOI' not in data.gdf_vec_md.values:
        # If no Area of Interest vector layers have been provided, use the extents of the input point clouds.
        queue.put({
            'msg': '<b>WARNING: </b>No area of interest has been provided. Point cloud extents will be used as the area'
                   ' of interest.',
            'pbar_size': (1, 1)
        })

        data.gdf_aoi = data.gdf_pc_md.copy()

        # Rename the cloud type field to the layer type field for standardization of outputs
        data.gdf_aoi.rename(columns={'cloud_type': 'layer_type'},
                            inplace=True)
        data.gdf_aoi.set_crs(crs=proj_wkt,
                             inplace=True)
        queue.put({'progress': (50, 0)})
    else:
        # Prepare the partial progress bar for new signals
        filt: np.ndarray[bool] = data.gdf_vec_md.layer_type == 'AOI'
        queue.put({'pbar_size': (np.count_nonzero(filt), 1)})

        for i, aoi in data.gdf_vec_md[data.gdf_vec_md.layer_type == 'AOI'].iterrows():
            gdf_aoi: gpd.GeoDataFrame = gpd.read_file(aoi['filename'],
                                                      layer=aoi['name'])
            # Check the SRS of the input file, and set or reproject the data if required.
            file_crs: pyproj.CRS | None = gdf_aoi.crs
            if file_crs is None:
                queue.put({
                    'msg': f'<b>WARNING: </b>Layer {aoi["name"]} in file {aoi.filename} has no spatial reference system'
                           f' defined. Its SRS will be set to the current project SRS.'
                })
                gdf_aoi.set_crs(crs=proj_wkt,
                                inplace=True)
            elif file_crs.to_wkt() != proj_wkt:
                gdf_aoi.to_crs(crs=proj_wkt,
                               inplace=True)

            # Save the relevant information for the layers being added
            gdf_aoi['layer_type'] = aoi['layer_type']
            gdf_aoi = gdf_aoi[['layer_type', 'geometry']]

            # Concatenate the loaded file with the AOI GeoDataFrame and emit progress signals
            data.gdf_aoi = pd.concat([data.gdf_aoi, gdf_aoi], axis=0)
            queue.put({'progress': (i + 1, 1)})
            queue.put({'progress': (35 + int(15 * (i + 1) / np.count_nonzero(filt)), 0)})

    # Dissolve the AOI into a single shape to be used
    data.gdf_aoi = data.gdf_aoi.dissolve(by=['layer_type'])

    queue.put({
        'progress': (55, 0),
        'msg': 'Area of interest defined.'
    })

    # Save the area of interest layer to the project geopackage
    data.gdf_aoi.to_file(str(out_file),
                         layer='AOI',
                         crs=proj_wkt,
                         engine='fiona')
    queue.put({
        'progress': (60, 0),
        'msg': 'Area of interest saved to Geopackage.',
        'desc': ('Opening water feature layers.', 1)
    })

    # Load the water features for the project if they exist.
    if 'Water Feature' in data.gdf_vec_md.values:
        # Prepare the partial progress bar for new signals
        filt: np.ndarray[bool] = data.gdf_vec_md.layer_type == 'Water Feature'
        queue.put({'pbar_size': (np.count_nonzero(filt), 1)})

        for i, wf in data.gdf_vec_md[filt].iterrows():
            # Read the file. Using the fiona engine will ensure that the bounding box is clipped using a matching CRS.
            gdf_wf: gpd.GeoDataFrame = gpd.read_file(wf['filename'],
                                                     layer=wf['name'],
                                                     bbox=data.gdf_aoi.geometry.boundary,
                                                     engine='fiona')

            # Check the SRS of the input file, and set or reproject the data if required.
            file_crs: pyproj.CRS = gdf_wf.crs
            if file_crs is None:
                queue.put({
                    'msg': f'<b>WARING: </b>Layer {wf["name"]} in file {wf.filename} has no spatial reference system '
                           f'defined. Its SRS will be set to the current project SRS.'
                })
                gdf_wf.set_crs(crs=proj_wkt,
                               inplace=True)
            elif file_crs.to_wkt() != proj_wkt:
                gdf_wf.to_crs(crs=proj_wkt,
                              inplace=True)

            # If the water feature layer's geometry type is linestring or linestring z, buffer the features to
            # represent small bodies of water.
            if 'linestring' in wf['geometry_type'].lower():
                gdf_wf.geometry = gdf_wf.geometry.buffer(distance=2)

            # Add source information to the layer
            gdf_wf['filename'] = wf['filename']
            gdf_wf['source_layer'] = wf['name']

            # Concatenate the file with the Water Feature GeoDataFrame, and emit progress signals
            data.gdf_water = pd.concat([data.gdf_water, gdf_wf])
            queue.put({'progress': (i + 1, 1)})
            queue.put({'progress': (60 + int(20 * (i + 1) / np.count_nonzero(filt)), 0)})

        # Send final emits for the process
        queue.put({
            'progress': (80, 0),
            'msg': 'Water features identified.'
        })

        # Save the water features to the project geopackage
        if len(data.gdf_water) > 0:
            data.gdf_water.to_file(str(out_file),
                                   layer='Water_Features',
                                   crs=proj_wkt,
                                   engine='fiona')
            queue.put({'msg': 'Water features saved to Geopackage'})
        else:
            queue.put({'msg': 'No water features found in area of interest'})
    else:
        # If there are no water features found, emit required signals
        queue.put({
            'pbar_size': (1, 1),
            'msg': 'No water features found. Continuing.'
        })

    # Send the process progress emission
    queue.put({'progress': (85, 0)})

    # Save the vector metadata GeoDataFrame to the output file if there is any.
    if len(data.gdf_vec_md) > 0:
        data.gdf_vec_md.to_file(str(out_file),
                                layer='Vector_Metadata')

    queue.put({'result': (None, data)})


def tile_and_merge(
        df: gpd.GeoDataFrame,
        aoi: Polygon,
        tile_size: float,
        proj_name: Path,
        epsg_list: list[str],
        no_cpus: int,
        queue: Queue
):
    """
    Function that splits the project's input point clouds into gridded tiles that encompass the area of interest.
    This function is run in a multiprocessing process to prevent the GUI from freezing during completion.

    :param df:        Dataframe containing the metadata for the input point cloud files.
    :type df:         geopandas.GeoDataFrame

    :param aoi:       Polygon representing the project's area of interest.
    :type aoi:        shapely.Polygon

    :param tile_size: Size of each tile to be created in metres.
    :type tile_size:  float

    :param proj_name: Path to the project file.
    :type proj_name:  pathlib.Path

    :param epsg_list: List of horizontal and vertical spatial reference systems for the project.
    :type epsg_list:  list[str]

    :param no_cpus:   Number of CPUs to be used for multiprocessing pools.
    :type no_cpus:    int

    :param queue:     Queue in which to put progress updates for the main GUI.
    :type queue:      multiprocessing.Queue

    :return:          None. Puts an updated point cloud dataframe on the queue for retrieval from the main GUI upon
                      completion.
    """
    # Create the wkt string for the output point clouds from the epsg list
    wkt_out = combine_epsg_codes(*epsg_list)

    # Use the bounding box extents of the area of interest, subdivide the area of interest into a grid.
    minx, miny, maxx, maxy = aoi.bounds

    # Starting from the northwest corner, tile the area moving east to west, and north to south
    grid: list[Polygon] = []
    y = maxy

    # Iterate over each dimension with a while condition.
    while y > miny:
        x = minx

        # Limit the tile size in the y direction when it extends past the area of interest, but using the ceiling
        # function ensures that the raster is still an integer number of metres, which is important for merging the DEMs
        # in the generate_dem function.
        y_tile_size = np.ceil(min(tile_size, (y - miny)))

        while x < maxx:
            # Limit the tile size in the x direction when it extends past the area of interest
            x_tile_size = np.ceil(min(tile_size, (maxx - x)))

            # Set the bounds of the grid based on the current (x, y) position and tile size
            x0 = x - 5
            y0 = y - y_tile_size - 5
            x1 = x + x_tile_size + 5
            y1 = y + 5

            # Create the box polygon given the bounds and add it to the list if it intersects with the original AOI.
            sq: Polygon = box(x0, y0, x1, y1)
            if sq.intersects(aoi):
                grid.append(sq)
            x += tile_size
        y -= tile_size

    # Determine the folder in which outputs will be saved
    proj_folder = proj_name.parent

    # Create a dataframe of the grid tiles
    gdf_grid: gpd.GeoDataFrame = gpd.GeoDataFrame(geometry=grid,
                                                  crs=df.crs)

    # Add the relevant metadata for the tile geodataframe.
    gdf_grid['Size (bytes)'] = 0
    gdf_grid['Read Time (s)'] = 0.0
    gdf_grid['cloud_type'] = 'Tile Cloud'
    gdf_grid['# Points'] = 0
    gdf_grid['min x'] = 0.0
    gdf_grid['max x'] = 0.0
    gdf_grid['min y'] = 0.0
    gdf_grid['max y'] = 0.0
    gdf_grid['Classified'] = 'False'
    gdf_grid['Horizontal EPSG'] = epsg_list[0]
    gdf_grid['Vertical EPSG'] = epsg_list[1]
    gdf_grid['wkt'] = wkt_out
    gdf_grid['name'] = 'Tile_' + (gdf_grid.index + 1).astype(str).str.zfill(3) + '.laz'

    # Retrieve the project folder from the project name
    gdf_grid['filename'] = f'{proj_folder.as_posix()}/las_files/' + gdf_grid['name']

    queue.put({
        'progress': (1, 1),
        'msg': 'Grid created'
    })

    # Iterate through the input clouds, separating them into the grid tiles.
    for i, pc in df.iterrows():
        # Keep track how long it takes to open the file.
        start_time = time()

        # Retrieve the points from the file
        pc_pts: np.ndarray = read_las_points_in_chunks(filename=pc['filename'],
                                                       no_points=pc['# Points'],
                                                       chunk_size=1_000_000,
                                                       wkt_in=pc['wkt'],
                                                       wkt_out=wkt_out,
                                                       bar_pos=1,
                                                       no_cpus=no_cpus,
                                                       queue=queue)

        df.loc[i, 'Read Time (s)'] += time() - start_time

        # Emit the overall process signal.
        queue.put({'progress': (2 * i + 1, 0)})

        # Determine which grid squares intersect with the current point cloud.
        tile_filt: pd.Series = gdf_grid.intersects(pc.geometry)
        gdf_intersect: gpd.GeoDataFrame = gdf_grid[tile_filt].reset_index()

        queue.put({
            'pbar_size': (len(gdf_intersect), 1),
            'desc': (f'Tiling file {pc["name"]}', 1)
        })

        # Iterate over the intersecting tiles, and write their points to file.
        for j, gd in gdf_intersect.iterrows():
            # Keep track of how long it takes to process the tile.
            start_time = time()

            # Determine which points lie within the tile
            minx, miny, maxx, maxy = gd.geometry.bounds
            inliers: np.ndarray = np.logical_and(
                np.logical_and(pc_pts['X'] >= minx, pc_pts['X'] <= maxx),
                np.logical_and(pc_pts['Y'] >= miny, pc_pts['Y'] <= maxy),
            )

            pts_out: np.ndarray = pc_pts[inliers]

            # If there are no points in the grid tile, emit signals and move on to the next
            if len(pts_out) == 0:
                queue.put({'progress': (j + 1, 1)})
                continue

            # If some tile points have already been written to file, read them into memory, and add them to pts_out
            out_file: Path = Path(gd['filename'])
            if out_file.is_file():
                # Read the points into memory, and add them to the output points
                existing_pts: np.ndarray = read_las_points_in_chunks(filename=gd['filename'],
                                                                     no_points=gd['# Points'],
                                                                     chunk_size=1_000_000,
                                                                     wkt_in=wkt_out,
                                                                     wkt_out=wkt_out,
                                                                     bar_pos=2,
                                                                     no_cpus=no_cpus,
                                                                     queue=queue)
                pts_out = np.concatenate([pts_out, existing_pts])

            # Write the points to file in a separate process, using a child queue to monitor progress.
            child_queue: Queue = Queue()
            write_process: Process = Process(target=write_las_process,
                                             args=(pts_out,
                                                   gd['filename'],
                                                   gd['wkt'],
                                                   child_queue))

            # Start the process
            write_process.start()

            # While the process is running collect statistics from the point cloud
            cur_cloud: np.ndarray = gdf_grid["name"] == gd["name"]
            gdf_grid.loc[cur_cloud, '# Points'] = len(pts_out)
            gdf_grid.loc[cur_cloud, 'min x'] = np.min(pts_out['X'])
            gdf_grid.loc[cur_cloud, 'max x'] = np.max(pts_out['X'])
            gdf_grid.loc[cur_cloud, 'min y'] = np.min(pts_out['Y'])
            gdf_grid.loc[cur_cloud, 'max y'] = np.max(pts_out['Y'])
            if 'Classification' in pts_out.dtype.names:
                gdf_grid.loc[cur_cloud, 'Classified'] = str(np.any(pts_out['Classification'] == 2))

            # Create a while loop that checks the queue and ensures the thread is still running.
            loop = True
            while loop:
                sleep(1)
                if not child_queue.empty():
                    break

                if not write_process.is_alive():
                    loop = False

            if not loop:
                queue.put({'error': f'Process to merge {gd["filename"]} crashed without providing a result.'})
                return

            # Get the time to split the input cloud, and add that information to the dataframe time
            _ = child_queue.get()
            gdf_grid.loc[cur_cloud, 'Read Time (s)'] += time() - start_time

            # Compute the file's size, and add it to the dataframe data
            gdf_grid.loc[cur_cloud, 'Size (bytes)'] = Path(gd['filename']).stat().st_size

            # Emit the signals for completion of the current grid square.
            queue.put({
                'progress': (j + 1, 1),
            })

            # Terminate the write process if it is still running.
            write_process.terminate()

            # Remove the points from memory before moving to the next tile.
            del pts_out
            gc.collect(2)

        # Remove the file points from memory and emit signals before moving to next file.
        del pc_pts
        gc.collect(2)

        queue.put({
            'progress': (2 * i + 2, 0),
            'msg': f'Completed tiling file: {pc["name"]}'
        })

    # Remove any grids that have no points
    gdf_grid = gdf_grid[gdf_grid['# Points'] > 0]

    # Concatenate the tile geodataframe to the original geodataframe, and submit the geodataframe to the queue.
    df = pd.concat([df, gdf_grid],
                   axis=0,
                   ignore_index=True)

    queue.put({'result': (df, None)})

    # Save the dataframe to file.
    vec_file_out: Path = proj_name.with_suffix('.gpkg')
    df.to_file(str(vec_file_out),
               layer='Point_Clouds',
               crs=wkt_out,
               engine='fiona')


def generate_dem(
        gdf_tiles: gpd.GeoDataFrame,
        home_path: str,
        dem_res: list[str],
        class_flag: bool,
        no_cpus: int,
        queue: Queue
):
    """
    Function that generates a DEM for the project from the input point cloud tiles. If indicated by class_flag, points
    are classified as ground/not ground during this process. This function is run in a multiprocessing process to
    prevent the GUI from freezing during completion.

    :param gdf_tiles:  GedDataFrame of the point cloud tiles to be processed.
    :type gdf_tiles:   geopandas.GeoDataFrame

    :param home_path:  Path for the project to be used for raster path creation.
    :type home_path:   str

    :param dem_res:    List of output resolutions for the DEM rasters.
    :type dem_res:     list[str]

    :param class_flag: Flag indicating if point clouds should undergo ground/not ground classification.
    :type class_flag:  bool

    :param no_cpus:    Number of cpus to be used for pool processes.
    :type no_cpus:     int

    :param queue:      Queue to post intermediate results of the function.
    :type queue:       multiprocessing.Queue

    :return:           None return, results are put on the queue.
    """
    # Create a child queue to collect the results from child processes.
    child_queue: Queue = Queue()

    # Determine the number of steps that will be taken for each tile based on the inputs
    # One step is for reading the file
    # If the file is classified 1 step for classification
    # One step for each raster to be created
    # One step for resaving the tile as just the ground points for memory efficiency
    tile_steps = 2 + int(class_flag) + len(dem_res)

    # Iterate through each of the tiles. Resetting the index provides a counter for the overall progress of the function
    for i, tile in gdf_tiles.reset_index().iterrows():
        # Emit the signals to set the child process bar
        cur_step = 0
        queue.put({
            'desc': (f'Reading {tile["name"]}', 1),
            'pbar_size': (tile_steps, 1),
            'disp_perc': 1
        })

        # Open the tile and retrieve the points
        tile_pts: np.ndarray = read_las_points_in_chunks(filename=tile['filename'],
                                                         no_points=tile['# Points'],
                                                         chunk_size=1_000_000,
                                                         wkt_in=tile['wkt'],
                                                         wkt_out=tile['wkt'],
                                                         bar_pos=2,
                                                         no_cpus=no_cpus,
                                                         queue=queue)

        cur_step += 1
        queue.put({
            'desc': (f'Processing {tile["name"]}', 1),
            'progress': (cur_step, 1)
        })

        # In a new process, classify points as ground / not ground if desired
        if class_flag:
            # Keep track of the time it takes to classify the points
            start_time = time()

            # Create the classification process, and start it
            classify_process: Process = Process(target=classify_ground_pts,
                                                args=(tile_pts,
                                                      child_queue))

            classify_process.start()

            # Create a while loop that checks the queue and ensures the thread is still running.
            loop = True
            while loop:
                sleep(1)
                if not child_queue.empty():
                    break

                if not classify_process.is_alive():
                    loop = False

            if not loop:
                queue.put({'error': f'Process to classify {tile["filename"]} crashed without providing a result.'})
                return

            # Retrieve the classified points from the queue
            tile_pts = child_queue.get()

            # Update the time taken to classify the file
            gdf_tiles.loc[gdf_tiles['name'] == tile['name'], 'Read Time (s)'] += time() - start_time

            # Emit update signals, and terminate the process if it is still running
            cur_step += 1
            queue.put({'progress': (cur_step, 1)})
            if classify_process.is_alive():
                classify_process.terminate()

        # Keep only points classified as ground for subsequent operations.
        tile_pts = tile_pts[tile_pts['Classification'] == 2]
        queue.put({'msg': f'{tile["name"]} classified. {len(tile_pts):,} points identified as ground.'})

        # Save the filtered points to file, and update the number of points in the dataframe.
        write_cloud_process: Process = Process(target=write_las_process,
                                               args=(tile_pts,
                                                     tile['filename'],
                                                     tile['wkt'],
                                                     child_queue))
        write_cloud_process.start()

        gdf_tiles.loc[gdf_tiles['name'] == tile['name'], '# Points'] = len(tile_pts)

        # Create a while loop that checks the queue and ensures the thread is still running.
        loop = True
        while loop:
            sleep(1)
            if not child_queue.empty():
                break

            if not write_cloud_process.is_alive():
                loop = False

        if not loop:
            queue.put({'error': f'Process to write {tile["filename"]} crashed without providing a result.'})
            return

        # Retrieve the classified points from the queue
        gdf_tiles.loc[gdf_tiles['name'] == tile['name'], 'Read Time (s)'] += child_queue.get()

        # Emit progress signals
        cur_step += 1
        queue.put({'progress': (cur_step, 1)})

        # Iterate through each of the DEM resolutions, and create the DEM tile for that resolution
        for resolution in dem_res:
            # Create the output filename for the raster tile.
            tile_name = Path(tile['filename']).stem + f'_{resolution}.tif'
            out_path: Path = Path(home_path) / 'rasters' / 'tiles' / resolution / tile_name

            # Create and start the process to generate the raster
            rast_process: Process = Process(target=create_dem_tile,
                                            args=(tile_pts,
                                                  str(out_path),
                                                  float(resolution.replace('_cm', '')) / 100,
                                                  tile['geometry'],
                                                  tile['wkt'],
                                                  child_queue))
            rast_process.start()

            # Create a while loop that checks the queue and ensures the thread is still running.
            loop = True
            while loop:
                sleep(1)
                if not child_queue.empty():
                    break

                if not rast_process.is_alive():
                    loop = False

            if not loop:
                queue.put({'error': f'Process to write {out_path} crashed without providing a result.'})
                return

            _ = child_queue.get()

            # Emit progress signals
            cur_step += 1
            queue.put({'progress': (cur_step, 1)})
        queue.put({'progress': (i + 1, 0)})

    # For each resolution, merge the tiles into a single raster, and fill gaps with no data.
    for resolution in dem_res:
        # Emit signals for the second progress bar
        queue.put({
            'pbar_size': (2, 1),
            'disp_perc': 1,
            'desc': (f'Creating DEM_{resolution}.tif', 1)
        })

        # Retrieve the folder containing the tile files
        res_folder: Path = Path(home_path) / 'rasters' / 'tiles' / resolution
        out_path: Path = Path(home_path) / 'rasters' / f'DEM_{resolution}.tif'

        # Retrieve all the rasters from the input directory
        rast_list: list[gdal.Dataset] = [gdal.OpenEx(str(rast)) for rast in res_folder.glob('*.tif')]

        # For each of the rasters, identify their size in the x and y direction and their geotransform
        rast_data = {
            'name': [ds.GetName() for ds in rast_list],
            'x': [ds.RasterXSize for ds in rast_list],
            'y': [ds.RasterYSize for ds in rast_list],
            'gt': [ds.GetGeoTransform() for ds in rast_list]
        }

        df_rast: pd.DataFrame = pd.DataFrame(data=rast_data)
        df_rast = pd.concat(
            [df_rast,
             pd.DataFrame(df_rast['gt'].tolist(),
                          index=df_rast.index,
                          columns=[f'gt_{i}' for i in range(6)])],
            axis=1)

        # Identify the north-west corner of the merged raster.
        origin: np.ndarray = np.array([np.min(df_rast['gt_0']), np.max(df_rast['gt_3'])])

        # For each tile, determine how many cells away their origin is from the main origin using the geotransform data
        df_rast['i_x'] = (df_rast['gt_0'] - origin[0]) / df_rast['gt_1']
        df_rast['i_y'] = (df_rast['gt_3'] - origin[1]) / df_rast['gt_5']

        # Determine the cell position of the opposite corner for the raster
        df_rast['j_x'] = df_rast['i_x'] + df_rast['x']
        df_rast['j_y'] = df_rast['i_y'] + df_rast['y']

        # Make all array index columns integer values
        df_rast[['i_x', 'i_y', 'j_x', 'j_y']] = df_rast[['i_x', 'i_y', 'j_x', 'j_y']].astype(int)

        # Find the maximum values of j_x and j_y, and use those values to create a new raster
        arr_merge: np.ndarray = np.zeros((2, np.max(df_rast['j_y']), np.max(df_rast['j_x'])))

        for idx, ds in enumerate(rast_list):
            rast: pd.Series = df_rast.iloc[idx]
            rast_arr: np.ndarray = ds.ReadAsArray()
            arr_merge[0, rast['i_y']:rast['j_y'], rast['i_x']:rast['j_x']] += np.where(rast_arr != -9999.,
                                                                                       rast_arr,
                                                                                       0)
            arr_merge[1, rast['i_y']:rast['j_y'], rast['i_x']:rast['j_x']] += np.where(rast_arr != -9999.,
                                                                                       1,
                                                                                       0)
        arr_merge = np.where(arr_merge[1] == 0, np.nan, arr_merge)
        arr_merge = arr_merge[0] / arr_merge[1]
        arr_merge = np.where(np.isnan(arr_merge), -9999., arr_merge)

        # Create the driver needed to create the output raster.
        drv: gdal.Driver = gdal.GetDriverByName('GTiff')

        # Create the output raster, and save to filee
        with drv.Create(str(out_path),
                        xsize=int(np.max(df_rast['j_x'])),
                        ysize=int(np.max(df_rast['j_y'])),
                        bands=1,
                        eType=gdal.GDT_Float64) as ds_out:

            # Set the spatial reference of the output raster based on one of the tile rasters
            ds_out.SetSpatialRef(rast_list[0].GetSpatialRef())

            # Get the GeoTransform for the raster in the northwest corner of the grid using the values of gt_0 and gt_3
            gt_out = df_rast.loc[(df_rast['gt_0'] == origin[0]) & (df_rast['gt_3'] == origin[1]), 'gt']
            ds_out.SetGeoTransform(gt_out.values[0])
            ds_out.GetRasterBand(1).SetNoDataValue(-9999.)

            # Write the array to disk
            ds_out.WriteArray(arr_merge)

            # Emit progress signals
            queue.put({'progress': (1, 1)})

            # Fill regions of no data
            gdal.FillNodata(ds_out.GetRasterBand(1),
                            None,
                            20.,
                            0)

        # Emit progress signals
        queue.put({'progress': (2, 1)})
        i += 1
        queue.put({
            'progress': (i, 0),
            'msg': f'DEM_{resolution}.tif created.'
        })

    # Place the updated tile geodataframe on the queue.
    queue.put({'result': (gdf_tiles, None)})


def main():
    """
    Main program loop for COLDS.py

    :return: None
    """
    # Create and show the main window
    colds_window = Colds(qgs)
    colds_window.show()

    # Run the application
    qgs.exec()

    # Upon completion, exit QGIS
    qgs.exitQgis()


if __name__ == '__main__':
    gdal.UseExceptions()              # Can be removed when GDAL 4.0 is released.
    set_start_method('spawn')         # Explicitly defines multiprocessing process creation method.

    main()
