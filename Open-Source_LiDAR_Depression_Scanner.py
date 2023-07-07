# =====================================================================================================================
# Filename:     Open-Source_LiDAR_Depression_Scanner.py
# Version:      1.1.1
# Written by:   Keith Cusson                Date: Jun 2023
# Description:  A tool to detect sinkholes in QGIS from a classified LiDAR point
#               cloud. Based on NS_Sinkholes_Tool.tbx created by Mitch Maracle
#               in ArcGIS Pro.
# License:      MIT License (c) 2023 Keith Cusson
# =====================================================================================================================
# ---------------------------------------------------------------------------------------------------------------------
# Import packages
# ---------------------------------------------------------------------------------------------------------------------
from osgeo import ogr, osr, gdal               # Import python libraries for geographic data and spatial reference
import pdal                                    # Import PDAL python interface
import json                                    # Import json handlers
import numpy                                   # Import numpy array handlers for handling metadata
import psutil                                  # Import system tools
import time                                    # Import time tools for event timing
import os                                      # Import file manipulation tools
import PySimpleGUI as Gui                      # Import gui tools
from qgis.core import *                        # Import QGIS functionality
from qgis.PyQt import QtGui


# ---------------------------------------------------------------------------------------------------------------------
# QGIS installation location
# ---------------------------------------------------------------------------------------------------------------------
# As appropriate, change these values to match your QGIS installation location
qgisPrefix = 'C:/OSGeo4W/apps/qgis'                     # QGIS Installation location

# ---------------------------------------------------------------------------------------------------------------------
# Initialize all elements that will be used throughout the script
# ---------------------------------------------------------------------------------------------------------------------
# Initialize the QGIS environment and initialize processing
QgsApplication.setPrefixPath(qgisPrefix, True)          # Defines the location of QGIS
qgs = QgsApplication([], False)                         # Start the application
qgs.initQgis()                                          # Initialize QGIS

# Processing must be imported after QGIS has been initialized
import processing                                       # Import QGIS processing tools
processing.core.Processing.Processing.initialize()      # Initialize processing to allow the use of QGIS functions

# Set the GUI theme and options
Gui.theme('Dark Blue 3')
Gui.set_options(font=('Arial', 11))

# Set OSGEO shapefile driver
shpDriver = ogr.GetDriverByName('ESRI Shapefile')


# =====================================================================================================================
# CLASSES
# =====================================================================================================================
# This class corresponds to project data
class Project(object):
    # Version info
    version = '1.1.1'

    # Project object constructor
    def __init__(self):
        self.name = ''
        self.path = ''
        self.fileName = ''
        self.created = 0.0
        self.saved = 0.0
        self.state = 0
        self.classified = False
        self.AOI = ShapeFile()
        self.waterFeatures = ShapeFile()
        self.inputFiles = PointCloudCollection()
        self.tileFiles = PointCloudCollection()
        self.mergedFiles = PointCloudCollection()
        # Create dictionaries to hold all raster and shapefile information for multiple DEM resolutions
        self.DEMTiles = {'100_cm': RasterCollection(),
                         '50_cm': RasterCollection(),
                         '25_cm': RasterCollection()}
        self.DEM = {'100_cm': Raster(),
                    '50_cm': Raster(),
                    '25_cm': Raster()}
        self.depressions = {'100_cm': ShapeFile(),
                            '50_cm': ShapeFile(),
                            '25_cm': ShapeFile()}
        # noinspection PyArgumentList
        self.qgsProj = QgsProject.instance()                  # This field holds a QGIS project instance to save results

    # Calling the project creates a new project file.
    def __call__(self, name, path):
        # Name and path will be determined outside the class.
        self.name = name
        self.path = path + f"/{validFilename(name)}"
        self.fileName = f"{validFilename(self.name)}.json"
        self.created = time.time()
        self.saved = self.created
        # Create the folders for the project based on the path
        folders = ['json',
                   'las_files/tiles',
                   'metadata',
                   'rasters/25_cm/tiles',
                   'rasters/50_cm/tiles',
                   'rasters/100_cm/tiles',
                   'shapefiles/25_cm',
                   'shapefiles/50_cm',
                   'shapefiles/100_cm']
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # Set the working directory to the project path for future read/write functionality
        os.chdir(self.path)

        for folder in folders:
            sub_folder = f'{self.path}/{folder}'
            if not os.path.exists(sub_folder):
                os.makedirs(sub_folder)

        self.save()

    # -----------------------------------------------------------------------------------------------------------------
    # This method loads a project file
    def load(self, saveFile):
        # Try to open the save file and load the project data
        # noinspection PyBroadException
        try:
            with open(saveFile) as f:
                data = json.load(f)

            self.name = data['name']
            self.path = data['path']
            os.chdir(self.path)
            self.fileName = data['fileName']
            self.created = data['created']
            self.saved = data['saved']
            self.state = data['state']
            self.classified = data['classified']
            self.AOI.load(data['AOI'])
            self.waterFeatures.load(data['water'])
            self.inputFiles.load(data['input_files'])
            self.tileFiles.load(data['tile_files'])
            self.mergedFiles.load(data['merged_files'])
            for reso in data['dem_tiles']:
                self.DEMTiles[reso].load(data['dem_tiles'][reso])
                self.DEM[reso].load(data['DEM'][reso])
                self.depressions[reso].load(data['depressions'][reso])
            self.qgsProj.read(f"{self.fileName[:-4]}qgz")
            self.qgsProj.pathResolver()

        # If the process fails, re-initialize the file and return false
        except:
            Gui.popup_ok('Invalid Project File',
                         title='Error')
            self.__init__()
            return False

        # Return true if there are no errors
        else:
            return True

    # -----------------------------------------------------------------------------------------------------------------
    # This method saves the project file
    def save(self):
        saveDict = {'name': self.name,
                    'fileName': self.fileName,
                    'path': self.path,
                    'created': self.created,
                    'saved': time.time(),
                    'state': self.state,
                    'classified': self.classified,
                    'AOI': self.AOI.save(),
                    'water': self.waterFeatures.save(),
                    'input_files': self.inputFiles.export(),
                    'tile_files': self.tileFiles.export(),
                    'merged_files': self.mergedFiles.export(),
                    'dem_tiles': {'100_cm': self.DEMTiles['100_cm'].export(),
                                  '50_cm': self.DEMTiles['50_cm'].export(),
                                  '25_cm': self.DEMTiles['25_cm'].export()},
                    'DEM': {'100_cm': self.DEM['100_cm'].save(),
                            '50_cm': self.DEM['50_cm'].save(),
                            '25_cm': self.DEM['25_cm'].save()},
                    'depressions': {'100_cm': self.depressions['100_cm'].save(),
                                    '50_cm': self.depressions['50_cm'].save(),
                                    '25_cm': self.depressions['25_cm'].save()}}

        with open(self.fileName, 'w') as projFile:
            json.dump(saveDict, projFile, indent=4)

        self.qgsProj.write(f"{self.fileName[:-4]}qgz")

    # -----------------------------------------------------------------------------------------------------------------
    # This method indicates which PointCloudCollections contain files
    def pcLists(self) -> list:
        pcLists = []
        if len(self.inputFiles.clouds) > 0:
            pcLists.append('Input Files')
        if len(self.tileFiles.clouds) > 0:
            pcLists.append('Point Cloud Tiles')
        if len(self.mergedFiles.clouds) > 0:
            pcLists.append('Merged Point Clouds')
        return pcLists

    # -----------------------------------------------------------------------------------------------------------------
    # This method returns a list of the point cloud names for a specific collection
    def pcNames(self, pcListName) -> list:
        if pcListName == 'Input Files':
            outList = self.inputFiles.fList()
        elif pcListName == 'Point Cloud Tiles':
            outList = self.tileFiles.fList()
        elif pcListName == 'Merged Point Clouds':
            outList = self.mergedFiles.fList()
        else:
            outList = []

        return outList


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This class defines a Raster Object, and stores it in a QGIS format
class Raster(object):
    # Raster object constructor
    def __init__(self):
        # Create the QGIS vector layer
        self.path = ''
        self.name = ''
        self.time = 0.0
        self.qObj = None
        self.range = (0.0, 0.0)

    # Calling the object assigns its values
    def __call__(self, path, name):
        self.path = path
        self.name = name
        self.qObj = QgsRasterLayer(self.path, self.name)

    # This method loads a raster from a dictionary from a save file
    def load(self, d: dict):
        if d['path'] != '':
            self.__call__(d['path'], d['name'])
            self.time = d['time']

    # This method generates a dictionary to save data to a save file
    def save(self) -> dict:
        save = {'path': self.path,
                'name': self.name,
                'time': self.time}
        return save

    # This method fills no data spots in a raster by apply the GDAL No Fill process to it
    def fillVoids(self):
        # Fill blank spaces in the raster by calculating the inter-distance weighted value from nearest neighbours
        # Create a dictionary to hold the algorithm parameters
        alg_params = {'INPUT': self.qObj,
                      'BAND': 1,
                      'DISTANCE': 20,
                      'OUTPUT': f'{self.path[:-4]}_no_voids.tif'}
        # Run the fill algorithm
        processing.run('gdal:fillnodata', alg_params)

        self.__call__(f'{self.path[:-4]}_no_voids.tif', self.name)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This class is a collection of point clouds, along with the functions applied to such a collection
class RasterCollection(object):
    # PointCloudCollection constructor
    def __init__(self):
        self.tifs = []

    # Calling the collection with an index returns the individual point cloud
    def __call__(self, index) -> Raster:
        return self.tifs[index]

    # -----------------------------------------------------------------------------------------------------------------
    # This method adds PointCloud objects to a collection. Can be a single object or list of objects
    def add(self, addition):
        if type(addition) == list:
            for tif in addition:
                self.tifs.append(tif)
        else:
            self.tifs.append(addition)

    # This method exports a PointCloudCollection by converting individual PointCloud objects to a list of dictionaries
    def export(self) -> list:
        exportList = []
        for tif in self.tifs:
            exportList.append(tif.save())
        return exportList

    # This method loads a list of dictionaries to fill a PointCloudCollection with PointCloud objects
    def load(self, tifList):
        for tif in tifList:
            x = Raster()
            x.load(tif)
            self.tifs.append(x)

    # This method merges a list of rasters into a single raster
    def merge(self, outFile, name) -> Raster:
        # Keep track of how long the function takes
        startTime = time.time()

        # Create a dictionary with the argument list for the GDAL merge function. NODATA_OUTPUT ensures that nulls are
        # preserved and not set to 0
        arg_list = {'INPUT': [tif.qObj for tif in self.tifs],
                    'NODATA_OUTPUT': -999999,
                    'DATA_TYPE': 5,
                    'OUTPUT': outFile}

        # Run the merge function
        processing.run('gdal:merge', arg_list)
        raster = Raster()
        raster(outFile, name)
        raster.time = time.time() - startTime

        return raster


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This class corresponds to a basic point cloud, and all the functions that can be applied to one.
class PointCloud(object):
    # PointCloud object constructor
    def __init__(self):
        self.path = ''
        self.name = ''
        self.fileSize = 0
        self.loadTime = 0.0
        self.splitTime = 0.0
        self.classTime = 0.0
        self.mergeTime = 0.0
        self.DEMTime = 0.0
        self.minx = 0.0
        self.maxx = 0.0
        self.miny = 0.0
        self.maxy = 0.0
        self.count = 0
        self.crs = ''
        self.polygon = ''
        self.classification = [0] * 32

    # Calling a PointCloud with a path measures it
    def __call__(self, path: str):
        self.path = path
        self.name = path.split('/')[-1][:-4]
        self.fileSize = os.path.getsize(path)

    # This method assigns point cloud values from dictionary values
    def open(self, d: dict):
        self.__dict__.update(d)

    # This method scans the Point Cloud, and stores relevant data.
    def scan(self):
        # Keep track of how long it takes to load and analyze the file
        startTime = time.time()

        # Create the pipeline dictionary that calculates LiDAR point cloud stats
        pl = {"pipeline": [self.path,
                           {"type": "filters.hexbin"},
                           {"type": "filters.stats",
                            "dimensions": "X, Y, Classification",
                            "count": "Classification"}]}

        # Save the pipeline to a json file in the json folder
        with open(f"json/stats_{self.name}.json", 'w') as f:
            json.dump(pl, f, indent=4)

        # Construct the PDAL pipeline, and run it
        pipeline = pdal.Pipeline(json.dumps(pl))
        pipeline.execute_streaming()

        # Save the results to the object and export the metadata to a file
        self.loadTime = time.time() - startTime

        md = pipeline.metadata['metadata']
        self.loadMetadata(md)

        # Full metadata record is saved to file
        with open(f"metadata/{self.name}.json", 'w') as f:
            json.dump(md, f, indent=4)

        # Set the pipeline to nothing to release resources
        pipeline = None

    # This function creates the point cloud from existing metadata
    def loadMetadata(self, md: dict):
        self.minx = md['filters.stats']['statistic'][0]['minimum']
        self.maxx = md['filters.stats']['statistic'][0]['maximum']
        self.miny = md['filters.stats']['statistic'][1]['minimum']
        self.maxy = md['filters.stats']['statistic'][1]['maximum']
        self.count = md['filters.stats']['statistic'][0]['count']
        self.polygon = md['filters.hexbin']['boundary']

        # When loading metadata from a file that has been merged, the 'readers.las' dictionary may be different
        if isinstance(md['readers.las'], list):
            self.crs = md['readers.las'][0]['srs']['compoundwkt']
        else:
            self.crs = md['readers.las']['srs']['compoundwkt']

        # Store classification data from the file
        for classification in md['filters.stats']['statistic'][2]['counts']:
            c, n = classification.split('/')
            self.classification[int(float(c))] = int(float(n))

    # Create a function that splits a point cloud into tiles
    def splitCloud(self, origin: list[float], length: float) -> list:
        # Keep track of how long it takes to load and analyze the file
        startTime = time.time()

        # Create the pipeline dictionary that calculates LiDAR point cloud stats
        pl = {"pipeline": [self.path,
                           {"type": "filters.splitter",
                            "length": f'{length}',
                            "origin_x": f'{origin[0] + length / 10}',
                            "origin_y": f'{origin[1] + length / 10}',
                            "buffer": f'{length / 10}'},
                           {"type": "writers.las",
                            "filename": f"las_files/tiles/{self.name}_#.las",
                            "forward": "all"}]}

        # Save the pipeline to a json file in the json folder
        with open(f"json/split_{self.name}.json", 'w') as f:
            json.dump(pl, f, indent=4)

        pipeline = pdal.Pipeline(json.dumps(pl))
        pipeline.execute()

        self.splitTime = time.time() - startTime

        splitList = pipeline.metadata['metadata']['writers.las']['filename']

        # Return the list of file names
        return splitList

    # -----------------------------------------------------------------------------------------------------------------
    # This function takes classifies the points in a PointCloud object as ground/not ground
    def classify(self):
        # Keep track of how long it takes to load and analyze the file
        startTime = time.time()

        # Create the pipeline dictionary that classifies the LiDAR files.
        pl = {'pipeline': [self.path,
                           {"type": "filters.csf",
                            "resolution": 0.1,
                            "threshold": 0.1},
                           {"type": "filters.hexbin"},
                           {"type": "filters.stats",
                            "dimensions": "Classification",
                            "count": "Classification"},
                           self.path]}

        # Save the pipeline to a json file in the json folder
        with open(f"json/class_{self.name}.json", 'w') as f:
            json.dump(pl, f, indent=4)

        # Construct the PDAL pipeline, and run it
        pipeline = pdal.Pipeline(json.dumps(pl))
        pipeline.execute()

        # Update the data for the point cloud
        self.classTime = time.time() - startTime
        self.classification = [0] * 32                             # Classification list must be returned to zero state
        for classification in pipeline.metadata['metadata']['filters.stats']['statistic'][0]['counts']:
            c, n = classification.split('/')
            self.classification[int(float(c))] = int(float(n))

    # -----------------------------------------------------------------------------------------------------------------
    # This function generates a DEM from a PointCloud object, and return it as a QGIS raster object
    def createDEM(self, outPath: str, name: str, res: float) -> Raster:
        # Keep track of how long it takes to generate the DEM
        startTime = time.time()

        # Create the pipeline dictionary that generates a binned 1m DEM
        pl = {'pipeline': [self.path,
                           {"type": "filters.range",
                            "limits": "Classification[2:2]"},
                           {"type": "writers.gdal",
                            "filename": f"{outPath}/{name}.tif",
                            "gdaldriver": "GTiff",
                            "resolution": res,
                            "output_type": "mean"}]}

        # Save the pipeline to a json file in the json folder
        with open(f"json/dem_{name}.json", 'w') as f:
            json.dump(pl, f, indent=4)

        # Construct the PDAL pipeline, and run it
        pipeline = pdal.Pipeline(json.dumps(pl))
        pipeline.execute()

        self.DEMTime = time.time() - startTime

        raster = Raster()
        raster(f"{outPath}/{name}.tif", name)
        raster.time = self.DEMTime
        return raster

    # -----------------------------------------------------------------------------------------------------------------
    # These methods determine the range of a particular dimension of the point cloud bounding box
    def rangeX(self) -> float:
        return self.maxx - self.minx

    def rangeY(self) -> float:
        return self.maxy - self.miny

    # This method returns the values for the stats pane in the main window
    def displayStats(self, pcType) -> dict:
        stats = {'_STAT_X_RANGE_': f"{self.minx:,.3f} to {self.maxx:,.3f}",
                 '_STAT_Y_RANGE_': f"{self.miny:,.3f} to {self.maxy:,.3f}",
                 '_STAT_NO_POINTS_': f"{self.count:,}",
                 '_STAT_FILE_SIZE_': f"{self.fileSize:,} bytes"}

        if pcType == 'Input Files':
            stats['_STAT_TIME_VAL_'] = f"{secFmt(self.loadTime)}"
        elif pcType == 'Point Cloud Tiles':
            stats['_STAT_TIME_VAL_'] = f"{secFmt(self.splitTime)}"
        elif pcType == 'Merged Point Clouds':
            stats['_STAT_TIME_VAL_'] = f"{secFmt(self.mergeTime)}"

        return stats


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This class is a collection of point clouds, along with the functions applied to such a collection
class PointCloudCollection(object):
    # PointCloudCollection constructor
    def __init__(self):
        self.clouds = []

    # Calling the collection with an index returns the individual point cloud
    def __call__(self, index) -> PointCloud:
        return self.clouds[index]

    # -----------------------------------------------------------------------------------------------------------------
    # This method adds PointCloud objects to a collection. Can be a single object or list of objects
    def add(self, addition):
        if type(addition) == list:
            for pc in addition:
                self.clouds.append(pc)
        else:
            self.clouds.append(addition)

    # This method exports a PointCloudCollection by converting individual PointCloud objects to a list of dictionaries
    def export(self) -> list:
        exportList = []
        for pc in self.clouds:
            exportList.append(pc.__dict__)
        return exportList

    # This method loads a list of dictionaries to fill a PointCloudCollection with PointCloud objects
    def load(self, pcList):
        for pc in pcList:
            x = PointCloud()
            x.open(pc)
            self.clouds.append(x)

    # -----------------------------------------------------------------------------------------------------------------
    # This method returns a list of the names of the files in the collection
    def fList(self) -> list:
        fList = []
        for pc in self.clouds:
            fList.append(pc.name)
        return fList

    # -----------------------------------------------------------------------------------------------------------------
    # This method returns the saved stats data for a specific point cloud in the collection
    def stats(self, index: int, pcType: str) -> dict:
        x = self.clouds[index].displayStats(pcType)
        return x

    # -----------------------------------------------------------------------------------------------------------------
    # This method returns the total number of points contained in this collection
    def noPoints(self) -> int:
        noPoints = 0
        for pc in self.clouds:
            noPoints += pc.count
        return noPoints

    # -----------------------------------------------------------------------------------------------------------------
    # This method returns the bounding box around the whole point collection
    def boundingBox(self) -> list:
        # Start by setting the first point cloud's bounding box as the collection bounding box
        minx = self.clouds[0].minx
        miny = self.clouds[0].miny
        maxx = self.clouds[0].maxx
        maxy = self.clouds[0].maxy

        # Compare each point cloud's boundaries with the collection boundaries
        for i in range(1, len(self.clouds)):
            minx = min(minx, self.clouds[i].minx)
            miny = min(miny, self.clouds[i].miny)
            maxx = max(maxx, self.clouds[i].maxx)
            maxy = max(maxy, self.clouds[i].maxy)

        return [minx, miny, maxx, maxy]

    # -----------------------------------------------------------------------------------------------------------------
    # This method generates a shapefile containing the LiDAR boundaries for all files in the collection
    # Process modified from GDAL/OGR Cookbook
    def boundaryShapefile(self, fileName, layerName) -> QgsVectorLayer:
        # Create the output for the shapefile if required
        if not os.path.exists(f"shapefiles/{fileName}"):
            os.makedirs(f"shapefiles/{fileName}")
        shpFile = shpDriver.CreateDataSource(f"shapefiles/{fileName}/{fileName}.shp")

        # Create the spatial reference for the file from Well Known Text
        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.clouds[0].crs)

        # Create the layer, and add the appropriate fields
        layer = shpFile.CreateLayer('Input Boundaries', srs, ogr.wkbPolygon)
        # noinspection PyArgumentList
        layer.CreateField(ogr.FieldDefn('Name', ogr.OFTString))
        layer.CreateField(ogr.FieldDefn('no_Points', ogr.OFTInteger64))

        # Iterate through the shapes, and add them to the layer
        for pc in self.clouds:
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField('Name', pc.name)
            feature.SetField('no_Points', pc.count)
            wkt = pc.polygon
            poly = ogr.CreateGeometryFromWkt(wkt)
            feature.SetGeometry(poly)
            layer.CreateFeature(feature)
            feature = None

        # Save and close the shapefile, then return a QGIS vector layer
        shpFile = None
        return QgsVectorLayer(f"shapefiles/{fileName}/{fileName}.shp", layerName)

    # -----------------------------------------------------------------------------------------------------------------
    # This method merges the point clouds in the collection
    def merge(self, outFile) -> PointCloud:
        # Keep track of how long it takes to load and analyze the file
        startTime = time.time()

        # Create the pipeline dictionary that merges the LiDAR files.
        pl = {"pipeline": [pc.path for pc in self.clouds]}
        pl['pipeline'].append({"type": "filters.merge"})
        pl['pipeline'].append({"type": "filters.hexbin"})
        pl['pipeline'].append({"type": "filters.stats",
                               "dimensions": "X, Y, Classification",
                               "count": "Classification"})
        pl['pipeline'].append({"type": "writers.las",
                               "filename": f"las_files/{outFile}",
                               "forward": "all",
                               "offset_x": "auto",
                               "offset_y": "auto",
                               "offset_z": "auto"})

        # Save the pipeline to a json file in the json folder
        outName = outFile.split('/')[-1]
        with open(f"json/merge_{outName[:-4]}.json", 'w') as f:
            json.dump(pl, f, indent=4)

        # Construct the PDAL pipeline, and run it
        pipeline = pdal.Pipeline(json.dumps(pl))
        pipeline.execute_streaming()

        # Generate a PointCloud object from the pipeline metadata
        pc = PointCloud()
        pc(f"las_files/{outFile}")

        pc.loadMetadata(pipeline.metadata['metadata'])
        pc.mergeTime = time.time() - startTime

        return pc


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This class represents a shapefile and stores it in an OGR format, and in a QGIS format if requested:
class ShapeFile(object):
    # ShapeFile object constructor
    def __init__(self):
        # Create the QGIS vector layer
        self.path = ''
        self.name = ''
        self.oObj = None
        self.qObj = None
        self.count = 0

    # Calling the ShapeFile object assigns values
    def __call__(self, path, name):
        self.path = path
        self.name = name
        self.oObj = shpDriver.Open(self.path, 0)
        self.count = self.oObj.GetLayer().GetFeatureCount()

    # This method creates a QGIS Vector Layer object from the ogr file
    def createQgsLayer(self):
        self.qObj = QgsVectorLayer(self.path, self.name)

    # This method exports a ShapeFile object for saving
    def save(self) -> dict:
        out = {'path': self.path,
               'name': self.name,
               'count': self.count,
               'inQGIS': self.qObj is not None}
        return out

    # This method loads a ShapeFile object from save data
    def load(self, data):
        self.path = data['path']
        self.name = data['name']
        self.count = data['count']
        self.oObj = shpDriver.Open(self.path, 0)
        if data['inQGIS']:
            self.qObj = QgsVectorLayer(self.path, self.name)

    # This method generates a list of field names
    # Taken from the GDAL/OGR cookbook
    def fields(self) -> list:
        fields = []
        defn = self.oObj.GetLayer().GetLayerDefn()

        for i in range(defn.GetFieldCount()):
            fields.append(defn.GetFieldDefn(i).GetName())

        return fields

    # This method generates a list of unique values for any set of fields.
    def uniqueValues(self, fieldList) -> list:
        values = []
        # Reset the reading on the OGR object to ensure you are on the first record
        lay = self.oObj.GetLayer()
        lay.ResetReading()
        # Read through each record, and store the value to an array
        feat = lay.GetNextFeature()
        while feat is not None:
            x = feat.ExportToJson(as_object=True)['properties']
            values.append(tuple([x[param] for param in fieldList]))
            feat = lay.GetNextFeature()

        uniqueValues = list(set(values))
        return [list(uV) for uV in uniqueValues]

    # This method generates a shapefile's bounding box
    def boundingBox(self) -> list:
        # Reset the reading on the OGR object to ensure you are on the first record
        lay = self.oObj.GetLayer()
        lay.ResetReading()
        # Read the first record, and create an initial bounding box from it
        feat = lay.GetNextFeature()
        bbox = list(feat.GetGeometryRef().GetEnvelope())
        feat = lay.GetNextFeature()
        while feat is not None:
            featBox = feat.GetGeometryRef().GetEnvelope()
            for i in range(2):
                bbox[i] = min(bbox[i], featBox[i])
                bbox[i + 1] = max(bbox[i + 1], feat[i + 1])
            feat = lay.getNextFeature()

        return bbox

    # This method generates a filtered copy of the shapefile
    def filteredCopy(self, fileName, name, query):
        # Filter the shapefile according to the file
        self.oObj.GetLayer().SetAttributeFilter(query)

        # Export it to a new file
        if not os.path.exists(f'shapefiles/{fileName}'):
            os.makedirs(f'shapefiles/{fileName}')
        newFeat = shpDriver.CreateDataSource(f'shapefiles/{fileName}/{fileName}.shp')
        newLay = newFeat.CopyLayer(self.oObj.GetLayer(), name)
        del newFeat, newLay


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This class defines the GUI, and the main event loop
class DetectorGUI(object):
    # DetectorGUI object constructor
    def __init__(self):
        # Create a new project
        self.proj = Project()

        # Create the process state list
        self.stateList = ['__ADD_AOI__',
                          '__ADD_PCS__',
                          '__ADD_WATER__',
                          '__MERGE/SPLIT__',
                          '__CLASSIFY__',
                          '__GEN_DEM__',
                          '__FIND_SINK__',
                          '__RESULTS__']

        # Create the layout for the window's menu bar
        menu_def = [['File', ['New Project', 'Open...', 'About', '---', 'Exit']]]

        # Create the layout for the LiDAR file selection frame
        selCol = [[Gui.Combo(self.proj.pcLists(),
                             default_value='',
                             size=20,
                             enable_events=True,
                             readonly=True,
                             key='_LIST_SELECTOR_')],
                  [Gui.Listbox(self.proj.inputFiles.fList(),
                               size=(30, 10),
                               enable_events=True,
                               key='_SEL_FILE_')]]

        # Set the layout for the LiDAR statistics frame
        statCol = [[Gui.Text('File Statistics',
                             size=26,
                             justification='center')],
                   [Gui.Text('Process Time:',
                             size=13,
                             justification='right'),
                    Gui.Text('',
                             size=30,
                             justification='left',
                             key='_STAT_TIME_VAL_')],
                   [Gui.Text('X Range:',
                             size=13,
                             justification='right'),
                    Gui.Text('',
                             size=30,
                             justification='left',
                             key='_STAT_X_RANGE_')],
                   [Gui.Text('Y Range:',
                             size=13,
                             justification='right'),
                    Gui.Text('',
                             size=30,
                             justification='left',
                             key='_STAT_Y_RANGE_')],
                   [Gui.Text('# Points:',
                             size=13,
                             justification='right'),
                    Gui.Text('',
                             size=30,
                             justification='left',
                             key='_STAT_NO_POINTS_')],
                   [Gui.Text('File Size:',
                             size=13,
                             justification='right'),
                    Gui.Text('',
                             size=30,
                             justification='left',
                             key='_STAT_FILE_SIZE_')]]

        optCol = [[Gui.Button(button_text='Add Area of Interest',
                              key='__ADD_AOI__',
                              enable_events=True,
                              disabled=True,
                              tooltip='Add the area of interest to the project')],
                  [Gui.Button(button_text='Add Point Clouds',
                              key='__ADD_PCS__',
                              enable_events=True,
                              disabled=True,
                              tooltip='Add point clouds to the project')],
                  [Gui.Button(button_text='Add Water Features',
                              key='__ADD_WATER__',
                              enable_events=True,
                              disabled=True,
                              tooltip='Add masking water features to the project')],
                  [Gui.Button(button_text='Merge/Split Point Clouds',
                              key='__MERGE/SPLIT__',
                              enable_events=True,
                              disabled=True,
                              tooltip='Merge of split the files to process them')],
                  [Gui.Button(button_text='Classify Point Clouds',
                              key='__CLASSIFY__',
                              enable_events=True,
                              disabled=True,
                              tooltip='Classify ground points')],
                  [Gui.Button(button_text='Generate DEM',
                              key='__GEN_DEM__',
                              enable_events=True,
                              disabled=True,
                              tooltip='Generate a DEM for all loaded points')],
                  [Gui.Button(button_text='Find Sinkholes',
                              key='__FIND_SINK__',
                              enable_events=True,
                              disabled=True,
                              tooltip='Find sinkholes from the DEM')],
                  [Gui.Button(button_text='View Results',
                              key='__RESULTS__',
                              enable_events=True,
                              disabled=True,
                              tooltip='View the results of the sinkhole analysis')]
                  ]

        # Define the primary window for the GUI
        self.win = Gui.Window('COLDS',
                              [[Gui.Menu(menu_def,
                                         key='_MENU_')],
                               [Gui.Column(selCol,
                                           element_justification='center'),
                                Gui.Column(statCol),
                                Gui.Column(optCol,
                                           element_justification='left')],
                               [Gui.Column(
                                   [[Gui.Text('Messages')],
                                    [Gui.Multiline(size=(105, 10),
                                                   autoscroll=True,
                                                   key='_OUT_PRINT_' + Gui.WRITE_ONLY_KEY)]],
                                   element_justification='center')]])

    # -----------------------------------------------------------------------------------------------------------------
    # Calling the class will run the window's event loop
    def __call__(self):
        while True:
            evt, self.val = self.win.read()

            # Exit the loop if the main window is closed, or exit is selected
            if evt == Gui.WINDOW_CLOSED or evt == 'Exit':
                # When the program is shut down properly, save the project if one has been opened
                if self.proj.path != '':
                    self.proj.save()
                # Close the Window
                self.win.close()
                # Close the QGIS instance
                qgs.exitQgis()
                # Exit the python program
                exit()

            # Show details about the program if about is selected
            elif evt == 'About':
                Gui.popup_ok('Cusson Open-source LiDAR Depression Scanner (COLDS)',
                             'Created by Keith Cusson',
                             f'Version {self.proj.version}',
                             'MIT License (c) 2023',
                             title='LiDAR Basic Sinkhole Detector')

            elif evt == 'New Project':
                self.newProject()

            elif evt == 'Open...':
                self.openProject()

            elif evt == '__ADD_AOI__':
                self.addAOI()

            elif evt == '__ADD_PCS__':
                self.loadInputs()
                self.next()

            elif evt == '_LIST_SELECTOR_':
                self.changeCombo()

            elif evt == '_SEL_FILE_':
                self.updateStats()

            elif evt == '__ADD_WATER__':
                self.addWater()

            elif evt == '__MERGE/SPLIT__':
                self.mergeSplit()

            elif evt == '__CLASSIFY__':
                self.classify()

            elif evt == '__GEN_DEM__':
                self.gen_DEM()

            elif evt == '__FIND_SINK__':
                self.findSink()

                self.results()

            elif evt == '__RESULTS__':
                self.results()

    # -----------------------------------------------------------------------------------------------------------------
    # This method sends a message to the main window message box
    def print(self, msg):
        self.win['_OUT_PRINT_' + Gui.WRITE_ONLY_KEY].print(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}')
        self.win.refresh()

    # -----------------------------------------------------------------------------------------------------------------
    # This method generates a GUI window that allows the user to input a project name and pick a save location.
    def newProject(self):
        # Create the layout for the window
        layout = [[Gui.Text('Enter a project name:')],
                  [Gui.Input(default_text='default',
                             key='Project_Name')],
                  [Gui.Text('Where would you like to save your files?')],
                  [Gui.Text(f'No Folder Selected'),
                   Gui.FolderBrowse(initial_folder=os.getcwd(),
                                    enable_events=True,
                                    key='Project_Folder')],
                  [Gui.Ok()]]

        # Create the window element
        pWin = Gui.Window('Create a new project',
                          layout,
                          modal=True)

        # Create the event loop for the window element
        while True:
            e, v = pWin.read()

            # Exit the loop if the main window is closed, or exit is selected
            if e == Gui.WINDOW_CLOSED:
                break

            # Change the displayed folder if a new folder is selected
            elif e == 'Project_Folder':
                pWin['Project_Folder'].Update(value=v['Project_Folder'])

            # Exit the loop if OK is selected
            elif e == 'Ok':
                self.proj(v['Project_Name'], v['Project_Folder'])
                self.win['_MENU_'].update(menu_definition=[['File',
                                                            ['!New Project',
                                                             '!Open...',
                                                             'About',
                                                             '---',
                                                             'Exit']]])
                self.win['__ADD_AOI__'].update(disabled=False)
                break

        pWin.close()

    # -----------------------------------------------------------------------------------------------------------------
    # This method generates a GUI window that allows the user to select an existing project
    def openProject(self):
        projFile = Gui.popup_get_file('Please select an existing project file',
                                      title='Open Project File',
                                      file_types=(('JSON files', '*.json'),
                                                  ('All files', '*.* *')))

        if self.proj.load(projFile):
            self.win['_MENU_'].Update(menu_definition=[['File',
                                                        ['!New Project',
                                                         '!Open...',
                                                         'About',
                                                         '---',
                                                         'Exit']]])

            # Update the selector pane with any files that have been loaded
            index = None
            if len(self.proj.pcLists()) > 0:
                index = 0
            self.win['_LIST_SELECTOR_'].Update(values=self.proj.pcLists(),
                                               set_to_index=index)
            self.win.refresh()
            index = self.win['_LIST_SELECTOR_'].get()
            self.win['_SEL_FILE_'].Update(values=self.proj.pcNames(index))
            for i in range(0, len(self.stateList)):
                self.win[self.stateList[i]].Update(disabled=i != self.proj.state)

    # -----------------------------------------------------------------------------------------------------------------
    # This method determines what the next available process should be
    def next(self):
        self.proj.state += 1
        # If the point cloud has been classified, skip highlighting that button
        if self.stateList[self.proj.state] == '__CLASSIFY__' and self.proj.classified is True:
            self.proj.state += 1
        for i in range(0, len(self.stateList)):
            # If a process is the next step, enable the button, otherwise enable it
            self.win[self.stateList[i]].Update(disabled=i != self.proj.state)

        self.proj.save()

    # -----------------------------------------------------------------------------------------------------------------
    # This method changes the file selector list and clears the stats pane when the combo box is changed
    def changeCombo(self):
        pcType = self.val['_LIST_SELECTOR_']
        self.win['_SEL_FILE_'].Update(values=self.proj.pcNames(pcType))
        self.win['_STAT_TIME_VAL_'].Update('')
        self.win['_STAT_X_RANGE_'].Update('')
        self.win['_STAT_Y_RANGE_'].Update('')
        self.win['_STAT_NO_POINTS_'].Update('')
        self.win['_STAT_FILE_SIZE_'].Update('')

    # -----------------------------------------------------------------------------------------------------------------
    # This method updates the stats pane when a new file is selected
    def updateStats(self):
        pcType = self.val['_LIST_SELECTOR_']
        index = self.win['_SEL_FILE_'].get_indexes()[0]
        if pcType == 'Input Files':
            stats = self.proj.inputFiles.stats(index, pcType)
        elif pcType == 'Point Cloud Tiles':
            stats = self.proj.tileFiles.stats(index, pcType)
        elif pcType == 'Merged Point Clouds':
            stats = self.proj.mergedFiles.stats(index, pcType)

        # Update the stats with the values from the dictionary
        for stat in stats:
            self.win[stat].Update(stats[stat])

    # -----------------------------------------------------------------------------------------------------------------
    def addAOI(self):
        self.win['__ADD_AOI__'].Update(disabled=True)
        # Ask the user to add a file
        aoiPath = Gui.popup_get_file('Please select a shapefile that outlines your area of interest.',
                                     title='Select AOI shapefile',
                                     file_types=(('ESRI Shape Files', '*.shp'),
                                                 ('All Files', '*.* *')))

        # Only perform an operation if a valid file is selected
        if os.path.exists(aoiPath):
            self.proj.AOI(aoiPath, 'AOI')
            self.proj.AOI.createQgsLayer()
            self.next()

    # -----------------------------------------------------------------------------------------------------------------
    # This method generates a GUI window that allows the user to select input LiDAR point clouds
    def loadInputs(self):
        self.win['__ADD_PCS__'].Update(disabled=True)
        # Create a variable that stores the selected point clouds, as a list of paths and a list of names
        toLoad = [[], []]

        # Create the layout for the window
        layout = [[Gui.Text('Selected Lidar Tiles',
                            justification='center')],
                  [Gui.Listbox(toLoad[1],
                               size=(30, 10),
                               key='__TO_LOAD__')],
                  [Gui.Input(key='__ADD_FILE__',
                             size=(1, 1),
                             enable_events=True,
                             visible=False),
                   Gui.FileBrowse(button_text='Add Point Cloud',
                                  target='__ADD_FILE__',
                                  key='ADD',
                                  file_types=(('LIDAR Files', '*.las *.laz'),
                                              ('LASer File Format', '*.las'),
                                              ('LASer Zipped File Format', '*.laz'))),
                   Gui.Button('Remove',
                              key='__REMOVE__',
                              disabled=True),
                   Gui.ProgressBar(1,
                                   orientation='horizontal',
                                   size=(20, 15),
                                   key='__PROGRESS__',
                                   relief='RELIEF_SUNKEN',
                                   visible=False)],
                  [Gui.Text('0/0',
                            justification='center',
                            visible=False,
                            size=(28, 1),
                            key='__PROG_TEXT__')],
                  [Gui.HorizontalSeparator()],
                  [Gui.Ok(),
                   Gui.Cancel()]]

        # Create the new window from the layout
        addWin = Gui.Window('Add LiDAR Files',
                            layout,
                            modal=True)

        # Run the event loop
        while True:
            e, v = addWin.read()

            # Exit the loop and do nothing if the main window is closed, or cancel is selected
            if e == Gui.WINDOW_CLOSED or e == 'Cancel':
                break

            # Add a file if the button is pressed
            elif e == '__ADD_FILE__':
                # If the file already exists in the list, raise an error
                if v['__ADD_FILE__'] in toLoad[0]:
                    Gui.popup_ok('File already added to list.',
                                 title='Error - Duplicate File')
                else:
                    toLoad[0].append(v['__ADD_FILE__'])
                    toLoad[1].append(v['__ADD_FILE__'].split('/')[-1])
                    addWin['__REMOVE__'].Update(disabled=len(toLoad[0]) == 0)
                    addWin['__TO_LOAD__'].Update(values=toLoad[1])

            # Remove a file if the button is pressed and a file is selected
            elif e == '__REMOVE__' and len(v['__TO_LOAD__']) == 1:
                i = addWin['__TO_LOAD__'].get_indexes()[0]
                toLoad[0].pop(i)
                toLoad[1].pop(i)
                addWin['__REMOVE__'].Update(disabled=len(toLoad[0]) == 0)
                addWin['__TO_LOAD__'].Update(values=toLoad[1])

            # Add the files once the user selects OK.
            elif e == 'Ok' and len(toLoad[0]) > 0:
                # Confirm that the user wants to add the selected files, and notify them the process can be lengthy for
                # some point clouds
                if 'Yes' == Gui.popup_yes_no(f'Would you like to add the following {len(toLoad[0])} file(s)? You will '
                                             f'not have the opportunity to add more later. Each file may take several '
                                             f'minutes to load depending on your hardware.'):
                    # Change the interface to display a progress bar and disable all buttons
                    addWin['ADD'].Update(visible=False)
                    addWin['__REMOVE__'].Update(visible=False)
                    addWin['__PROGRESS__'].Update(current_count=0, max=len(toLoad[0]), visible=True)
                    addWin['__PROG_TEXT__'].Update(value=f'0/{len(toLoad[0])}', visible=True)
                    addWin['Ok'].Update(disabled=True)
                    addWin['Cancel'].Update(disabled=True)
                    addWin.refresh()

                    # Load each file in turn, and update the message window and progress bar, checking to see if the
                    # point clouds are already classified
                    self.win['_OUT_PRINT_' + Gui.WRITE_ONLY_KEY].print('=' * 60)
                    self.print(f'Loading {len(toLoad[0])} file(s). Please wait...')
                    classified = True
                    for i in range(0, len(toLoad[0])):
                        x = PointCloud()
                        x(toLoad[0][i])
                        x.scan()
                        classified = classified and x.classification[2] > 0
                        self.proj.inputFiles.add(x)
                        self.print(f'Point cloud {toLoad[1][i]} loaded in {secFmt(self.proj.inputFiles(i).loadTime)}')
                        addWin['__PROGRESS__'].Update(current_count=i+1)
                        addWin['__PROG_TEXT__'].Update(value=f'{i+1}/{len(toLoad[0])}')
                        addWin.refresh()
                        self.proj.save()

                    # If the all the point clouds are classified, ask the user if they would like to reclassify them
                    if classified is True:
                        reclass = Gui.popup_yes_no('These point clouds appear to have already been classified. Would '
                                                   'you like to re-classify ground points prior to creating a DEM?',
                                                   title='Classified Point Clouds')
                        if reclass == 'No':
                            self.proj.classified = True

                    # Exit the loop, and let the full progress bar linger for a second.
                    time.sleep(1)
                    break

        # Update the combobox and list box in the main window, and close the dialog box
        self.win['_LIST_SELECTOR_'].Update(values=self.proj.pcLists(),
                                           set_to_index=0)
        self.win['_SEL_FILE_'].Update(values=self.proj.pcNames('Input Files'))
        addWin.close()

        # Compare the Coordinate Reference Systems of the point clouds, and notify the user of the dangerous if they do
        # not match
        crsList = [pc.crs for pc in self.proj.inputFiles.clouds]
        if crsList.count(self.proj.inputFiles(0).crs) != len(crsList):
            Gui.popup_ok('Files do not have matching spatial reference systems. Unmatched reference systems may cause '
                         'unreliable results.',
                         title='Caution - Unmatched Spatial Reference System')

        # Set the QGIS project coordinate reference to match that of the first point cloud
        crs = QgsCoordinateReferenceSystem(self.proj.inputFiles(0).crs)
        self.proj.qgsProj.setCrs(crs)

        # Create a shapefile representing the boundary of the input point clouds
        shapes = self.proj.inputFiles.boundaryShapefile(f"{self.proj.fileName[:-5]}_input_bounds",
                                                        'Input Point Cloud Boundaries')

        # Add the AOI and input file boundary shapefiles into QGIS
        self.proj.qgsProj.addMapLayer(self.proj.AOI.qObj)
        self.proj.qgsProj.addMapLayer(shapes)
        # Reduce the opacity of the boundary shapefile to not obscure visibility

        # Set the visual extent for the project when viewed in QGIS.
        bbox = self.proj.inputFiles.boundingBox()
        extent = QgsRectangle(bbox[0], bbox[1], bbox[2], bbox[3])
        self.proj.qgsProj.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent, crs))

        # Apply a style to the layers
        layer1 = self.proj.qgsProj.mapLayersByName('AOI')[0]
        layer2 = self.proj.qgsProj.mapLayersByName('Input Point Cloud Boundaries')[0]
        style1 = QgsStyle.defaultStyle().symbol('outline red')
        style2 = QgsStyle.defaultStyle().symbol('outline green')
        layer1.renderer().setSymbol(style1)
        layer2.renderer().setSymbol(style2)
        layer1.triggerRepaint()
        layer2.triggerRepaint()

        # Reduce the opacity of the boundary shapefile to not obscure visibility
        layer2.setOpacity(0.3)

    # This method allows the addition of water features to mask those areas in processing
    def addWater(self):
        self.win['__ADD_WATER__'].Update(disabled=True)
        self.win['_OUT_PRINT_' + Gui.WRITE_ONLY_KEY].print('=' * 60)
        # Create a layout for a custom window to select water features.
        layout0 = [[Gui.Text('Select a shapefile that contains water polygons for the area. If none are required, press'
                             ' Cancel.',
                             size=(30, 3))],
                   [Gui.Checkbox('Filtering Required',
                                 default=True,
                                 tooltip='Check this box if the provided file needs to be subset or filtered',
                                 key='__FILTER__')],
                   [Gui.Input(key='__ADD_FILE__',
                              size=(1, 1),
                              enable_events=True,
                              visible=False),
                    Gui.FileBrowse(button_text='Add Shape File',
                                   target='__ADD_FILE__',
                                   key='ADD',
                                   file_types=(('ESRI Shape Files', '*.shp'),
                                               ('All Files', '*.* *')))]]

        layout1 = [[Gui.Text('Which fields would you like to filter on?',
                             size=(30, 2))],
                   [Gui.Listbox([],
                                select_mode=Gui.LISTBOX_SELECT_MODE_MULTIPLE,
                                size=(30, 10),
                                key='__FIELD_LIST__')],
                   [Gui.ProgressBar(1,
                                    orientation='horizontal',
                                    size=(20, 15),
                                    key='__SHAPE_NUM__',
                                    relief='RELIEF_SUNKEN',
                                    visible=False)]]

        waterWin = Gui.Window('Select Water Feature Shapefile',
                              [[Gui.Column(layout0,
                                           key='_COL_0_'),
                                Gui.Column(layout1,
                                           key='_COL_1_',
                                           visible=False)],
                               [Gui.Ok(disabled=True), Gui.Cancel()]],
                              layout0,
                              modal=True)

        # This counter will determine what set of operations is performed by the Ok button
        i = 0

        # Water feature window event loop
        while True:
            e, v = waterWin.read()

            # Exit the loop and return an empty if the main window is closed, or cancel is selected
            if e == Gui.WINDOW_CLOSED:
                break

            elif e == 'Cancel':
                self.next()
                break

            elif e == '__ADD_FILE__':
                waterWin['Ok'].Update(disabled=False)

            elif e == 'Ok' and i == 0:
                # If the user does not want to filter the shapefile, make it the water features file.
                if not v['__FILTER__']:
                    self.proj.waterFeatures(v['__ADD_FILE__'], "Water Features")
                    self.print(f'{v["__ADD_FILE__"]} added as project water feature file.')
                    self.next()
                    break

                # Otherwise, determine how the user wishes to filter the shapefile by switching to COL 1.
                else:
                    unFilt = ShapeFile()
                    unFilt(v['__ADD_FILE__'], f"{self.proj.name} Water Features")

                    # Clip the shapefile to the AOI.
                    lay = self.proj.AOI.oObj.GetLayer()
                    feat = lay.GetFeature(0)
                    geom = feat.GetGeometryRef()
                    unFilt.oObj.GetLayer().SetSpatialFilter(geom)
                    i += 1
                    waterWin['__FIELD_LIST__'].Update(values=unFilt.fields())
                    waterWin['Cancel'].Update(disabled=True)
                    waterWin['_COL_0_'].Update(visible=False)
                    waterWin['_COL_1_'].Update(visible=True)

            elif e == 'Ok' and i == 1:
                self.print(f"Finding unique field sets for {unFilt.count:,} records. Please wait...")
                startTime = time.time()
                waterWin['__SHAPE_NUM__'].Update(current_count=0, visible=True, max=unFilt.count)
                waterWin['Ok'].Update(disabled=True)
                waterWin.refresh()
                fieldList = v['__FIELD_LIST__']
                values = []

                # Reset the reading on the OGR object to ensure you are on the first record
                lay = unFilt.oObj.GetLayer()
                lay.ResetReading()

                # Read through each record, and store the value to an array
                feat = lay.GetNextFeature()
                j = 0
                while feat is not None:
                    x = feat.ExportToJson(as_object=True)['properties']
                    values.append(tuple([x[param] for param in fieldList]))
                    j += 1
                    if j % 10 == 0:
                        waterWin['__SHAPE_NUM__'].Update(current_count=j)
                        waterWin.refresh()
                    feat = lay.GetNextFeature()

                # Because a set can only have distinct elements, the set of the list values will return unique tuples.
                # Converting that set to a list makes it iterable for other uses.
                uniqueValues = list(set(values))
                self.print(f"{len(uniqueValues)} unique field sets found in {secFmt(time.time() - startTime)}")
                i += 1
                break

        # Close the pop_up window
        waterWin.close()

        # Once distinct values have been generated, display the values in a table to be selected from in a new window
        if i == 2:
            # Generate the layout for the table window
            layout2 = [[Gui.Text('Which features would you like to keep?',
                                 size=(30, 1))],
                       [Gui.Table(sorted(uniqueValues),
                                  headings=fieldList,
                                  size=(20, 15),
                                  select_mode=Gui.TABLE_SELECT_MODE_EXTENDED,
                                  key='__FILT_TABLE__',
                                  tooltip='Hold CTRL while clicking rows to select multiple rows.')],
                       [Gui.Ok()]]

            tabWin = Gui.Window('Select Water Feature Shapefile',
                                layout2,
                                modal=True)
            # Table Window Event loop
            while True:
                e, v = tabWin.read()

                # Exit the loop and return an empty if the main window is closed
                if e == Gui.WINDOW_CLOSED:
                    break

                if e == 'Ok':
                    # Create a conditional SQL clause based on the user's selected filters
                    sqlArgs = []
                    for pos in v['__FILT_TABLE__']:
                        eq = []
                        for k in range(0, len(fieldList)):
                            eq.append(f"{fieldList[k]} = '{sorted(uniqueValues)[pos][k]}'")
                        sqlArgs.append(f"({' AND '.join(tuple(eq))})")

                    query = ' OR '.join(tuple(sqlArgs))
                    # Create a new shapefile using the filter
                    unFilt.filteredCopy(f'{self.proj.fileName[:-5]}_water', 'Water Features', query)
                    watPath = f'shapefiles/{self.proj.fileName[:-5]}_water/{self.proj.fileName[:-5]}_water.shp'

                    self.proj.waterFeatures(watPath, 'Water Features')
                    self.proj.waterFeatures.createQgsLayer()
                    self.proj.qgsProj.addMapLayer(self.proj.waterFeatures.qObj)
                    self.next()
                    break

            tabWin.close()

            # Apply a style to the layer
            layer = self.proj.qgsProj.mapLayersByName('Water Features')[0]
            style = QgsStyle.defaultStyle().symbol('topo water')
            layer.renderer().setSymbol(style)
            layer.triggerRepaint()

    # -----------------------------------------------------------------------------------------------------------------
    # This method merges or splits the point cloud as appropriate
    def mergeSplit(self):
        self.win['__MERGE/SPLIT__'].Update(disabled=True)
        self.win['_OUT_PRINT_' + Gui.WRITE_ONLY_KEY].print('=' * 60)

        # Determine how many tiles the points should be divided into so that it can be processed
        tiles = 1000 / (psutil.virtual_memory().total / self.proj.inputFiles.noPoints())

        # If the above calculation returns a number <= 1, there are few enough points to merge the input files
        if tiles <= 1:
            self.print(f'Merging {len(self.proj.inputFiles.clouds)} file(s). Please wait...')
            merge = self.proj.inputFiles.merge(f'{self.proj.fileName[:-4]}las')
            self.proj.mergedFiles.add(merge)
            self.print(f'Point clouds merged in {secFmt(self.proj.mergedFiles(0).mergeTime)}')
            self.next()

        # Otherwise, split each point cloud, and then merge overlapping tiles.
        else:
            # Get the bounding box around the point cloud, and round the minimum values to even values
            bbox = self.proj.inputFiles.boundingBox()
            brange = [bbox[2] - bbox[0], bbox[3] - bbox[1]]
            origin = numpy.floor(bbox[:2])

            # Generate the layout for the dialog box
            splLay = [[Gui.Text("Tile Generator",
                                size=28,
                                font=('Arial', 16, 'bold'),
                                justification='center')],
                      [Gui.Text('X Range:',
                                size=12,
                                justification='right'),
                       Gui.Text(f'{bbox[0]:,.3f} to {bbox[2]:,.3f}',
                                size=25,
                                justification='left')],
                      [Gui.Text('',
                                size=12,
                                justification='right'),
                       Gui.Text(f'[{brange[0]:,.3f}]',
                                size=25,
                                justification='left')],
                      [Gui.Text('Y Range:',
                                size=12,
                                justification='right'),
                       Gui.Text(f'{bbox[1]:,.3f} to {bbox[3]:,.3f}',
                                size=25,
                                justification='left')],
                      [Gui.Text('',
                                size=12,
                                justification='right'),
                       Gui.Text(f'[{brange[1]:,.3f}]',
                                size=25,
                                justification='left')],
                      [Gui.Text('Minimum tiles:',
                                size=12,
                                justification='right'),
                       Gui.Text(f'{numpy.ceil(tiles)}',
                                size=25,
                                justification='left')],
                      [Gui.Text('Estimated Tiles:',
                                size=12,
                                justification='right'),
                       Gui.Text(f'{numpy.ceil(brange[0] / 50) * numpy.ceil(brange[1] / 50)}',
                                size=25,
                                justification='left',
                                key='_ESTIMATED_TILES_')],
                      [Gui.Slider(range=(50, max(brange)),
                                  default_value=50,
                                  resolution=25,
                                  tick_interval=200,
                                  orientation='h',
                                  enable_events=True,
                                  size=(37, 10),
                                  key='_TILE_SIZE_')],
                      [Gui.Button('Split Tiles',
                                  key='_SPLIT_')]]

            # Define the tiling window
            tileWin = Gui.Window('Create LiDAR Tiles', splLay)

            # Event loop for the tiling dialog
            while True:
                e, v = tileWin.read()

                # If the window is closed, return nothing
                if e == Gui.WINDOW_CLOSED:
                    Gui.popup_ok("Split not executed.")
                    return

                # If the slider is moved, update the estimated number of tiles, and disable the Split button if it is
                # too few
                elif e == '_TILE_SIZE_':
                    across = numpy.ceil(brange[0] / v['_TILE_SIZE_'])
                    up = numpy.ceil(brange[1] / v['_TILE_SIZE_'])
                    if across * up < tiles:
                        tileWin['_ESTIMATED_TILES_'].update(f'{across * up}',
                                                            text_color='red',
                                                            background_color='white',
                                                            font=('Arial', 13, 'bold'))

                    else:
                        tileWin['_ESTIMATED_TILES_'].update(f'{across * up}',
                                                            text_color='White',
                                                            background_color='#64778d',
                                                            font=('Arial', 11, 'normal'))

                    tileWin['_SPLIT_'].update(disabled=across * up < tiles)
                    tileWin.refresh()

                # Once the split button is pressed, return the value of the slider
                elif e == '_SPLIT_':
                    a = v['_TILE_SIZE_']
                    break

            # Close the dialog, and return the value.
            tileWin.close()

            # Notify the user that it takes a long time to process the files, and give them the option to cancel
            if Gui.popup_yes_no("This process can take several minutes to several hours to complete, depending on "
                                "your computer's hardware. Are you sure you would like to "
                                "continue?",
                                title="Long Process Warning") == 'No':
                Gui.popup_ok("Split not executed.")
                return

            self.print(f"Splitting {len(self.proj.inputFiles.clouds)} file(s). Please wait.")

            # Keep track of how long the entire process takes
            startTime = time.time()

            # Loop through the files to split
            for i in range(0, len(self.proj.inputFiles.clouds)):
                self.win['_SEL_FILE_'].Update(set_to_index=i)
                self.win.refresh()
                self.updateStats()
                self.win['_OUT_PRINT_' + Gui.WRITE_ONLY_KEY].print('=' * 60)
                self.print(f"Splitting {self.proj.inputFiles(i).name}")
                outList = self.proj.inputFiles(i).splitCloud(origin, a)
                self.print(f"{self.proj.inputFiles(i).name} split in {secFmt(self.proj.inputFiles(i).splitTime)}")
                self.print(f"Now scanning {len(outList)} tiles.\n")

                # Once a file has been split, scan each of the new tiles to compare them
                for pc in outList:
                    self.proj.tileFiles.add(PointCloud())
                    self.print(f'Scanning tile {outList.index(pc) + 1}/{len(outList)}')
                    self.proj.tileFiles(-1)(pc)
                    self.proj.tileFiles(-1).scan()
                    self.print(f'Tile scanned in {secFmt(self.proj.tileFiles(-1).loadTime)}\n')

            # Once the files have been split into tiles, merge any overlapping tiles.
            # Start by creating a grid of bounding boxes
            grid = []
            xcor = origin[0] + a / 10
            while xcor < bbox[2]:
                ycor = origin[1] + a / 10
                while ycor < bbox[3]:
                    grid.append([xcor - a / 10,
                                 ycor - a / 10,
                                 xcor + a + a / 10,
                                 ycor + a + a / 10])
                    ycor += a
                xcor += a

            # Iterate through the grid and the tiles to find the grid square that each tile fits into.
            mergeDict = {}
            for i in range(0, len(grid)):
                box = grid[i]
                mergeDict[f'box_{i}'] = []
                for pc in self.proj.tileFiles.clouds:
                    if pc.minx >= box[0] and pc.miny >= box[1] and pc.maxx <= box[2] and pc.maxy <= box[3]:
                        mergeDict[f'box_{i}'].append(pc)

            # Iterate through the boxes in the merge dictionary, and merge the boxes that include point clouds
            i = 0
            self.win['_OUT_PRINT_' + Gui.WRITE_ONLY_KEY].print('=' * 60)
            for box in mergeDict:
                if len(mergeDict[box]) > 0:
                    x = PointCloudCollection()
                    x.add(mergeDict[box])
                    self.print(f'Merging {self.proj.fileName[:-5]}_{i}.las. Please wait...')
                    pc = x.merge(f'{self.proj.fileName[:-5]}_{i}.las')
                    self.print(f'{pc.name} merged in {secFmt(pc.mergeTime)}\n')
                    self.proj.mergedFiles.add(pc)
                    i += 1

            self.print(f"All files split and merged in {secFmt(time.time() - startTime)}")
            self.next()

            # Ask the user if they would like to remove the intermediary files to save disk space
            # Calculate the size of the tiles folder
            fold_size = 0
            for file in os.listdir('las_files/tiles'):
                fold_size += os.path.getsize(f'las_files/tiles/{file}')

            if 'Yes' == Gui.popup_yes_no(f'This program has created a total of {sizeConvert(fold_size)} worth of '
                                         f'intermediary files which are no longer required? Would you like to delete'
                                         f'them?',
                                         title='Delete intermediate files?'):
                for file in os.listdir('las_files/tiles'):
                    os.remove(f'las_files/tiles/{file}')

        # Update the combobox and list box in the main window, and close the dialog box
        self.win['_LIST_SELECTOR_'].Update(values=self.proj.pcLists(),
                                           set_to_index=0)
        self.changeCombo()

    # -----------------------------------------------------------------------------------------------------------------
    # This method classifies the split/merged point clouds
    def classify(self):
        self.win['__CLASSIFY__'].Update(disabled=True)

        self.win['_OUT_PRINT_' + Gui.WRITE_ONLY_KEY].print('=' * 60)
        self.print(f'Classifying {len(self.proj.mergedFiles.clouds)} point cloud tiles. Please wait.')
        # Keep track of how long the entire process takes.
        startTime = time.time()

        for i in range(0, len(self.proj.mergedFiles.clouds)):
            self.print(f'Classifying ground points for {self.proj.mergedFiles(i).name}')
            self.proj.mergedFiles(i).classify()
            self.print(f'{self.proj.mergedFiles(i).name} ground points classified in '
                       f'{secFmt(self.proj.mergedFiles(i).classTime)}\n')

        self.print(f'All point clouds classified in {secFmt(time.time() - startTime)}\n')
        self.next()

    # -----------------------------------------------------------------------------------------------------------------
    # This method generates a single DEM from the split/merged point clouds
    def gen_DEM(self):
        self.win['__GEN_DEM__'].Update(disabled=True)

        # Iterate through each resolution of the DEM
        for resStr in self.proj.DEM:
            reso = float(resStr.split('_')[0])/100
            path = f'rasters/{resStr}'
            dem_name = f'{self.proj.fileName[:-5]}_{resStr}'
            self.win['_OUT_PRINT_' + Gui.WRITE_ONLY_KEY].print('=' * 60)
            self.print(f'Generating {resStr.replace("_"," ")} DEM. Please wait...')

            # If there is only a single point cloud, generate a DEM from that point cloud
            if len(self.proj.mergedFiles.clouds) == 1:
                self.proj.DEM[resStr] = self.proj.mergedFiles(-1).createDEM(path,
                                                                            dem_name,
                                                                            reso)
                self.print(f'DEM generated in {secFmt(self.proj.DEM[resStr].time)}')

            # Otherwise, each tile will be converted to a raster, and then they will be merged.
            else:
                self.print(f'Generating {len(self.proj.mergedFiles.clouds)} DEM tiles. Please wait.')
                # Generate a DEM for each tile
                for i in range(0, len(self.proj.mergedFiles.clouds)):
                    self.print(f'Generating {resStr.replace("_"," ")} DEM tile for {self.proj.mergedFiles(i).name}')
                    self.proj.DEMTiles[resStr].add(self.proj.mergedFiles(i).createDEM(f'rasters/{resStr}/tiles',
                                                                                      self.proj.mergedFiles(i).name,
                                                                                      reso))
                    self.print(f'{self.proj.mergedFiles(i).name} {resStr.replace("_"," ")} DEM tile generated in '
                               f'{secFmt(self.proj.DEMTiles[resStr](-1).time)}\n')
                self.print(f'Merging {len(self.proj.DEMTiles[resStr].tifs)} DEM tiles. Please wait...')
                self.proj.DEM[resStr] = self.proj.DEMTiles[resStr].merge(f'{path}/{dem_name}.tif',
                                                                         dem_name)
                self.print(f'DEM merged in {secFmt(self.proj.DEM[resStr].time)}')

            # Fill any voids in the DEM
            self.print('Filling DEM voids.')
            self.proj.DEM[resStr].fillVoids()
            self.print('DEM voids filled.')

            # Create a group for the data at that resolution
            root = self.proj.qgsProj.layerTreeRoot()                                      # Find the layer tree root
            group = root.insertGroup(-1, resStr)

            # Add the DEM to the QGIS project, and place it in the group
            # Taken from the PyQGIS Cookbook
            rastLayer = self.proj.qgsProj.addMapLayer(self.proj.DEM[resStr].qObj, False)  # Don't display map layer
            group.addLayer(rastLayer)                                                     # Insert the raster at the end

        # Go to the next step
        self.next()

    # -----------------------------------------------------------------------------------------------------------------
    # This method finds sinkholes by following the method in Maracle's ArcGIS toolbox
    def findSink(self):
        # Iterate through each resolution of the DEM
        for resStr in self.proj.DEM:
            self.win['_OUT_PRINT_' + Gui.WRITE_ONLY_KEY].print('=' * 60)
            self.print(f'Finding sinkholes from {resStr.replace("_", " ")} DEM. Please wait')

            # Select the group for adding shapefile layers
            root = self.proj.qgsProj.layerTreeRoot()
            group = root.findGroup(resStr)

            # Define the path for rasters, shapefiles, and the filename prefix
            rPath = f'rasters/{resStr}'
            sPath = f'shapefiles/{resStr}'
            name = f'{self.proj.fileName[:-5]}_{resStr}'

            # The first step is to create a duplicate raster that has depressions in the surface filled
            self.print('Creating a filled copy of the DEM.')
            startTime = time.time()
            arg_params = {'input': self.proj.DEM[resStr].qObj,
                          'format': 1,
                          'output': f'{rPath}/{name}_filled.tif',
                          'direction': f'{rPath}/{name}_dir.tif',
                          'areas': f'{rPath}/{name}_areas.tif'}

            temp_rast = processing.run('grass7:r.fill.dir', arg_params)['output']
            self.print(f'Filled DEM generated in {secFmt(time.time() - startTime)}\n')

            # The raster calculator is used to compute the difference between the filled DEM and the original DEM to
            # create a preliminary sinkhole raster
            self.print('Creating preliminary sinkhole raster.')
            startTime = time.time()
            arg_params = {'INPUT_A': temp_rast,
                          'BAND_A': 1,
                          'INPUT_B': self.proj.DEM[resStr].qObj,
                          'BAND_B': 1,
                          'FORMULA': 'A - B',
                          'RTYPE': 5,
                          'OUTPUT': f'{rPath}/{name}_diff.tif'}

            temp_rast = processing.run('gdal:rastercalculator', arg_params)['OUTPUT']
            self.print(f'Preliminary sinkhole raster generated in {secFmt(time.time() - startTime)}\n')

            # The slope/aspect tool will measure the slope of the raster at every pixel, as well as the vertical and
            # tangential curvature.
            self.print('Measuring sinkhole slope and curvature')
            startTime = time.time()
            arg_params = {'elevation': temp_rast,
                          'slope': f'{rPath}/{name}_slope.tif',
                          'pcurvature': f'{rPath}/{name}_pcurve.tif',
                          'tcurvature': f'{rPath}/{name}_tcurve.tif'}

            processing.run('grass7:r.slope.aspect', arg_params)
            self.print(f'Slope and curvature computed in {secFmt(time.time() - startTime)}\n')

            # The slope and aspect rasters will be added to the project.
            r_layer = QgsRasterLayer(f'{rPath}/{name}_slope.tif', f'{name}_slope')
            self.proj.qgsProj.addMapLayer(r_layer, False)
            layer = group.insertLayer(-1, r_layer)
            provider = r_layer.dataProvider()
            renderer = QgsSingleBandGrayRenderer(provider, 1)
            dataType = renderer.dataType(1)
            enhancement = QgsContrastEnhancement(dataType)
            contrast = QgsContrastEnhancement.StretchToMinimumMaximum
            enhancement.setContrastEnhancementAlgorithm(contrast, True)
            enhancement.setMinimumValue(0.0)
            enhancement.setMaximumValue(90.0)
            layer.layer().setRenderer(renderer)
            layer.layer().renderer().setContrastEnhancement(enhancement)
            layer.layer().triggerRepaint()

            # If water features have been added to the project combine them with the AOI file
            if self.proj.waterFeatures.path != '':

                # Ensure that the water features geometry has no geometric errors that would hamper processing
                self.print('Checking water features for geometry errors')
                startTime = time.time()
                arg_params = {'INPUT': self.proj.qgsProj.mapLayersByName('Water Features')[0],
                              'METHOD': 1,
                              'OUTPUT': 'TEMPORARY_OUTPUT'}

                temp_poly = processing.run('native:fixgeometries', arg_params)['OUTPUT']
                self.print(f'Check complete in {secFmt(time.time() - startTime)}')

                # The water features will be erased from the AOI to create the processing area
                self.print('Removing water features from AOI.')
                if not os.path.exists(f'{sPath}/AOI_edited/'):
                    os.makedirs(f'{sPath}/AOI_edited')

                startTime = time.time()
                arg_params = {'INPUT': self.proj.qgsProj.mapLayersByName('AOI')[0],
                              'OVERLAY': temp_poly,
                              'OUTPUT': f'{sPath}/AOI_edited/AOI_edited.shp'}

                processing.run('qgis:difference', arg_params)
                clip_AOI = ShapeFile()
                clip_AOI(f'{sPath}/AOI_edited/AOI_edited.shp', 'Edited AOI')
                clip_AOI.createQgsLayer()
                self.print(f'AOI clipped in {secFmt(time.time() - startTime)}\n')

                # Clip the preliminary sinkhole raster with the edited AOI to only process areas of interest.
                self.print('Clipping preliminary sinkhole raster.')

            # If water features have not been added to the project, set the AOI as the clip polygon
            else:
                clip_AOI = self.proj.AOI

            startTime = time.time()
            arg_params = {'INPUT': temp_rast,
                          'MASK': clip_AOI.qObj,
                          'SOURCE_CRS': clip_AOI.qObj.crs(),
                          'TARGET_CRS': clip_AOI.qObj.crs(),
                          'OUTPUT': f'{rPath}/{self.proj.fileName[:-5]}_diff_clipped.tif'}

            temp_rast = processing.run('gdal:cliprasterbymasklayer', arg_params)['OUTPUT']
            self.print(f'Sinkhole raster clipped in {secFmt(time.time() - startTime)}\n')

            # Using the GRASS r.neighbours operator on the clipped raster produces the same result as the ArcGIS Pro
            # Spatial Analyst Filter tool set to low band pass
            self.print('Smoothing preliminary sinkhole raster.')
            startTime = time.time()
            arg_params = {'input': temp_rast,
                          'method': 0,
                          'size': 3,
                          'output': f'{rPath}/{self.proj.fileName[:-5]}_filt.tif'}

            temp_rast = processing.run('grass7:r.neighbors', arg_params)['output']
            self.print(f'Sinkhole raster smoothed in {secFmt(time.time() - startTime)}\n')

            # The resulting raster is reclassified to create a binary raster. A value of 1 will be assigned to cells
            # that are deeper than 0.15m which corresponds to the vertical accuracy of LiDAR data. All other cells will
            # have a value of 0.
            self.print('Classifying sinkholes by depth.')
            # Get raster band statistics in QGIS
            tmp_layer = QgsRasterLayer(temp_rast, 'Temporary Layer')
            dataProvider = tmp_layer.dataProvider()
            bandStats = dataProvider.bandStatistics(1, QgsRasterBandStats.Max)
            r_max = bandStats.maximumValue

            arg_params = {'INPUT_RASTER': temp_rast,
                          'RASTER_BAND': 1,
                          'TABLE': ['0.15', f'{numpy.ceil(r_max) + 1}', '1'],
                          'OUTPUT': f'{rPath}/{self.proj.fileName[:-5]}_reclass.tif'}

            temp_rast = processing.run('native:reclassifybytable', arg_params)['OUTPUT']
            self.print(f'Sinkhole raster reclassified in {secFmt(time.time() - startTime)}\n')

            # The reclassified raster is converted to polygons. Each polygon will represent a depression in the surface.
            self.print('Converting sinkhole raster to polygons.')
            startTime = time.time()

            poly_root = f'{sPath}/{self.proj.fileName[:-5]}_sink_rast_poly'
            poly_fn = f'{self.proj.fileName[:-5]}_sink_rast_poly.shp'

            if not os.path.exists(poly_root):
                os.makedirs(poly_root)

            arg_params = {'INPUT': temp_rast,
                          'BAND': 1,
                          'FIELD': 'SINK_CLASS',
                          'OUTPUT': f'TEMPORARY_OUTPUT'}

            # Running this process will save the shapefile to a temporary layer, and return a QgsVectorLayer
            temp_poly = QgsVectorLayer(processing.run('gdal:polygonize', arg_params)['OUTPUT'], 'TEMP')

            # Select the features that correspond to sinkholes, and save them to file
            temp_poly.selectByExpression('"SINK_CLASS" = 1')
            arg_params = {'INPUT': temp_poly,
                          'OUTPUT': f'{poly_root}/{poly_fn}'}

            temp_poly = processing.run('native:saveselectedfeatures', arg_params)['OUTPUT']

            sink_rast_poly = ShapeFile()
            sink_rast_poly(f'{poly_root}/{poly_fn}', 'Sink Raster Polygons')

            self.print(f'Sinkhole raster converted to polygons in {secFmt(time.time() - startTime)}\n')

            # .........................................................................................................
            # Each processing algorithm generates a new shapefile, and joining them will produce all the fields required
            # to categorize each feature to determine if they are likely to be a sinkhole. Rather than saving multiple
            # shapefiles to the project folder, temporary outputs are used to generate each of the fields required, and
            # a new shapefile will be created when all the fields have been joined.
            # .........................................................................................................

            # The resulting polygons have a jagged appearance inherited from the pixelation of the Raster. To reduce the
            # effects of pixelation, the polygons will be smoothed.
            self.print('Smoothing sinkhole polygons')

            arg_params = {'INPUT': temp_poly,
                          'OFFSET': 0.5,
                          'OUTPUT': f'TEMPORARY_OUTPUT'}

            tmp_poly = processing.run('native:smoothgeometry', arg_params)['OUTPUT']

            # Add a feature id to each feature in the vector layer so multiple files can be joined later in the
            # algorithm
            self.print('Creating FEAT_ID field')
            arg_params = {'INPUT': tmp_poly,
                          'FIELD_NAME': 'FEAT_ID',
                          'OUTPUT': 'TEMPORARY_OUTPUT'}

            tmp_poly = processing.run('native:addautoincrementalfield', arg_params)['OUTPUT']

            # To determine the aspect ratio of each shape, the major and minor axis of the shapes need to be computed.
            # By computing the minimum oriented rectangle, the closest fit of a rectangle oriented along the major axis
            # and spanning across the minor axis is produced, providing the length of both sides. While not exact, it
            # should provide a close approximation of the length of both axes.
            self.print('Calculating shapefile minor/major axes lengths')
            arg_params = {'INPUT': tmp_poly,
                          'FIELD': 'FEAT_ID',
                          'TYPE': 1,
                          'OUTPUT': 'TEMPORARY_OUTPUT'}

            rect = processing.run('qgis:minimumboundinggeometry', arg_params)['OUTPUT']

            # Join these calculated fields back to the original temporary file
            self.print('Joining shapefiles.')
            arg_params = {'INPUT': tmp_poly,
                          'FIELD': 'FEAT_ID',
                          'INPUT_2': rect,
                          'FIELD_2': 'FEAT_ID',
                          'FIELDS_TO_COPY': ['width', 'height'],
                          'PREFIX': 'r_',
                          'OUTPUT': 'TEMPORARY_OUTPUT'}

            tmp_poly = processing.run('native:joinattributestable', arg_params)['OUTPUT']

            # For the remaining attributes, they can be computed with successive instances of the field calculator
            # algorithm
            # Create a list of field name and formula pairs so this process can be iterated over
            field_list = [['Aspect', '"r_width" / "r_height"'],
                          ['Roundness', 'roundness(@geometry)'],
                          ['Convex', 'area(@geometry) / area(convex_hull(@geometry))'],
                          ['Area', 'area(@geometry)'],
                          ['Perimeter', 'perimeter(@geometry)'],
                          ['Sinkhole_S', '("Convex" ^ 2) * ("Aspect" ^ 1.5) * "Roundness"']]

            for pair in field_list:
                self.print(f'Calculating {pair[0]} field.')
                arg_params = {'INPUT': tmp_poly,
                              'FIELD_NAME': pair[0],
                              'FIELD_TYPE': 0,
                              'FIELD_LENGTH': 12,
                              'FIELD_PRECISION': 5,
                              'FORMULA': pair[1],
                              'OUTPUT': 'TEMPORARY_OUTPUT'}

                tmp_poly = processing.run('native:fieldcalculator', arg_params)['OUTPUT']

            # Using zonal statistics, the slope and curvature information can be added to the depression shapefile.
            self.print('Computing zonal statistics')
            arg_params = {'INPUT': tmp_poly,
                          'INPUT_RASTER': f'{rPath}/{name}_slope.tif',
                          'RASTER_BAND': 1,
                          'COLUMN_PREFIX': 'S_',
                          'STATISTICS': [2, 3, 5, 6],
                          'OUTPUT': 'TEMPORARY_OUTPUT'}

            tmp_poly = processing.run('native:zonalstatisticsfb', arg_params)['OUTPUT']

            arg_params = {'INPUT': tmp_poly,
                          'INPUT_RASTER': f'{rPath}/{name}_tcurve.tif',
                          'RASTER_BAND': 1,
                          'COLUMN_PREFIX': 'T_',
                          'STATISTICS': [2, 3, 5, 6],
                          'OUTPUT': 'TEMPORARY_OUTPUT'}

            tmp_poly = processing.run('native:zonalstatisticsfb', arg_params)['OUTPUT']

            arg_params = {'INPUT': tmp_poly,
                          'INPUT_RASTER': f'{rPath}/{name}_pcurve.tif',
                          'RASTER_BAND': 1,
                          'COLUMN_PREFIX': 'P_',
                          'STATISTICS': [2, 3, 5, 6],
                          'OUTPUT': 'TEMPORARY_OUTPUT'}

            tmp_poly = processing.run('native:zonalstatisticsfb', arg_params)['OUTPUT']

            # Once all fields have been calculated, save the layer
            self.print('Saving shapefile.')
            poly_root = f'shapefiles/{resStr}/{self.proj.fileName[:-5]}_sink_rast_smoothed'
            poly_fn = f'{self.proj.fileName[:-5]}_sink_rast_smoothed.shp'
            if not os.path.exists(poly_root):
                os.makedirs(poly_root)
            arg_params = {'INPUT': tmp_poly,
                          'OUTPUT': f'{poly_root}/{poly_fn}'}

            processing.run('native:savefeatures', arg_params)

            # Apply a style to the vector layer to highlight low and high probability
            self.proj.depressions[resStr](f'{poly_root}/{poly_fn}', f'{resStr}_Smoothed_Depressions')
            self.proj.depressions[resStr].createQgsLayer()
            geom = self.proj.depressions[resStr].qObj.geometryType()
            self.proj.qgsProj.addMapLayer(self.proj.depressions[resStr].qObj, False)
            layer = group.insertLayer(0, self.proj.depressions[resStr].qObj)

            rangeDict = {
                'Low Possibility': {
                    'Colour': QtGui.QColor('#ffa500'),
                    'Opacity': 0.4,
                    'Min': 0.0,
                    'Max': 0.49999},
                'High Possibility': {
                    'Colour': QtGui.QColor('#00ff00'),
                    'Opacity': 1,
                    'Min': 0.5,
                    'Max': 1.0}}
            rangeList = []

            for r in rangeDict:
                sym = QgsSymbol.defaultSymbol(geom)
                sym.setColor(rangeDict[r]['Colour'])
                sym.setOpacity(rangeDict[r]['Opacity'])
                rg = QgsRendererRange(rangeDict[r]['Min'], rangeDict[r]['Max'], sym, r)
                rangeList.append(rg)

            rend = QgsGraduatedSymbolRenderer('', rangeList)
            meth = QgsApplication.classificationMethodRegistry().method('EqualInterval')
            rend.setClassificationMethod(meth)
            rend.setClassAttribute('Sinkhole_S')
            layer.layer().setRenderer(rend)
            layer.layer().triggerRepaint()

            self.proj.save()

        self.next()

    # This method returns the results of the analysis
    def results(self):
        # Create a layout for the results window.
        layout = [[Gui.Text('Results',
                            font=('Arial', 13, 'bold'))],
                  [],
                  [Gui.Button('Open Project',
                              key='__OPEN__')]]

        # Iterate though different ranges of sinkhole scores to count how many features are in each range
        table_vals = {}

        for resStr in self.proj.depressions:
            layer = self.proj.depressions[resStr].oObj.GetLayer()
            table_vals[resStr] = []

            for i in range(5):
                filt = f"Sinkhole_S <= {1.0 - i * 0.1} AND Sinkhole_S > {0.9 - i * 0.1}"
                layer.SetAttributeFilter(filt)
                table_vals[resStr].append([f'{90 - i * 10} - {100 - i * 10}%', layer.GetFeatureCount()])

            filt = f'Sinkhole_S <= 0.5'
            layer.SetAttributeFilter(filt)
            table_vals[resStr].append([f'Potential Sinkholes', layer.GetFeatureCount()])
            table_vals[resStr].append(['Total Depressions', self.proj.depressions[resStr].count])

            col = [[Gui.Text(f'{resStr.replace("_"," ")} DEM Sinkhole Breakdown')],
                   [Gui.Table(values=table_vals[resStr],
                              headings=['Sinkhole Score', 'Count'],
                              num_rows=8,
                              size=(20, 7))]]

            layout[1].append(Gui.Column(col,
                                        element_justification='center'))

        result_win = Gui.Window('Analysis Results',
                                layout,
                                modal=True)

        e, v = result_win.read()

        if e == '__OPEN__':
            self.print('Opening project...')
            # Close the Windows
            result_win.close()
            self.win.close()

            # Close the QGIS instance
            qgs.exitQgis()

            # Open the QGIS project file
            os.startfile(f'{self.proj.fileName[:-4]}qgz')

            # Exit the python program
            exit()

        result_win.close()


# =====================================================================================================================
# FUNCTIONS
# =====================================================================================================================
# This function converts a string into a string that can be used as a valid filename
def validFilename(instr):
    instr = instr.strip()
    outstr = ''
    for char in instr:
        if char.isalnum() or char in "'[]{}()-_":
            outstr += char
        else:
            outstr += '_'

    return outstr


# This function converts a time in seconds to a HH:MM:SS.sss format for printing purposes
def secFmt(sec):
    t = [0, 0, 0]
    t[2] = sec % 60
    t[1] = ((sec - t[2]) / 60) % 60
    t[0] = numpy.floor(sec / 3600)

    out = f'{int(t[1]):02d}:{t[2]:06.3f}'
    if t[0] > 0:
        out = f'{int(t[0]):02d}:{out}'

    return out


# This function converts a file size input in bytes to a human-readable format
def sizeConvert(size: int) -> str:
    i = 0

    while size > 1000 and i < 7:
        size /= 1000
        i += 1

    suf = ['B', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB']

    return f'{size:,.2f} {suf[i]}'


# =====================================================================================================================
# MAIN LOOP
# =====================================================================================================================
# Initialize the project
prog = DetectorGUI()
prog()
