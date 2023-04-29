# LiDAR_sinkhole_detector
A tool created to take an input of an airborne LiDAR dataset of an area, and automatically detect potential sinkholes in that geographic area using QGIS, GRASS, and PDAL.

# Installation Instructions
## Installing OSGEO Applications
1. Download the OSGeo4W network installer from the website: https://trac.osgeo.org/osgeo4w/
2. Run the installer.

![Install_1](https://user-images.githubusercontent.com/95769776/235305337-8d5b46a2-b346-4e1c-844b-d0390180e029.jpg)

3. Select **Advanced Install**, and click **Next**.
4. Select **Install from internet**, and click **Next**.
5. Select your installation directory. It is recommended that you select the default path, but this path will be referred to as {root directory}. Click **Next**.
6. Select the location where temporary installation files will be installed, and the name of the Start Menu folder to place program links. Click **Next**.
7. Select your internet connection settings, and click **Next**.
8. Select https://download.osgeo.org as the download site, and click **Next**.
9. On the select packages screen, ensure that **Curr** is selected (for current).

![Install_2](https://user-images.githubusercontent.com/95769776/235305684-27732a78-b0e9-47ec-a6f9-3635766d893f.jpg)

10. Do not deselct any default packages that are selected. To select a package, click the arrows under the header **New** until the most recent version appears:
    a. Under **Commandline_Utilities**, ensure the following packages are selected:
