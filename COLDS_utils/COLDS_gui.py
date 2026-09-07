# =====================================================================================================================
# Filename:     COLDS_gui.py
# Written by:   Keith Cusson                Date: Aug 2025
# Description:  This script contains the graphical interface for the Cusson Open-source LiDAR Depression Scanner
# License:      MIT License (c) 2025 Keith Cusson
# =====================================================================================================================
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt5.QtCore import (Qt,
                          pyqtSignal,
                          QAbstractTableModel,
                          QItemSelection,
                          QItemSelectionModel,
                          QModelIndex,
                          QObject,
                          QTimer)
from PyQt5 import QtCore
from PyQt5.QtGui import QCloseEvent, QColor, QContextMenuEvent, QFont, QPalette
from PyQt5.QtWidgets import *
from time import strftime
from typing import Any

""" --------------------------------------------------------------------------------------------------------------------
Style Settings
 - Change as desired
---------------------------------------------------------------------------------------------------------------------"""
# Create the dark mode palette
dark_mode: QPalette = QPalette()
dark_mode.setColor(QPalette.Window, QColor(53, 53, 53))
dark_mode.setColor(QPalette.WindowText, Qt.white)
dark_mode.setColor(QPalette.Base, QColor(25, 25, 25))
dark_mode.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
dark_mode.setColor(QPalette.ToolTipBase, Qt.black)
dark_mode.setColor(QPalette.ToolTipText, Qt.white)
dark_mode.setColor(QPalette.Text, Qt.white)
dark_mode.setColor(QPalette.Button, QColor(53, 53, 53))
dark_mode.setColor(QPalette.ButtonText, Qt.white)
dark_mode.setColor(QPalette.BrightText, Qt.red)
dark_mode.setColor(QPalette.Link, QColor(42, 130, 218))
dark_mode.setColor(QPalette.Highlight, QColor(42, 130, 218))
dark_mode.setColor(QPalette.HighlightedText, Qt.black)

# Create font styles for required elements.
header_font: QFont = QFont('Arial', 14)
header_font.setBold(True)

button_font: QFont = QFont('Arial', 12)
button_font.setWeight(57)

button_font_selected: QFont = QFont('Arial', 12)
button_font_selected.setWeight(63)
button_font_selected.setUnderline(True)

button_font_strike: QFont = QFont('Arial', 12)
button_font_strike.setStrikeOut(True)

table_data_font: QFont = QFont('Consolas', 11)
table_head_font: QFont = QFont('Arial', 11)
table_head_font.setWeight(63)

# Create the variables to be used to size the various elements of the application.
control_width = 225
# ----------------------------------------------------------------------------------------------------------------------


class ActionButtons(QWidget):
    """
    A widget containing the main group of action buttons for the application.
    """
    def __init__(
            self,
            parent: QMainWindow = None
    ):
        """
        Initializes the layout for the action button group, and establishes handles for the buttons themselves.

        :param parent: Parent window to the widget.
        :type parent:  PyQt5.QtWidgets.QMainWindow
        """
        super().__init__(parent=parent)
        # Create the layout for the widget
        layout_buttons: QVBoxLayout = QVBoxLayout()

        # Add an invisible visual element that aligns the action buttons with the table to the left
        space_button: QPushButton = QPushButton(" ",
                                                enabled=False,
                                                font=button_font,
                                                flat=True)

        # Create the individual buttons
        self.button_tile: QPushButton = QPushButton(" Merge/Split Point Clouds ",
                                                    enabled=False,
                                                    font=button_font,
                                                    checkable=True,
                                                    clicked=parent.click_tile)

        self.button_dem: QPushButton = QPushButton(" Generate DEM ",
                                                   enabled=False,
                                                   font=button_font,
                                                   checkable=True,
                                                   clicked=parent.click_dem)

        self.button_sinkholes: QPushButton = QPushButton(" Find Sinkholes ",
                                                         enabled=False,
                                                         font=button_font,
                                                         checkable=True,
                                                         clicked=parent.click_sinkholes)

        self.button_view: QPushButton = QPushButton(" View Results ",
                                                    enabled=False,
                                                    font=button_font,
                                                    clicked=parent.click_view)

        # Add the buttons to the layout
        layout_buttons.addWidget(space_button)
        layout_buttons.addWidget(self.button_tile)
        layout_buttons.addWidget(self.button_dem)
        layout_buttons.addWidget(self.button_sinkholes)
        layout_buttons.addWidget(self.button_view)

        # Align the buttons with the top of the widget, and add the layout to the widget.
        layout_buttons.setAlignment(Qt.AlignTop)
        self.setLayout(layout_buttons)
        self.setMaximumWidth(control_width)

    def enable_step_btn(
            self,
            proj_state: dict[str, int]
    ):
        """
        Method that activates a specific action button based on the project state provided.

        :param proj_state: Project state provided in a dictionary of str/int pairs.
        :type proj_state:  dict[str, int]

        :return:           None. Updates the enabled state of the appropriate buttons based on the project state.
        """
        state = proj_state['state']

        # Create a list of the buttons in the widget
        btn_list: list[QPushButton] = [self.button_tile,
                                       self.button_dem,
                                       self.button_sinkholes,
                                       self.button_view]

        # Disable all buttons and set their fonts
        for btn in btn_list:
            btn.setEnabled(False)
            btn.setFont(button_font)

        # For actions already completed, set the font to strike out
        if state > 1:
            for i in range(state - 1):
                btn_list[i].setFont(button_font_strike)

        # Enable the button for the current state.
        btn_list[state - 1].setEnabled(True)

    def disable_all(self):
        """
        Method that disables all buttons.

        :return: None, disables all buttons
        """
        self.button_tile.setEnabled(False)
        self.button_dem.setEnabled(False)
        self.button_sinkholes.setEnabled(False)
        self.button_view.setEnabled(False)


class DialogListSelector(QDialog):
    """
    A custom dialog box with a listbox selector.
    """
    def __init__(
            self,
            title: str,
            message_txt: str,
            vals: list[str],
            parent: QMainWindow = None
    ):
        """
        Initializes the List Selector Dialog window

        :param title:       Title to be displayed on the window
        :type title:        str

        :param message_txt: Message to be displayed to the user.
        :type message_txt:  str

        :param vals:        Values to be displayed in the listbox selector
        :type vals:         list[str]

        :param parent:      Modal parent window for the dialog box
        :type parent:       PyQt5.QtWidgets.QMainWindow
        """
        super().__init__(parent)

        # Set the window title.
        self.setWindowTitle(title)

        # Create the elements that make up the dialog box.
        lbl_message: QLabel = QLabel(message_txt)

        self.list_box: QListWidget = QListWidget(self)
        self.list_box.setSelectionMode(QAbstractItemView.MultiSelection)
        self.list_box.addItems(vals)

        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.button_box: QDialogButtonBox = QDialogButtonBox(buttons)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Set the layout for the window
        layout: QVBoxLayout = QVBoxLayout()
        layout.addWidget(lbl_message)
        layout.addWidget(self.list_box)
        layout.addWidget(self.button_box)
        self.setLayout(layout)


class DialogPandaTableSelector(QDialog):
    """
    A custom dialog box with a table view that enables multiple selections
    """
    def __init__(
            self,
            title: str,
            message_txt: str,
            data_model: QAbstractTableModel,
            selectable_columns: list[int] = None,
            parent: QMainWindow = None
    ):
        """
        Initializes the Pandas Table Selector Dialog Box.

        :param title:              Title to be displayed on the window.
        :type title:               str

        :param message_txt:        Message to be displayed to the user.
        :type message_txt:         str

        :param data_model:         Data model to be loaded into the Table View
        :type data_model:          PyQt5.QtCore.QAbstractTableModel

        :param selectable_columns: List of columns to be selectable in the table.
        :type selectable_columns:  list[int]

        :param parent:             Modal parent window for the dialog box
        :type parent:              PyQt5.QtWidgets.QMainWindow
        """
        # Call the super class
        super().__init__(parent)
        self.selectable_columns = selectable_columns

        # Set the window title.
        self.setWindowTitle(title)

        # Create a label to hold the dialog box message
        lbl_message: QLabel = QLabel(message_txt)

        # Create the table to display information in the window, and set the table's model.
        self.table: QTableView = QTableView()
        self.model: PandasTableModel = data_model
        self.table.setModel(self.model)
        self.table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.table.resizeColumnsToContents()

        # Set the selection mode for the dialog buttons
        custom_selection_model: QItemSelectionModel = SingleSelectPerColumn(self.model,
                                                                            selectable_columns=selectable_columns,
                                                                            parent=self)
        self.table.setSelectionModel(custom_selection_model)
        self.table.setSelectionMode(QTableView.ExtendedSelection)
        self.table.setSelectionBehavior(QTableView.SelectItems)

        # Hide the columns that will contain tooltip information
        for i in range(3):
            self.table.hideColumn(i)

        # Create the dialog box buttons, and connect their actions.
        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.button_box: QDialogButtonBox = QDialogButtonBox(buttons)
        self.button_box.accepted.connect(self.check_results)
        self.button_box.rejected.connect(self.reject)

        # Set the layout for the window
        layout: QVBoxLayout = QVBoxLayout()
        layout.addWidget(lbl_message)
        layout.addWidget(self.table)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def check_results(self):
        """
        This method ensures that the dialog box is only accepted if there is a selection in each selectable column.

        :return: None
        """
        # Check that there is a selection in all the selectable columns
        selected_set: set = {idx.column() for idx in self.table.selectedIndexes()}
        required_set: set = set(self.selectable_columns)

        if selected_set == required_set:
            self.accept()
        else:
            result = QMessageBox.warning(self,
                                         'Selection Required',
                                         'You must select both a Horizontal and Vertical Coordinate Reference System')


class DialogResolutionSelector(QDialog):
    """
    A custom dialog box that allows a user to select multiple resolutions for output DEMs
    """
    def __init__(
            self,
            parent: QMainWindow = None
    ):
        """
        Initializes the custom Resolution Selector Dialog window.

        :param parent:  Modal parent window for the dialog box.
        :type parent:   PyQt5.QtWidgets.QMainWindow
        """
        # Create the dialog box and set the window title
        super().__init__(parent=parent)
        self.setWindowTitle('DEM resolutions')

        # Create a label indicating to the user what to do
        lbl_message: QLabel = QLabel('Select the desired output raster resolutions')

        # Create the buttons for the possible resolutions
        self.res_buttons: list[QPushButton] = []
        res_btn_layout: QHBoxLayout = QHBoxLayout()
        for btn_text in ['10 cm', '25 cm', '50 cm', '100 cm']:
            self.res_buttons.append(QPushButton(
                btn_text,
                checkable=True,
                font=button_font,
                clicked=self.press_button
            ))
            self.res_buttons[-1].setFixedWidth(100)
            res_btn_layout.addWidget(self.res_buttons[-1])

        # Create the dialog box buttons, and connect their actions.
        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.button_box: QDialogButtonBox = QDialogButtonBox(buttons)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Disable the ok button until at least one resolution button has been selected
        self.button_box.button(QDialogButtonBox.Ok).setEnabled(False)

        # Set the layout for the window
        layout: QVBoxLayout = QVBoxLayout()
        layout.addWidget(lbl_message)
        layout.addLayout(res_btn_layout)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def press_button(self):
        """
        Function that changes the font of each button when it is pressed.

        :return: None, updates the button display on screen
        """
        # Set a flag that will enable the ok button if one resolution button has been selected.
        ok_enabled = False

        # Iterate through the buttons setting their fonts based on their selection state.
        for btn in self.res_buttons:
            if btn.isChecked():
                btn.setFont(button_font_selected)
                # If a resolution button has been selected, enable the ok button flag
                ok_enabled = True
            else:
                btn.setFont(button_font)

        self.button_box.button(QDialogButtonBox.Ok).setEnabled(True)


class DialogTreeWidget(QDialog):
    """
    A custom dialog box that displays a message and a hierarchical tree
    """
    def __init__(
            self,
            title: str,
            message: str,
            tree: QTreeWidget,
            parent: QMainWindow = None
    ):
        """
        Initializes the custom Tree Widget Dialog window.

        :param title:   Title to be displayed on the window.
        :type title:    str

        :param message: Message to be displayed in the window above the hierarchical tree.
        :type message:  str

        :param tree:    Tree view to be displayed in the dialog box.
        :type tree:     PyQt5.QtWidgets.QTreeWidget

        :param parent:  Modal parent window for the dialog box.
        :type parent:   PyQt5.QtWidgets.QMainWindow
        """
        # Create the dialog box and set the window title
        super().__init__(parent=parent)
        self.setWindowTitle(title)

        # Create the other elements that will be displayed in the dialog box
        lbl_message: QLabel = QLabel(message)
        lbl_message.setTextFormat(Qt.MarkdownText)

        # Create the dialog box buttons, and connect their actions.
        buttons = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.button_box: QDialogButtonBox = QDialogButtonBox(buttons)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Set the layout for the window
        layout: QVBoxLayout = QVBoxLayout()
        layout.addWidget(lbl_message)
        layout.addWidget(tree)
        layout.addWidget(self.button_box)
        self.setLayout(layout)
        self.setMinimumWidth(tree.width())


class MessageFrame(QWidget):
    """
    A widget containing the message area for the main application.
    """
    def __init__(
            self,
            parent: QMainWindow = None
    ):
        super().__init__(parent=parent)
        layout_messages: QVBoxLayout = QVBoxLayout()

        # # Create the label for the text box
        lbl_msg: QLabel = QLabel('Messages')
        lbl_msg.setFont(header_font)
        lbl_msg.setAlignment(Qt.AlignHCenter)
        layout_messages.addWidget(lbl_msg)

        # Create the message box display area
        self.msg_box: QTextBrowser = QTextBrowser(self)
        self.msg_box.setAcceptRichText(True)
        self.msg_box.setUndoRedoEnabled(True)
        layout_messages.addWidget(self.msg_box)

        # Add the layout to the message
        self.setLayout(layout_messages)
        self.setFixedHeight(250)

    def print_message(
            self,
            msg
    ):
        """
        This method prints a message to the message box with a time stamp.

        :param msg: Message to be printed
        :type msg:  Any

        :return:    No return, prints a message to the message box.
        """
        msg_time = strftime('%Y-%m-%d %H:%M:%S')
        self.msg_box.append(f'<i>[{msg_time}]</i> - {str(msg)}')

    def update_last_line(
            self,
            msg
    ):
        """
        This method changes the last line in the message box.

        :param msg: New message to be printed

        :return:    No return, prints a message to the message box.
        """
        # Create a cursor to move to the beginning of the last line
        self.msg_box.undo()
        self.print_message(msg)


class MenuBar(QMenuBar):
    """
    The menu bar for the main application.
    """
    def __init__(
            self,
            parent: QMainWindow = None
    ):
        """
        Initializes the menu bar for the application.

        :param parent: The parent window for the menu bar.
        :type parent:  PyQt5.QtWidgets.QMenuBar
        """
        super().__init__(parent=parent)

        """ ------------------------------------------------------------------------------------------------------------
        File Menu
        ------------------------------------------------------------------------------------------------------------ """
        # --- Actions ---
        act_about: QAction = QAction('About',
                                     self,
                                     triggered=self.about)

        self.act_style: QAction = QAction('Dark Mode',
                                          self,
                                          triggered=self.change_palette)

        self.recent_menu: QMenu = QMenu('Recent Files',
                                        self)

        self.act_exit: QAction = QAction('Exit',
                                         self,
                                         triggered=parent.close)

        # --- Menus ---
        file_menu: QMenu = self.addMenu('&File')

        file_menu.addActions([act_about,
                              self.act_style])

        file_menu.addSeparator()

        file_menu.addMenu(self.recent_menu)

        file_menu.addSeparator()

        file_menu.addAction(self.act_exit)

        """ ------------------------------------------------------------------------------------------------------------
        Project Menu
        -------------------------------------------------------------------------------------------------------------"""
        # --- Actions ---
        act_new: QAction = QAction('New Project',
                                   self,
                                   triggered=parent.new)
        act_open: QAction = QAction('Open Project',
                                    self,
                                    triggered=parent.open)

        # --- Menus ---
        proj_menu: QMenu = self.addMenu('&Project')

        proj_menu.addActions([act_new,
                              act_open])

        """ ------------------------------------------------------------------------------------------------------------
        Add Menu
        ------------------------------------------------------------------------------------------------------------ """
        # --- Actions ---
        self.act_aoi: QAction = QAction('Area of Interest',
                                        self,
                                        triggered=parent.open_aoi,
                                        enabled=False)
        self.act_cloud: QAction = QAction('Point Cloud',
                                          self,
                                          triggered=parent.open_cloud,
                                          enabled=False)
        self.act_water: QAction = QAction('Water Features',
                                          self,
                                          triggered=parent.open_water,
                                          enabled=False)

        # --- Menus ---
        add_menu: QMenu = self.addMenu('&Add')

        add_menu.addActions([self.act_aoi,
                             self.act_cloud,
                             self.act_water])

        """ ------------------------------------------------------------------------------------------------------------
        Settings Menu
        ------------------------------------------------------------------------------------------------------------ """
        # --- Actions ---
        mp_1: QAction = QAction('1',
                                self,
                                checkable=True,
                                checked=False)
        mp_2: QAction = QAction('2',
                                self,
                                checkable=True,
                                checked=False)
        mp_4: QAction = QAction('4',
                                self,
                                checkable=True,
                                checked=False)
        mp_6: QAction = QAction('6',
                                self,
                                checkable=True,
                                checked=False)
        mp_8: QAction = QAction('8',
                                self,
                                checkable=True,
                                checked=True)

        # --- Menus ---
        settings_menu: QMenu = self.addMenu('&Settings')
        mp_menu: QMenu = settings_menu.addMenu("Number of CPUs")

        # Create an action group that will make the cpu number selection exclusive
        cpu_group: QActionGroup = QActionGroup(mp_menu)
        mp_menu.addActions([mp_1,
                            mp_2,
                            mp_4,
                            mp_6,
                            mp_8])
        cpu_group.addAction(mp_1)
        cpu_group.addAction(mp_2)
        cpu_group.addAction(mp_4)
        cpu_group.addAction(mp_6)
        cpu_group.addAction(mp_8)

        cpu_group.setExclusive(True)
        cpu_group.triggered.connect(parent.set_cpus)

        if parent.mode == 'Dark Mode':
            self.change_palette()
        self.update_recent()

    def about(self):
        """
        This method displays details about the appliction.

        :return: None. Opens the "about" dialog box.
        """
        # Produce the formatted message to be displayed.
        about_msg = ('<b>Cusson Open-source LiDAR Depression Scanner (COLDS)</b><br>'
                     'Created by Keith Cusson<br>'
                     f'Version <b>{self.parent().__version__}</b><br>'
                     'MIT License &copy; 2025')

        # Create the dialog box and apply its settings
        msg_box: QMessageBox = QMessageBox()
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setStandardButtons(QMessageBox.Ok)

        # Apply the custom text to the window.
        msg_box.setWindowTitle('About')
        msg_box.setText(about_msg)

        # Run the dialog box
        msg_box.exec_()

    def change_palette(self):
        """
        Method that changes the color palette for the application.

        :return: None. Toggles light/dark mode.
        """
        # Get the direction for the change
        win: MainWindow = self.parent()
        win.mode = self.act_style.text()

        # Set the menu text to the opposite selection
        menu_text = list({'Light Mode', 'Dark Mode'} - {win.mode})[0]

        # Apply the changes to the menu and the palette
        self.act_style.setText(menu_text)
        win.app.setPalette(win._palette[win.mode])

    def toggle_add_menu(
            self,
            enabled: bool
    ):
        """
        This method enables or disables all items in the add menu.

        :param enabled: Flag to enable/disable menu items.
        :type enabled:  bool

        :return:        None. Enables/Disables add menu actions
        """
        self.act_aoi.setEnabled(enabled)
        self.act_water.setEnabled(enabled)
        self.act_cloud.setEnabled(enabled)

    def update_recent(
            self,
            filename: str = None
    ):
        """
        Updates the recent files submenu in the File menu. with the list of recent files.

        :param filename: Filename to be added tothe recent menu.
        :type filename:  str

        :return:         None. Updates the recent files sub-menu.
        """
        win: MainWindow = self.parent()
        # Update the recent_files list with the new filename if it exists:
        if filename is not None:
            if filename in win.recent_files:
                win.recent_files.pop(win.recent_files.index(filename))
            win.recent_files.insert(0, filename)

        # Clear the current files in the recent menu
        self.recent_menu.clear()

        # Iterate through the recent files, and add those actions to the recent menu.
        for file in win.recent_files[:3]:
            act_recent: QAction = QAction(file,
                                          self,
                                          triggered=win.open)
            self.recent_menu.addAction(act_recent)


class PandasTableModel(QAbstractTableModel):
    """
    This class provides a custom table model to display a Pandas DataFrame in a QTableView widget.
    """
    def __init__(
            self,
            df: pd.DataFrame
    ):
        super().__init__()
        self._data = df

    def rowCount(
            self,
            index: QModelIndex
    ) -> int:
        return self._data.shape[0]

    def columnCount(
            self,
            index: QModelIndex
    ) -> int:
        return  self._data.shape[1]

    def headerData(
            self,
            section: int,
            orientation: int,
            role: int
    ) -> str:
        """
        This method applies the headers to the table model rows and columns.

        :param section:     Index for the corresponding row/column.
        :type section:      int

        :param orientation: Indicator for row/column.
        :type orientation:  int

        :param role:        Indicator for intended use of the input data.
        :type role:         int

        :return:            str representing the title for the row/column
        """
        # Set the font for the header being displayed
        if role == Qt.FontRole:
            return table_head_font

        # Section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Vertical:
                return str(self._data.index[section])

    def update_data(
            self,
            new_df: pd.DataFrame
    ) -> None:
        """
        This method refreshes the view with data from a new dataframe.

        :param new_df: New Data to be placed into the table.
        :type new_df:  pandas.DataFrame

        :return:       None. Updates table model.
        """
        self.beginResetModel()
        self._data = new_df
        self.endResetModel()


class ProgressBar(QWidget):
    """
    A widget containing a labelled progress bar.
    """
    def __init__(
            self,
            parent: QMainWindow = None
    ):
        """
        Method that initializes the progress bar.

        :param parent: Parent window for the widget.
        :type parent:  PyQt5.QtWidgets.QMainWindow
        """
        super().__init__(parent=parent)
        # Create the layout for the widget
        prog_layout: QVBoxLayout = QVBoxLayout()
        prog_layout.setAlignment(Qt.AlignTop)

        # Create the elements that make up the
        self.desc: QLabel = QLabel(' ')  # Description of the purpose for the progress bar
        self.pbar: QProgressBar = QProgressBar(self)
        self.pbar.setFixedHeight(25)
        self.pbar.setTextVisible(True)

        self.reset_bar()

        prog_layout.addWidget(self.desc)
        prog_layout.addWidget(self.pbar)

        self.setLayout(prog_layout)

    def reset_bar(
            self,
            maximum: int = 100
    ):
        """
        Method that resets the progress bar widget, sets its maximum value, and changes its reporting format.

        :param maximum: Max value for the progress bar. Defaults to 100 for a percentage view.
        :type maximum:  int

        :return:        None, resets the progress bar to zero and updates its format and max value
        """
        # Set the bar back to 0, and set the indicated maximum value
        self.pbar.setValue(0)
        self.pbar.setMaximum(maximum)

        # Set the progress format based on the max value.
        if maximum == 100:
            self.pbar.setFormat('%p%')
        else:
            self.pbar.setFormat('%v / %m')

    def update_bar(
            self,
            value
    ):
        """
        Method that updates the value of the progress bar and ensures it can never be above the maximum value.

        :param value: Value to be applied.
        :type value:  int

        :return:      None, updates the progress bar value.
        """
        # Ensure that the value never goes above the maximum value
        value = min(value, self.pbar.maximum())
        self.pbar.setValue(value)


class ProgressBarDisplay(QWidget):
    """
    A widget containing multiple progress bars
    """
    def __init__(
            self,
            no_bars: int = 5,
            parent: QMainWindow = None
    ):
        """
        Initializes the widget with the number of desired progress bars.

        :param no_bars: Number of progress bars to be displayed. Default: 1
        :type no_bars:  int

        :param parent:  Parent window to the widget.
        :type parent:   QMainWindow
        """
        super().__init__(parent=parent)

        # Create the layout that will hold all elements.
        pbar_layout: QVBoxLayout = QVBoxLayout()

        # Store all progress bars in a list, allowing individual ones to be accessed by index.
        self.pbar_list: list[ProgressBar] = []

        # Add the elements to the layout, with spacers to group them in the centre, and hide them by default.
        pbar_layout.addStretch()
        for i in range(no_bars):
            self.pbar_list.append(ProgressBar())
            pbar_layout.addWidget(self.pbar_list[i])
            self.pbar_list[i].hide()
        pbar_layout.addStretch()

        # Set the widget layout
        self.setLayout(pbar_layout)

    def show_bars(
            self,
            no_bars: int = 1
    ):
        """
        Resets the widget layout with new bars.

        :param no_bars: Number of progress bars to be displayed. Default: 1
        :type no_bars:  int

        :return:        None, resets the currently displayed layout.
        """
        # Iterate through the current widgets, and hide them or show them based on their position
        for i in range(len(self.pbar_list)):
            self.pbar_list[i].setHidden(i >= no_bars)

    def update_bar(
            self,
            value: int,
            bar_no: int = 0
    ):
        """
        Updates the value of the progress bar indicated by bar_no.

        :param value:  Progress bar value to be set.
        :type value:   int

        :param bar_no: Bar number to be modified. Default: 0
        :type bar_no:  int

        :return:       None. Updates the bar indicated if it exists.
        """
        if bar_no < len(self.pbar_list):
            self.pbar_list[bar_no].update_bar(value)

    def reset_bar(
            self,
            maximum: int,
            bar_no: int = 0
    ):
        """
        Updates the maximum value of the progress bar indicated by bar_no

        :param maximum: Maximum value to be set.
        :type maximum:  int

        :param bar_no:  Bar number to be modified. Default: 0
        :type bar_no:   int

        :return:        None updates the maximum of the bar indicated if it exists.
        """
        if bar_no < len(self.pbar_list):
            self.pbar_list[bar_no].reset_bar(maximum)

    def update_desc(
            self,
            desc: str,
            bar_no: int = 0
    ):
        """
        Updates the description label text of the progress bar indicated by bar_no

        :param desc:   New text to be displayed.
        :type desc:    str

        :param bar_no: Bar number to be modified. Default: 0
        :type bar_no:  int

        :return:       None. Updates the text description of the bar indicated if it exists.
        """
        if bar_no < len(self.pbar_list):
            self.pbar_list[bar_no].desc.setText(desc)

    def set_perc(
            self,
            bar_no: int = 0
    ):
        """
        Updates the bar progress format of the progress bar indicated by bar_no

        :param bar_no: Bar number to be modified. Default: 0
        :type bar_no:  int

        :return:       None. Updates the format of the progress bar indicated if it exists.
        """
        if bar_no < len(self.pbar_list):
            self.pbar_list[bar_no].pbar.setFormat('%p%')


class ProjectData(object):
    """
    Class representing the project data for the COLDS application, and methods corresponding to its components.
    """
    # Project object constructor
    def __init__(self):
        """
        Initialize the class with default values
        """
        # Create the geospatial variables for the project
        self.gdf_aoi: gpd.GeoDataFrame = gpd.GeoDataFrame()           # GeoDataFrame for the project's Area of Interest
        self.gdf_water: gpd.GeoDataFrame = gpd.GeoDataFrame()         # GeoDataFrame for all project water features
        self.gdf_pc_md: gpd.GeoDataFrame = gpd.GeoDataFrame()         # Point cloud files metadata
        self.gdf_vec_md: gpd.GeoDataFrame = gpd.GeoDataFrame()        # Input vector layer files metadata
        self.gdf_depressions: gpd.GeoDataFrame = gpd.GeoDataFrame()   # GeoDataFrame of all detected depressions
        self.state: int = 0                                           # Overall project process step

    def update_proj_state(
            self,
            variables: dict[str, str]
    ):
        """
        Updates the project state and classification flag based on the custom variables stored in a qgs file.

        :param variables: QGIS Project instance containing updated information.
        :type variables:  qgis.core.QgsProject

        :return:          None. Updates the state and classified properties
        """

        if 'state' in variables.keys():
            self.state = int(variables['state'])

    def export_project_state(self) -> dict[str, int]:
        """
        Produces a dictionary of the project state and classification variables

        :return:  dict[str, int] with keys state and classification.
        """
        return {'state': self.state}

    def update_gdf_pc_md(
            self,
            gdf: gpd.GeoDataFrame
    ):
        """
        Replace the point cloud file metadata geodataframe with a given geodataframe.

        :param gdf: New point cloud file metadata geodataframe.
        :type gdf:  geopandas.GeoDataFrame

        :return:    None, updates ProjectData object in place
        """
        self.gdf_pc_md = gdf

    def update_gdf_depressions(
            self,
            gdf: gpd.GeoDataFrame
    ):
        """
        Updates the depressions geodataframes with a given geodataframe.

        :param gdf: New depressions geodataframe.
        :type gdf:  geopandas.GeoDataFrame

        :return:    None, updates ProjectData object in place.
        """
        self.gdf_depressions = gdf


class SingleSelectPerColumn(QItemSelectionModel):
    """
    A QItemSelectionModel to select only a single element per column in a Table View implementing a PandasTableModel.
    """
    def __init__(
            self,
            model: QAbstractTableModel,
            selectable_columns: list[int] = None,
            parent: QWidget = None
    ):
        """
        Initialization for the selection model

        :param model:              Table model on which to set up the selection model.
        :type model:               PyQt5.QtCore.QAbstractTableModel

        :param selectable_columns: Restrictive list of columns to allow selections. Defaults to none allowing all
                                   columns to have a selection.
        :type selectable_columns:  list[int]

        :param parent:             Parent widget on which to apply the selection model.
        :type parent:              PyQt5.QtWidgets.QWidget
        """
        super().__init__(model, parent)

        # If the user has listed columns to be selected, create a column filter to be used
        self.selectable_columns: np.ndarray = np.ones(model._data.shape[1], dtype=np.bool_)
        if selectable_columns is not None and len(self.selectable_columns) > 0:
            for i in selectable_columns:
                self.selectable_columns[i] = False
            self.selectable_columns = ~self.selectable_columns

    def select(
            self,
            index_or_selection: QModelIndex | QItemSelection,
            command: QItemSelectionModel
    ):
        """
        Method that modifies the default selection method to only allow a single selection per column of the table in
        the columns that allow selections.

        :param index_or_selection: The cell(s), row, or column clicked on by the user.
        :type index_or_selection:  PyQt5.QtCore.QModelIndex or PyQt5.QtCore.QItemSelection

        :param command:            The command triggered by the user click action.
        :type command:             PyQt5.QtCore.QItemSelectionModel

        :return:                   None, eventually calls the super.select method on modified selection and command.
        """
        # Identify the type of selection object
        if isinstance(index_or_selection, QModelIndex):
            new_indexes: list[QModelIndex] = [index_or_selection]
        else:
            new_indexes: list[QModelIndex] = index_or_selection.indexes()

        # Create an array representing the current selection state of the table, and one representing the new selection
        current_arr: np.ndarray = np.zeros(self.model()._data.shape,
                                          dtype=np.bool_)
        select_arr: np.ndarray = np.zeros(self.model()._data.shape,
                                          dtype=np.bool_)

        for idx in self.selectedIndexes():
            current_arr[idx.row(), idx.column()] = True

        for idx in new_indexes:
            if idx.row() >= 0 and idx.column() >= 0:
                select_arr[idx.row(), idx.column()] = True

        # Ensure that the user is not attempting to select a column or perform a shift select
        if command & QItemSelectionModel.Columns or command & QItemSelectionModel.Current:
            command = QItemSelectionModel.NoUpdate

        # If a user selects a row, ensure that only the elements within the allowed selection range are made
        if command & QItemSelectionModel.Rows and not np.all(self.selectable_columns):
            command = command & ~QItemSelectionModel.Rows
            rows: np.ndarray = np.any(select_arr, axis=1)
            select_arr[rows, :] = True

        # Filter incoming selections with the selectable columns
        select_arr[:, ~self.selectable_columns] = False

        # If there are no new elements to be updated, set the command to NoUpdate
        if len(np.where(select_arr)) == 0:
            command = QItemSelectionModel.NoUpdate

        # If the selection will only select a new element, modify it to clear and then select, and update the selection
        # model.
        if command == QItemSelectionModel.Select:
            # Determine which columns are not being updated, and update the selection array as appropriate
            unchanged: np.ndarray = np.all(~select_arr, axis=0)
            select_arr[:, unchanged] = current_arr[:, unchanged]

            # Update the command to clear and select elements
            command = QItemSelectionModel.ClearAndSelect

        if command == QItemSelectionModel.ClearAndSelect:
            # Create the selection element
            index_or_selection = QItemSelection()
            items = np.where(select_arr)
            for item in range(len(items[0])):
                element: QModelIndex = self.model().index(items[0][item],
                                                          items[1][item])
                index_or_selection.select(element, element)

        # Complete the selection
        super().select(index_or_selection, command)


class EpsgTableModel(PandasTableModel):
    """
    This class modifies the PandasTableModelClass to display data for selecting the project's EPSG codes.
    """
    def __init__(
            self,
            df: pd.DataFrame
    ):
        """
        This method initiates the class.

        :param df: The dataframe to be loaded into the TableView.
        :type df:  pandas.DataFrame
        """
        super().__init__(df)

    def data(
            self,
            index: QModelIndex,
            role: int
    ):
        """
        This method places data values in the cells of the table.

        :param index: Index of the cell to be filled.
        :type index:  PyQt5.QtCore.Qt.QModelIndex

        :param role:  Indicator for the intended use of the input data.
        :type role:   int

        :return:      Any data type to the cell value.
        """
        value = None
        if role == Qt.FontRole:
            # Set the font for the data being displayed
            value = table_data_font
        elif role == Qt.DisplayRole:
            # Format the value being displayed
            value = self._data.iloc[index.row(), index.column()]
            if isinstance(value, np.integer):
                value = f'{value:>18,}'
            elif isinstance(value, np.floating):
                value = f'{value:>14,.3f}'
        elif role == Qt.ToolTipRole and index.column() > 3:
            value = self._data.iloc[index.row(), index.column() - 4]

        return value


class StatsTableModel(PandasTableModel):
    """
    This class modifies the PandasTableModelClass to display data for the StatsTable widget.
    """
    def __init__(
            self,
            df: pd.DataFrame
    ):
        """
        This method initiates the class.

        :param df: The dataframe to be loaded into the TableView.
        :type df:  pandas.DataFrame
        """
        super().__init__(df)

    def data(
            self,
            index: QModelIndex,
            role: int
    ):
        """
        This method places data values in the cells of the table.

        :param index: Index of the cell to be filled.
        :type index:  PyQt5.QtCore.Qt.QModelIndex

        :param role:  Indicator for the intended use of the input data.
        :type role:   int

        :return:      Any data type to the cell value.
        """
        # Set the font for the data being displayed
        if role == Qt.FontRole:
            return table_data_font

        # Format the value being displayed
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            if isinstance(value, np.integer):
                value = f'{value:>18,}'
            elif isinstance(value, np.floating):
                value = f'{value:>14,.3f}'
            return value
        elif role == Qt.ToolTipRole and index.column() == 1:
            return self._data.iloc[index.row(), 0]


class StatsTable(QWidget):
    """
    This widget displays a table containing the statistics of loaded files with a combo box to select different types
    of files.
    """
    def __init__(
            self,
            parent: QMainWindow = None
    ):
        """
        This method initializes the widget containing the pandas table with metadata from loaded files.

        :param parent: Parent window for the widget.
        :type parent:  PyQt5.QtWidgets.QMainWindow
        """
        super().__init__(parent=parent)
        layout_table: QGridLayout = QGridLayout()

        # Create the cloud type selector combo box
        self.combo_data_type: QComboBox = QComboBox(editable=False,
                                                    font=button_font,
                                                    activated=parent.change_type)
        self.combo_data_type.setMaximumWidth(control_width)
        layout_table.addWidget(self.combo_data_type, 0, 0)

        # Create a button that allows a user to accept all inputs
        self.btn_accept: QPushButton = QPushButton(' Finalize Inputs ',
                                                   font=button_font,
                                                   visible=False,
                                                   clicked=parent.finalize_start)
        self.btn_accept.setMaximumWidth(control_width)
        layout_table.addWidget(self.btn_accept, 0, 2)

        # Add the elements to the combobox, and disable them
        self.combo_items = ['Input Cloud', 'Vector File', 'Tile Cloud']
        self.combo_data_type.addItems(self.combo_items)
        self.combo_data_type.setCurrentIndex(-1)

        for i in range(0, len(self.combo_items)):
            self.combo_data_type.model().item(i).setEnabled(False)

        # Create the table that will hold the pandas data selected by the combo box
        self.table: QTableView = QTableView()
        self.table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.model: StatsTableModel = StatsTableModel(pd.DataFrame())
        self.table.setModel(self.model)
        self.model.modelReset.connect(self.handle_model_reset)

        # Add the table to the layout.
        layout_table.addWidget(self.table, 1, 0, 1, 3)

        self.setLayout(layout_table)

    def contextMenuEvent(
            self,
            a0: QContextMenuEvent
    ):
        """
        Method that generates a menu when an element on the Stats table view is right-clicked. This menu offers the user
        the ability to remove the file clicked on.

        :param a0: Right-click event.
        :type a0:  PyQt5.QtGui.QContextMenuEvent

        :return:   None. Creates the menu at the location clicked, asks the user to confirm if a selection is made, and
                   triggers the file remove script if it is accepted by the user.
        """
        # Determine the position of the click event
        table_vert_pos = self.table.pos().y()
        table_head_pos = self.table.horizontalHeader().size().height()
        index: QModelIndex = self.table.rowAt(a0.pos().y() - table_vert_pos - table_head_pos)

        # If the context action does not intersect with an item, end the event
        if index not in self.model._data.index or self.combo_data_type.currentIndex() not in range(2):
            return

        # Create the menu for the removal action
        menu: QMenu = QMenu()
        remove_act: QAction = menu.addAction('Remove')
        result = menu.exec_(a0.globalPos())

        # If the remove action is not selected, end the event
        if result != remove_act:
            return
        # Retrieve the main window handle, as well as the data for the selected row
        win: MainWindow = self.parent().parent().parent()
        data: pd.Series = self.model._data.iloc[index, :]

        if self.combo_data_type.currentIndex() == 0:
            title = 'Remove Point Cloud'
            message = (f'Would you like to remove this point cloud from the project?<br>'
                       f'<b>File:</b><br>{data["filename"]}')
        elif self.combo_data_type.currentIndex() == 1:
            title = f'Remove {data["layer_type"]} Layer'
            message = (f'Would you like to remove this {data["layer_type"]} layer from the project?<br>'
                       f'<b>File:</b><br> {data["filename"]}<br>'
                       f'<b>Layer:</b><br> {data["name"]}')
        else:
            return

        reply: int = win.dlg_confirm(title,
                                     message)
        if reply == QMessageBox.Yes:
            win.remove_file(index)

    def enable_selection(
            self,
            index: int
    ) -> None:
        """
        This method enables an entry in the combo selector

        :param index: Entry to be enabled.
        :type index:  str

        :return:      No return, enables the entry
        """
        if index in range(len(self.combo_items)):
            self.combo_data_type.model().item(index).setEnabled(True)

    def handle_model_reset(self):
        """
        This method indicates the actions to be undertaken after the model has been reset.

        :return: None
        """
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setColumnHidden(0, True)


class WorkerSignals(QObject):
    """
    This class represents a custom emitter to be used for transmitting signals from a running multiprocess process

    :cvar finished:  Signal that indicates the process has finished.

    :cvar result:    Signal containing the results of the function being run

    :cvar no_bars:   Signal containing the number of progress bars to be displayed in the progress bar widget

    :cvar progress:  Signal containing the position of a progress bar to be updated, and the value to be set

    :cvar disp_perc: Signal containing the position of a progress bar whose display format is to be converted to percent

    :cvar pbar_size: Signal containing the position of a progress bar to be updated, and its max value to be set

    :cvar desc:      Signal containing the position of a progress bar to be updated, and the text to be displayed on top
                     of it.

    :cvar msg:       Signal containing a message to be displayed in the message box area

    :cvar error:     Signal containing an error code and error message to be displayed.
    """
    finished: pyqtSignal = pyqtSignal()
    result: pyqtSignal = pyqtSignal(object, ProjectData)
    no_bars: pyqtSignal = pyqtSignal(int)
    progress: pyqtSignal = pyqtSignal(int, int)
    disp_perc: pyqtSignal = pyqtSignal(int)
    pbar_size: pyqtSignal = pyqtSignal(int, int)
    desc: pyqtSignal = pyqtSignal(str, int)
    msg: pyqtSignal = pyqtSignal(str)
    error: pyqtSignal = pyqtSignal(int, str)

    def __call__(
            self,
            signal: dict[str, Any]
    ):
        """
        Calling the QObject with a signal from the processing queue will emit the appropriate signals.

        :param signal: Dictionary containing data needed for the signal to be emitted.
        :type signal:  dict[str, Any]

        :return:       None. Emits the appropriate signal.
        """
        for key, value in signal.items():
            if key not in self.__annotations__:
                continue
            else:
                if value is None:
                    continue
                elif isinstance(value, tuple):
                    self.__getattr__(key).emit(*value)
                else:
                    self.__getattr__(key).emit(value)

        if {'result', 'finished'}.intersection(signal.keys()):
            self.finished.emit()


class MainWindow(QMainWindow):
    """
    This class represents the window making up the main interface for the COLDS graphical interface.
    """
    __version__ = '2.0a'
    program_name = 'Cusson Open-source LiDAR Depression Scanner'

    def __init__(
            self,
            app: QApplication = None
    ):
        """
        Initialization of the main window for the COLDS GUI.

        :param app: QApplication for the window. Used to apply theme to all PyQt5 windows the same.
        :type app:  PyQt5.QtWidgets.QApplication
        """
        super().__init__(None)
        # Set variables that are accessible to subclass methods of the MainWindow implementation.
        self.app = app
        self.data: ProjectData = ProjectData()
        self.no_cpus = 8

        # Create the palettes for the various styles
        self._palette: dict[str, QPalette] = {'Light Mode': QPalette(),
                                              'Dark Mode': dark_mode}
        self.mode = 'Light Mode'
        self.recent_files = []

        # Check for configuration settings
        if Path('colds.cfg').exists():
            with open('colds.cfg') as f:
                config = f.readlines()
                self.mode = config[0].strip()
                self.recent_files = [file.strip() for file in config[1:]]

        # Set the Window title
        self.setWindowTitle(self.program_name)

        # Add the window's menu bar
        self.menu_bar: MenuBar = MenuBar(self)
        self.setMenuBar(self.menu_bar)

        # Create the widgets
        self.wgt_message: MessageFrame = MessageFrame()              # Displays system notices
        self.wgt_buttons: ActionButtons = ActionButtons(self)        # Displays function buttons
        self.wgt_table: StatsTable = StatsTable(self)                # Displays all the statistics for the project files
        self.wgt_progress: ProgressBarDisplay = ProgressBarDisplay(parent=self)
        self.stack: QStackedWidget = QStackedWidget(self)

        # Add the stats table and progress displays to the stacked widget
        self.stack.addWidget(self.wgt_table)
        self.stack.addWidget(self.wgt_progress)

        # Set the window layout
        layout_page: QGridLayout = QGridLayout()

        layout_page.addWidget(self.stack, 0, 0)
        layout_page.addWidget(self.wgt_message, 1, 0, 1, 2)
        layout_page.addWidget(self.wgt_buttons, 0, 1)

        # Create the container
        container: QWidget = QWidget()
        container.setLayout(layout_page)

        # Add the layout to the window, and prevent window resizing
        self.setCentralWidget(container)
        self.setMinimumSize(800, 600)

        # Set the window to be displayed on initialization
        self.show()

        # Create the elements that will be used to handle CPU blocking processes
        self.timer: QTimer = QTimer(self)
        self.worker_signals: WorkerSignals = WorkerSignals()

        # Connect the worker signals to the progress bar display and the message area
        self.timer.timeout.connect(self.update_progress)
        self.worker_signals.no_bars.connect(self.wgt_progress.show_bars)
        self.worker_signals.disp_perc.connect(self.wgt_progress.set_perc)
        self.worker_signals.pbar_size.connect(self.wgt_progress.reset_bar)
        self.worker_signals.progress.connect(self.wgt_progress.update_bar)
        self.worker_signals.desc.connect(self.wgt_progress.update_desc)
        self.worker_signals.msg.connect(self.wgt_message.print_message)

    """ ----------------------------------------------------------------------------------------------------------------
    SUPERSEDING METHODS
    """
    def closeEvent(
            self,
            a0: QCloseEvent
    ):
        """
        Method that provides functionality for a triggered close event.

        :param a0: The close event.
        :type a0:  PyQt5.QtGui.QCloseEvent

        :return:   None. Confirms if the user wants to close the window, and performs shutdown if yes is selected.
        """
        # Set the accept value
        accept: bool = True

        # Display a confirmation dialog box
        reply = self.dlg_confirm('Quit',
                                 'Are you should you would like to quit? All your progress will be saved.')

        # If the user confirms, save configuration to file, and accept. Otherwise reject the close signal
        if reply == QMessageBox.Yes:
            out = [self.mode]
            out.extend(self.recent_files[:3])
            out = [line + '\n' for line in out]
            with open('colds.cfg', 'w') as f:
                f.writelines(out)
            a0.accept()
        else:
            a0.ignore()

    """
    ACTIONS
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Project Menu
    def new(self):
        """
        This method provides all the dialog boxes required when creating a new project

        :return: No return, stores values as a dictionary in class property user_input.
        """
        # Get a name for the project
        proj_name, ok1 = QInputDialog.getText(self,
                                              'Project Name',
                                              'What is the project name?')

        # Open the folder selection dialog box
        if ok1:
            folder_path = QFileDialog.getExistingDirectory(self,
                                                           "Select a directory")
            if folder_path:
                self.create_project(proj_name,
                                    folder_path)

    def create_project(
            self,
            name: str,
            proj_path: str
    ):
        """
        Method to be superseded in application for handling project creation.

        :param name:      Name of new project
        :type name:       str

        :param proj_path: Path to new project
        :type proj_path:  str

        :return:          None
        """
        pass

    def open(self):
        """
        Method for a dialog box to open a project file.

        :return: None. Executes open_project on completion if a file is selected.
        """
        # Determine what menu action called the method.
        if self.sender().text() == 'Open Project':
            file_path, _ = QFileDialog.getOpenFileName(self,
                                                       "Open Project File",
                                                       "",
                                                       "QGIS Project Files (*.qgz)")
        else:
            file_path = self.sender().text()

        # If a file path has been identified, open the file
        if file_path:
            self.open_project(file_path)

    def open_project(
            self,
            filename: str
    ):
        """
        Method to be superseded in application for handling project creation.

        :param filename: Path to project
        :type filename:  str

        :return:         None
        """
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # Add Menu
    def open_aoi(self):
        """
        This method provides a file dialog to open a vector file to be added as a project Area Of Interest.

        :return: No return, but stores a result in the class property dictionary user_input.
        """
        file_path, _ = QFileDialog.getOpenFileNames(self,
                                                    "Add Area of Interest",
                                                    "",
                                                    "Vector Files (*.gpkg;*.shp)")

        if len(file_path) > 0:
            self.add_vector('AOI', file_path)

    def open_cloud(self):
        """
        This method provides a file dialog to open a las file to be added to the project.

        :return: No return, but stores a result in the class property dictionary user_input.
        """
        file_path, _ = QFileDialog.getOpenFileNames(self,
                                                    "Add Point Cloud",
                                                    "",
                                                    "LAS/LAZ Point Cloud (*.las;*.laz)")
        if len(file_path) > 0:
            self.add_cloud('Input Cloud',
                           file_path)

    def open_water(self):
        """
        This method provides a file dialog to open a vector file to be added to the project as Water Features.

        :return: No return, but stores a result in the
        """
        file_path, _ = QFileDialog.getOpenFileNames(self,
                                                    "Add Water Features",
                                                    "",
                                                    "Vector Files (*.gpkg;*.shp)")

        if len(file_path) > 0:
            self.add_vector('Water Feature', file_path)

    def add_cloud(
            self,
            cloud_type: str,
            filename: list[str]
    ):
        """
        Method to be superseded in main application to handle adding point clouds to the project

        :param cloud_type: Cloud type label for loaded files.
        :type cloud_type:  str

        :param filename:   Path(s) the point cloud(s) to be loaded.
        :type filename:    list[str]

        :return:           None
        """
        pass

    def add_vector(
            self,
            lyr_type: str,
            filename: list[str]
    ):
        """
        Method to be superseded in the main application to handle adding vector files to the project.

        :param lyr_type: Layer Type to be added.
        :type lyr_type:  str

        :param filename: Path(s) to the vector file(s) to be added.
        :type filename:  list[str]

        :return:         None
        """
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # Settings Menu
    def set_cpus(
            self,
            action: QAction
    ):
        """
        Method that sets the number of cpus to be used in multiprocessing pools.

        :param action: Action denoting the number of cpus that was sent.
        :type action:  PyQt5.QtWidgets.QAction

        :return:       None. Sets the no_cpus variable in place.
        """
        self.no_cpus = int(action.text())

    # ------------------------------------------------------------------------------------------------------------------
    # Table View
    def change_type(self):
        """
        Method to be superseded in the Main application to update the display table after toggling type combo selector.

        :return: None
        """
        pass

    def remove_file(
            self,
            index: int
    ):
        """
        Method to be superseded in the Main application to remove a specif file from the table view.

        :param index: Index of the item to be removed.
        :type index:  int

        :return:      None
        """
        pass

    def finalize_start(self):
        """
        Method to be superseded in the main application to prevent more additions to the input files.

        :return: None
        """
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # Action buttons
    def set_buttons(self):
        """
        Method to set project buttons based on the current project state.

        :return: None. Sets the action button state
        """
        self.wgt_buttons.enable_step_btn(self.data.export_project_state())

    def click_tile(self):
        """
        Method to be superseded in the main application to merge point clouds and split them into tiles.

        :return: None
        """
        pass

    def click_dem(self):
        """
        Method to be superseded in the main application to create a DEM from point cloud tiles.

        :return: None
        """
        pass

    def click_sinkholes(self):
        """
        Method to be superseded in the main application to detect sinkholes from available DEMs

        :return: None
        """
        pass

    def click_view(self):
        """
        Method to be superseded in the main application to display the results.

        :return: None
        """
        pass

    """ ----------------------------------------------------------------------------------------------------------------
    THREADED PROCESSING
    """
    def update_progress(self):
        # Check the queue, and see if there are any new results
        if self.progress_queue.empty():
            return
        signal: dict = self.progress_queue.get()
        self.worker_signals(signal)

    """ ----------------------------------------------------------------------------------------------------------------
    DIALOG BOXES
    """
    def dlg_confirm(
            self,
            title: str,
            message: str
    ) -> int:
        """
        This method generates a dialog box to confirm a particular action, and returns the user selection.

        :param title:   Title for the dialog box
        :type title:    str

        :param message: Text to be displayed in the dialog box.
        :type message:  str

        :return:        int representing user selection.
        """
        # Create the message box, and configure its settings.
        msg_box: QMessageBox = QMessageBox(self)
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)

        # Apply the custom text to the window.
        msg_box.setWindowTitle(title)
        msg_box.setText(message)

        msg_box.exec_()

        return msg_box.result()

    def dlg_error(
            self,
            title: str,
            message: str
    ):
        """
        Method that generates an error message box.

        :param title:   Title to be affixed to the message box.
        :type title:    str

        :param message: Message to be displayed in the message box.
        :type message:  str

        :return:        None. Generates a message box to display an error message.
        """
        result = QMessageBox.critical(self,
                                      f'{title} Error',
                                      message)

    def dlg_list(
            self,
            title: str,
            message: str,
            vals: list[str]
    ) -> list[str] | None:
        """
        This method generates a dialog box with a list of values, and returns the selected values.

        :param title:   Title for the dialog box
        :type title:    str

        :param message: Text to be displayed in the dialog box.
        :type message:  str

        :param vals:    List of values to be displayed in the dialog box.
        :type vals:     list[str]

        :return:        list[str] of the selected values or None
        """
        dlg: DialogListSelector = DialogListSelector(title,
                                                     message,
                                                     vals,
                                                     self)
        if dlg.exec():
            result = [val.text() for val in dlg.list_box.selectedItems()]
        else:
            result = None

        return result

    def dlg_epsg_table(
            self,
            title: str,
            message: str,
            data: pd.DataFrame
    ) -> list[str]:
        """
        This method generates a dialog box with a table allowing for the selection of elements from a pandas table,
        forcing the user to select one and only one cell from each of the EPSG columns. Returns the selected cells in a
        list of strings. The first three columns are used as the tool tips for the remaining columns, and are hidden in
        the dialog display. The DataFrame of data to be displayed must be organized as follows:

        +----+--------------+--------------+--------------+---------------+-----------+-----------------+--------------+
        |    | filename     | H_Desc       | V_Desc       | layer_type    | Name      | Horizontal EPSG | Vertical EPSG|
        |====|==============|==============|==============|===============|===========|=================|==============|
        |   0|/path/to/file1| UTM20N NAD83 |  CGVD2013    | Input Point   | file1.las | 2961            | 6647         |
        +----+--------------+--------------+--------------+---------------+-----------+-----------------+--------------+
        |   1|/path/to/file2| UTM20N WGS84 |  CGVD2013    | AOI           | AOI_Layer | 32620           | 6647         |
        +----+--------------+--------------+--------------+---------------+-----------+-----------------+--------------+

        :param title:   Title of the dialog box to be displayed.
        :type title:    str

        :param message: Message to be displayed in the dialog box.
        :type message:  str

        :param data:    DataFrame of EPSG data to be displayed.
        :return:
        """
        # Set up the table model for the table view in the dialog box.
        table_model: QAbstractTableModel = EpsgTableModel(data)

        # Set up the dialog box to retrieve the EPSG codes.
        dlg: DialogPandaTableSelector = DialogPandaTableSelector(title=title,
                                                                 message_txt=message,
                                                                 data_model=table_model,
                                                                 selectable_columns=[5, 6],
                                                                 parent=self)

        result = ['', '']
        if dlg.exec():
            result = [idx.data() for idx in dlg.table.selectedIndexes()]

        return result

    def dlg_input_tree(
            self,
            title: str,
            message: str,
            vec_data: pd.DataFrame,
            pc_data: pd.DataFrame,
    ):
        """
        Method that displays a dialog box with a tree view to display input files for confirmation.

        :param title:    Title for the dialog window.
        :type title:     str

        :param message:  Message to be displayed in the dialog window.
        :type message:   str

        :param vec_data: Dataframe of vector input files.
        :type vec_data:  pandas.DataFrame

        :param pc_data:  Dataframe of point cloud input files.
        :type pc_data:   pandas.DataFrame

        :return:         None. Creates the input file confirmation dialog.
        """
        # Create the tree widget
        tree_display: QTreeWidget = QTreeWidget(None)

        # Set up the Tree Widget
        tree_display.setColumnCount(1)
        tree_display.setHeaderLabels(['Input Files'])

        # Create the top level nodes
        aoi: QTreeWidgetItem = QTreeWidgetItem(['Area of Interest'])
        wf: QTreeWidgetItem = QTreeWidgetItem(['Water Features'])
        pc: QTreeWidgetItem = QTreeWidgetItem(['Input Point Clouds'])

        # Add area of interest layers to the tree if they exist.
        if 'AOI' in vec_data.values:
            df_aoi: pd.DataFrame = vec_data[vec_data['layer_type'] == 'AOI']

            # Retrieve the filenames of the loaded Area of Interest files
            file_list: np.ndarray = np.unique(df_aoi['filename'])
            for file in file_list:
                file_item: QTreeWidgetItem = QTreeWidgetItem([file])

                # Retrieve the layers associated with this file
                layer_list: np.ndarray = np.unique(df_aoi.loc[df_aoi['filename'] == file, "name"])
                for layer in layer_list:
                    file_item.addChild(QTreeWidgetItem([layer]))

                # Add the file item to the aoi top level node
                aoi.addChild(file_item)
        else:
            aoi.addChild(QTreeWidgetItem(['None selected']))

        # Add water features to the tree if they exist
        if 'Water Feature' in vec_data.values:
            df_wf: pd.DataFrame = vec_data[vec_data['layer_type'] == 'Water Feature']
            # Retrieve the filenames of the loaded Area of Interest files
            file_list: np.ndarray = np.unique(df_wf['filename'])
            for file in file_list:
                file_item: QTreeWidgetItem = QTreeWidgetItem(wf,
                                                             [file])
                # Retrive the layers associated with this file
                layer_list: np.ndarray = np.unique(df_wf.loc[df_wf['filename'] == file, "name"])
                for layer in layer_list:
                    file_item.addChild(QTreeWidgetItem([layer]))

                # Add the file item to the aoi top level node
                wf.addChild(file_item)
        else:
            wf.addChild(QTreeWidgetItem(['None selected']))

        for file in pc_data['filename']:
            pc.addChild(QTreeWidgetItem([file]))

        # Add the top level nodes to the tree
        tree_display.addTopLevelItems([aoi,
                                       wf,
                                       pc])

        # Set the final display options for the tree display
        tree_display.expandAll()
        header: QHeaderView = tree_display.header()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)

        # Create the dialog box
        dlg: DialogTreeWidget = DialogTreeWidget(title,
                                                 message,
                                                 tree_display,
                                                 self)

        # Run the dialog, and return its result
        if dlg.exec():
            result = True
        else:
            result = False

        return result

    def dlg_resolution(self) -> list[str]:
        """
        Method that creates a custom dialog box that allows the user to indicate what resolution DEM rasters they would
        like.

        :return: List of DEM resolutions.
        """
        # Create the dialog box
        res_box = DialogResolutionSelector(self)

        out_list = []

        # Check the result of the box
        if res_box.exec():
            out_list = [btn.text().replace(' ', '_') for btn in res_box.res_buttons if btn.isChecked()]

        return out_list

    def dlg_warning(
            self,
            title: str,
            message: str
    ):
        """
        Method that creates a warning message dialog box.

        :param title:   Title to be displayed on the window.
        :type title:    str

        :param message: Message to be displayed on the window.
        :type message:  str

        :return:        None. Displays a warning message dialog box.
        """
        # Create the message box
        msg_box = QMessageBox(self)

        # Set the icon and title
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle(f'{title} Warning')

        # Set the message text, and the text format.
        msg_box.setTextFormat(Qt.MarkdownText)
        msg_box.setText(message)

        # Add the dialog buttons
        msg_box.setStandardButtons(QMessageBox.Ok)

        # Display the dialog box
        msg_box.exec()
