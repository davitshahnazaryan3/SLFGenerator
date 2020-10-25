"""
Runs the SLF generator
"""
# tools, libraries
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
# Path libraries
from pathlib import Path
# Data manipulation libraries
import pandas as pd
import pickle
import random
# Importing the SLF Generator tool
from tools.slf import SLF
# Data visualization
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure


class Root(tk.Tk):
    def __init__(self):
        # Initiating variables to be used later on
        self.data = None
        self.labelFrame = None
        self.MODES = None
        self.correlation_type_variable = None
        self.regression_variable = None
        self.idr_bin_entry = None
        self.pfa_bin_entry = None
        self.precision = None
        self.conf_level = None
        self.n_realizations = None
        self.conversion_factor = None
        self.replCost = None
        self.perform_grouping = None
        self.cache = None
        self.outputs = None
        self.canvas = None
        self.toolbarFrame = None
        self.figure_on = None
        self.ax = None
        self.fig = None
        self.project_name = None

        # Initial directory
        self.DIR = Path.cwd()
        # Color palettes for easy access
        self.color_grid = ['#840d81', '#6c4ba6', '#407bc1', '#18b5d8', '#01e9f5',
                           '#cef19d', '#a6dba7', '#77bd98', '#398684', '#094869']
        # Create root
        super(Root, self).__init__()
        # Default font style
        self.default_font = "Helvetica"
        self.option_add("*Font", self.default_font)

        # Defining title and icon of the toolbox
        self.title("Storey Loss Function Generator")
        self.ICON_PATH = self.DIR/"icon.ico"
        self.iconbitmap(self.ICON_PATH)

        # Geometry of the initial window
        self.geometry("250x250")
        self.base = tk.LabelFrame(self, text="", padx=50, pady=50)
        self.base.grid(row=0, column=0, padx=20, pady=20)

        # Title of the toolbox
        title_label = tk.Label(self, text="EDP - DV Function\n Generator", font=f"{self.default_font} 12 bold")
        title_label.grid(row=0, column=0)

        # Initial window options for initiating the project or shutting down the software
        project_control = [("New Project", 39),
                           # ("Open Project", 36),
                           ("Quit", 67)]

        # Button generator for project control
        cnt = 1
        for control, ipadx in project_control:
            self.create_button(control, ipadx, cnt)
            cnt += 1

    def main_SLF(self):
        """
        Creates the main window for input and output visualization
        :return: None
        """
        self.destroy()
        # Create root
        super(Root, self).__init__()
        self.title("Story Loss Function Generator")
        self.iconbitmap(self.ICON_PATH)
        self.geometry("250x250")
        self.base = tk.LabelFrame(self, text="", padx=50, pady=50)
        self.base.grid(row=0, column=0, padx=20, pady=20)

        # Title of toolbox
        title_label = tk.Label(self, text="EDP - DV Function\n Generator", font=f"{self.default_font} 12 bold")
        title_label.grid(row=0, column=0)

        project_control = [("New Project", 39),
                           # ("Open Project", 36),
                           ("Quit", 67)]

        cnt = 1
        for control, ipadx in project_control:
            self.create_button(control, ipadx, cnt)
            cnt += 1

    def create_entry(self, cnt, text, unit, default_value):
        """
        Creates an entry box
        :param cnt: int                             Row position on the root
        :param text: str                            Text to describe the variable name
        :param unit: str                            Unit of measure of the variable
        :param default_value: float                 Default value for the variable
        :return: tkinter entry
        """
        entry = tk.Entry(self.frame, width=10)
        entry.grid(row=cnt, column=1)
        entry.insert(0, f"{default_value}")
        name_label = tk.Label(self.frame, text=text)
        name_label.grid(row=cnt, column=0, sticky=tk.W)
        name_label = tk.Label(self.frame, text=unit)
        name_label.grid(row=cnt, column=2, sticky=tk.E)
        return entry

    def init_menu(self):
        """
        Initiates menu bar
        :return: None
        """
        menu_bar = tk.Menu(self)
        self.config(menu=menu_bar)

        def create_project():
            new_project, project = self.create_project_entry()
            save_button = tk.Button(new_project, text="Start", command=lambda: self.record_project_name(new_project,
                                                                                                        project))
            save_button.grid(row=1, column=0, columnspan=2, ipadx=8)
            cancel_button = tk.Button(new_project, text="Cancel", command=new_project.destroy)
            cancel_button.grid(row=1, column=1, columnspan=2, padx=80, sticky=tk.E)

        fileMenu = tk.Menu(menu_bar, tearoff=0)
        fileMenu.add_command(label="New Project...", font=f"{self.default_font} 10", command=create_project)
        # fileMenu.add_command(label="Open Project...", font=f"{self.default_font} 10")
        # fileMenu.add_command(label="Save Project...", font=f"{self.default_font} 10")
        fileMenu.add_command(label="Close Project", font=f"{self.default_font} 10", command=self.main_SLF)
        fileMenu.add_command(label="Quit", command=self.on_exit, font=f"{self.default_font} 10")
        menu_bar.add_cascade(label="File", menu=fileMenu)

    def on_exit(self):
        """
        Shuts down program
        :return: None
        """
        self.quit()

    def browse_button(self, labelframe, name):
        """
        Create a button for browsing
        :param labelframe: tkinter labelframe
        :param name: str                    Name of the key under which to store into the data dictionary
        :return: None
        """
        button = tk.Button(labelframe, text="Browse a file", command=lambda: self.file_dialog(labelframe, name),
                           width=20)
        button.grid(row=2, column=1)

    def file_dialog(self, labelframe, name):
        """
        Ask to open file and store file into the dictionary
        :param labelframe: tkinter labelframe
        :param name: str                    Name of the key under which to store into the data dictionary
        :return: None
        """
        filename = filedialog.askopenfilename(initialdir=self.DIR, title="Select a File",
                                              filetypes=(("csv files", f"*csv"), ("all files", "*.")))
        try:
            self.data[name] = pd.read_csv(filename)
        except:
            self.data[name] = None

        if self.data[name] is not None:
            labelframe.config(fg="green")

    def close_program(self):
        """
        Destroy project
        :return: None
        """
        tk.Button(self.frame, text='Close', command=self.main_SLF).grid(row=30, column=0, columnspan=2, sticky=tk.W,
                                                                        pady=(10, 0), padx=(20, 0), ipadx=86)

    def create_radio_button(self, cnt, variable):
        """
        Radio button creator
        :param: int                         Row of next widget
        :return: tkinter variable
        :return: int                        Row of next widget
        """
        for text, mode in self.MODES:
            radio_button = tk.Radiobutton(self.frame, text=text, variable=variable, value=mode)
            radio_button.grid(row=cnt, column=0, sticky=tk.W, padx=10)
            cnt += 1
        return cnt

    def record_project_name(self, new_project, project):
        """
        Defines project name
        :param new_project: tkinter root
        :param project: tkinter entry
        :return: None
        """
        # Get project name
        self.project_name = project.get()
        if self.project_name == "":
            # Default project name
            self.project_name = "Project 1"

        # Destroy the current project
        new_project.destroy()
        try:
            self.destroy()
        except:
            pass

        # Initiate new project
        self.start_project()

    def create_project_entry(self):
        """
        Creates entry box for project name
        :return: tkinter root, tkinter entry
        """
        new_project = tk.Tk()
        new_project.title("Storey Loss Function Generator")
        new_project.iconbitmap(self.ICON_PATH)
        project = tk.Entry(new_project, width=30)
        project.grid(row=0, column=1, padx=20, pady=10)
        project_label = tk.Label(new_project, text="Project Name: ")
        project_label.grid(row=0, column=0, pady=10)
        return new_project, project

    def create_button(self, text, ipadx, cnt):
        """
        Creates a button
        :param text: str                        Text of the button on the initial options screen
        :param ipadx: int                       Padding length in x direction
        :param cnt: int                         Row for packing the button
        :return: None
        """
        # Start new project
        if text == "New Project":
            pady = 30

            def command_apply():
                # Destroy old project if new project has been initiated, otherwise return to the old project
                self.destroy()
                new_project, project = self.create_project_entry()
                save_button = tk.Button(new_project, text="Start", command=lambda: self.record_project_name(new_project,
                                                                                                            project))
                save_button.grid(row=1, column=0, columnspan=2, ipadx=8)
                cancel_button = tk.Button(new_project, text="Cancel", command=new_project.destroy)
                cancel_button.grid(row=1, column=1, columnspan=2, padx=80, sticky=tk.E)

        # Placeholder
        elif text == "Open Project":
            pady = 5

            def command_apply():
                pass

        # Close project
        else:
            pady = 5

            def command_apply():
                self.destroy()

        button = tk.Button(self, text=text, command=command_apply, font=f"{self.default_font} 12")
        button.grid(row=cnt, column=0, padx=40, pady=(pady, 0), ipadx=ipadx, sticky=tk.W)

    def start_project(self):
        """
        Starts the project main menu for inputs and outputs
        :return: None
        """
        # Create root
        super(Root, self).__init__()
        # Default font style
        self.default_font = "Helvetica"
        self.option_add("*Font", self.default_font)

        # Toolbox title
        self.title("Storey Loss Function Generator")
        self.iconbitmap(self.ICON_PATH)

        # Create menu bar
        self.init_menu()

        # Create the main canvas for outputting
        self.canvas = tk.Frame(self, width=800, height=800)
        self.canvas.grid(row=0, column=1)

        # Create the input frame
        self.frame = tk.Frame(self)
        self.frame.grid(row=0, column=0, sticky="n")

        # Show Input Title
        title_label = tk.Label(self.frame, text="EDP - DV Function Inputs", fg=self.color_grid[0],
                               font=f"{self.default_font} 14 bold")
        title_label.grid(row=0, column=0, columnspan=2, sticky="nw")

        # Project name
        user_input_project_name = self.project_name
        project_label = tk.Label(self.frame, text=user_input_project_name, font=f"{self.default_font} 12 bold")
        project_label.grid(row=1, column=0, columnspan=2, pady=10, sticky=tk.W)

        # Initiate a dictionary to store all necessary information for the SLF generator
        self.data = {}

        # Browse for Component Data
        self.labelFrame = tk.LabelFrame(self.frame, text="Open Component Data", fg="red")
        self.labelFrame.grid(row=2, column=0, columnspan=2, padx=20, pady=(0, 5))
        self.browse_button(self.labelFrame, "Component Data")

        # Browse for correlation tree
        self.labelFrame = tk.LabelFrame(self.frame, text="Open Correlation Tree", fg="red")
        self.labelFrame.grid(row=3, column=0, columnspan=2, padx=20, pady=(0, 5))
        self.browse_button(self.labelFrame, "Correlation Tree")

        # Correlation option selection as a radio button
        correlation_label = tk.Label(self.frame, text="Select Correlation Type", font=f"{self.default_font} 12 bold")
        correlation_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(0, 0))
        self.MODES = [("Independent", "Independent"),
                      ("Correlated", "Correlated")]
        self.correlation_type_variable = tk.StringVar()
        self.correlation_type_variable.set("Independent")
        row_id = self.create_radio_button(5, self.correlation_type_variable)

        # Select regression function
        regression_label = tk.Label(self.frame, text="Select Regression function", font=f"{self.default_font} 12 bold")
        regression_label.grid(row=row_id+1, column=0, columnspan=2, sticky=tk.W)
        self.MODES = [("Weibull", "Weibull"),
                      ("Papadopoulos", "Papadopoulos")]
        self.regression_variable = tk.StringVar()
        self.regression_variable.set("Weibull")
        row_id = self.create_radio_button(row_id+2, self.regression_variable)

        # Define EDP steps
        edp_label = tk.Label(self.frame, text="Select EDP Bin", font=f"{self.default_font} 12 bold")
        edp_label.grid(row=row_id, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))

        # Entry for IDR bin in %
        self.idr_bin_entry = self.create_entry(row_id + 1, "IDR bin", "%", 0.1)

        # Entry for PFA bin in g
        self.pfa_bin_entry = self.create_entry(row_id + 2, "PFA bin", "g", 0.05)

        # Number of realizations for Monte Carlo simulation
        monte_carlo_label = tk.Label(self.frame, text="Monte Carlo Simulations", font=f"{self.default_font} 12 bold")
        monte_carlo_label.grid(row=row_id + 3, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        self.n_realizations = self.create_entry(row_id + 4, "Number of simulations", "", 20)

        # Conversion factor
        conversion_label = tk.Label(self.frame, text="Conversion Factor", font=f"{self.default_font} 12 bold")
        conversion_label.grid(row=row_id + 5, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        self.conversion_factor = self.create_entry(row_id + 6, "Conversion factor", "", 1.0)

        # Conversion factor
        replCost_label = tk.Label(self.frame, text="Replacement Cost", font=f"{self.default_font} 12 bold")
        replCost_label.grid(row=row_id + 7, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        self.replCost = self.create_entry(row_id + 8, "Replacement Cost", "", 0.0)

        # Radio button for performance grouping
        grouping_label = tk.Label(self.frame, text="Apply Performance Grouping", font=f"{self.default_font} 12 bold")
        grouping_label.grid(row=row_id + 9, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        self.MODES = [("Yes", 1),
                      ("No", 0)]
        self.perform_grouping = tk.IntVar()
        self.perform_grouping.set(0)
        row_id = self.create_radio_button(row_id + 10, self.perform_grouping)

        # Run button
        tk.Button(self.frame, text='Run', command=self.run_slf).grid(row=27, column=0, columnspan=2, sticky=tk.W,
                                                                     pady=(10, 0), padx=(20, 0), ipadx=94)

        # Store Outputs button
        tk.Button(self.frame, text='Store Outputs into a .pickle', command=self.store_results).\
            grid(row=28, column=0, columnspan=2, sticky=tk.W, pady=(10, 0), padx=(20, 0), ipadx=14)

        # Store Outputs button
        tk.Button(self.frame, text='Store Outputs into a .xlsx', command=self.store_into_xlsx).\
            grid(row=29, column=0, columnspan=2, sticky=tk.W, pady=(10, 0), padx=(20, 0), ipadx=22)

        # Shut down the toolbox
        self.close_program()

    def store_results(self):
        """
        Stores outputs and inputs into separate pickle files
        :return: None
        """
        files = [('All Files', '*.*'),
                 ('Pickle Files', '*.pickle')]
        f = filedialog.asksaveasfile(mode="wb", initialfile=f"{self.project_name}_{self.correlation_type_variable.get()}",
                                     filetypes=files, defaultextension=".pickle")
        if f is None:   # asksaveasfile return `None` if dialog closed with "cancel".
            return
        else:
            pickle.dump(self.cache, f)

    def store_into_xlsx(self):
        """
        Stores outputs into a .xlsx format
        :return: None
        """
        files = [('All Files', '*.*'),
                 ('Excel Files', '*.xlsx')]
        f = filedialog.asksaveasfile(mode='wb', initialfile=f"{self.project_name}_{self.correlation_type_variable.get()}",
                                     filetypes=files, defaultextension=".xlsx")

        if f is None:   # asksaveasfile return `None` if dialog closed with "cancel".
            return
        else:
            # Get file name specified by the user
            fileName = f.name
            # Writing into an Excel
            writer = pd.ExcelWriter(fileName, engine="xlsxwriter")

            # Start process
            for key in self.outputs.keys():
                # Write EDP-DV functions, fitting parameters of the performed regression
                if key != "SLFs":
                    for k in self.outputs[key]:
                        if k != "accuracy":
                            # Transform into a DataFrame
                            df = pd.DataFrame.from_dict(self.outputs[key][k])
                            # Assign a sheet name
                            sheet_name = f"{key}_{k}"
                            # Write into a worksheet
                            df.to_excel(writer, sheet_name=sheet_name)

                # Store the EDP-DVs normalized by the total story loss
                else:
                    # Initialize startcol
                    cnt = 0
                    for k in self.outputs[key]:
                        # Transform into a DataFrame
                        df = pd.DataFrame(self.outputs[key][k], columns=[k])
                        # Assign a sheet name
                        sheet_name = "Mean EDP-DV Normalized"
                        # Write into a worksheet
                        df.to_excel(writer, sheet_name=sheet_name, startcol=cnt)
                        # Increment startcol
                        cnt += 3

            # Close the Pandas Excel writer and output the Excel file
            writer.save()

    def run_slf(self):
        """
        Runs SLF backend function
        :return: None
        """
        project_name = self.project_name
        try:
            # Reads component data file
            component_data = self.data["Component Data"].copy()
            # Reads correlation tree data file
            try:
                correlation_tree = self.data["Correlation Tree"].copy()
            except:
                correlation_tree = None
            # Gets correlation type for analysis defined by the user
            correlation_type = self.correlation_type_variable.get()
            # Gets regression function type for analysis defined by the user
            regression_type = self.regression_variable.get()
            # Performance grouping if specified
            do_grouping = True if self.perform_grouping.get() == 1 else False
            # EDP bin definition
            edp_bin = [float(self.idr_bin_entry.get()), float(self.pfa_bin_entry.get())]
            # Number of realizations for Monte Carlo analysis
            n_realizations = int(self.n_realizations.get())
            # Conversion factor for costs, costs_provided * conversion_factor
            conversion_factor = float(self.conversion_factor.get())
            # Replacement cost of the building, used for normalization
            replCost = float(self.replCost.get())

            # Verify integrity of inputs
            if any(x <= 0.0 for x in [n_realizations, conversion_factor] + edp_bin):
                messagebox.showwarning("EXCEPTION", "Input Must be Non-Negative, Non-Zero!")
            else:
                pass
                """ Runs SLF generator """
                slf = SLF(project_name, component_data, correlation_tree, edp_bin, correlation_type, regression_type,
                          n_realizations, conversion_factor, replCost, do_grouping)
                # Obtains the outputs
                self.outputs, self.cache = slf.master()

        except:
            # Show warning if input data is missing
            messagebox.showwarning("EXCEPTION", "Input Data is Missing!")

        # Visualize figures on the canvas
        self.visualize()

    def visualize(self):
        """
        Visualization module
        :return: None
        """
        # All EDP types considered
        edps = [key for key in self.cache.keys() if key != "SLFs"]

        elements = {}
        # Parse for each EDP type
        for edp in edps:
            elements[edp] = {}
            # Parse for each element
            for item in self.cache[edp]["component"].index:
                elements[edp][item] = self.cache[edp]["component"].loc[item]["EDP"] + " " + \
                                      self.cache[edp]["component"].loc[item]["Component"]

        # DropDown list options
        options = ["Fragility", "EDP-DV"]

        # Show plots
        def show_plot(*args):
            global item_default, trace_curve, plotShow, edp_default_old
            # Update button
            button_back = tk.Button(self.canvas, text="<<", command=lambda: back(-1))
            button_forward = tk.Button(self.canvas, text=">>", command=lambda: forward(1))
            button_back.grid(row=1, column=0)
            button_forward.grid(row=1, column=3)

            # Visualize first default graph
            if curve.get() == "Fragility":

                # Remove past figure if it exists
                if plotShow is not None:
                    plotShow.get_tk_widget().destroy()

                # Trace curve type
                if trace_curve == "EDP-DV":
                    item_default = 1
                trace_curve = "Fragility"

                # Get group case for visualization
                # Assign the default case if unassigned
                try:
                    edp_default = str(curve_group.get())
                except:
                    edp_default = str(edps[0])

                if edp_default != edp_default_old:
                    item_default = 1

                # Select fragility plot of interest and plot
                self.fig, self.ax = fragility_plots[edp_default][item_default]
                if self.toolbarFrame is not None:
                    self.toolbarFrame.grid_forget()
                plotShow = FigureCanvasTkAgg(self.fig, self.canvas)
                plotShow.get_tk_widget().grid(row=2, column=0, columnspan=4, sticky=tk.W)
                plotShow.draw()

                self.toolbarFrame = tk.Frame(master=self.canvas)
                self.toolbarFrame.grid(row=3, column=0, columnspan=4, sticky=tk.W)
                toolbar = NavigationToolbar2Tk(plotShow, self.toolbarFrame)

                # Name of selected plot type and figure number
                status = tk.Label(self.canvas, text=f"EDP: {edp_default}; "
                                                    f"Item {item_default} of {len(fragility_plots[edp_default])}",
                                  bd=1, relief=tk.SUNKEN, anchor=tk.E)
                status.grid(row=4, column=0, columnspan=4, sticky=tk.W + tk.E)

                # Forward button
                if item_default == len(fragility_plots[edp_default]):
                    button_forward = tk.Button(self.canvas, text=">>", state=tk.DISABLED)
                    button_forward.grid(row=1, column=3)

                # Back button
                if item_default == 1:
                    button_back = tk.Button(self.canvas, text="<<", state=tk.DISABLED)
                    button_back.grid(row=1, column=0)

                edp_default_old = edp_default

            elif curve.get() == "EDP-DV":

                # Remove past figure if it exists
                if plotShow is not None:
                    plotShow.get_tk_widget().destroy()

                if trace_curve == "Fragility":
                    item_default = 1
                trace_curve = "EDP-DV"

                # Get group case for visualization
                # Assign the default case if unassigned
                try:
                    edp_default = curve_group.get()
                except:
                    edp_default = edps[0]

                self.fig, self.ax = edp_dv_plot[edp_default]
                if self.toolbarFrame is not None:
                    self.toolbarFrame.grid_forget()
                plotShow = FigureCanvasTkAgg(self.fig, self.canvas)
                plotShow.get_tk_widget().grid(row=2, column=0, columnspan=4, sticky=tk.W)
                plotShow.draw()

                self.toolbarFrame = tk.Frame(master=self.canvas)
                self.toolbarFrame.grid(row=3, column=0, columnspan=4, sticky=tk.W)
                toolbar = NavigationToolbar2Tk(plotShow, self.toolbarFrame)

                status = tk.Label(self.canvas, text=f"EDP Performance group: {edp_default}", bd=1,
                                  relief=tk.SUNKEN, anchor=tk.E)
                status.grid(row=4, column=0, columnspan=4, sticky=tk.W + tk.E)

                # Make back and forward buttons disabled
                button_forward = tk.Button(self.canvas, text=">>", state=tk.DISABLED)
                button_forward.grid(row=1, column=3)

                button_back = tk.Button(self.canvas, text="<<", state=tk.DISABLED)
                button_back.grid(row=1, column=0)

        # Plotting option
        label = tk.Label(self.canvas, text="Select Plotting Option")
        label.grid(row=0, column=1, sticky=tk.N)

        # Group option
        label = tk.Label(self.canvas, text="Select Group")
        label.grid(row=0, column=2, sticky=tk.N)

        # Default EDP and item
        global item_default, trace_curve, plotShow, edp_default_old

        # Initialize plotShow
        plotShow = None

        # Default EDP to plot
        edp_default = edps[0]
        edp_default_old = edps[0]
        item_default = 1

        # Get all possible plots and store them
        viz = VIZ(self.cache)
        fragility_plots = {}
        # Scan through each EDP type
        for edp in edps:
            fragility_plots[edp] = {}
            # Scan for each component
            for item in self.cache[edp]["fragilities"]["ITEMs"]:
                fragility_plots[edp][item] = viz.fragility_plot(edp, item)

        # EDP vs DV plots for each EDP type
        edp_dv_plot = {}
        for edp in edps:
            edp_dv_plot[edp] = viz.edp_dv_plots(edp)

        # Tracing the drop down selection of plot type
        curve = tk.StringVar(self.canvas)
        trace_curve = "EDP-DV"
        curve.trace('w', show_plot)
        curve.set(options[1])

        # Tracing the drop down selection of group for plotting
        curve_group = tk.StringVar(self.canvas)
        trace_curve = "EDP-DV"
        curve_group.trace('w', show_plot)
        curve_group.set(edps[0])

        # DropDown menu for selecting the type of the figure
        dropdown = tk.OptionMenu(self.canvas, curve, *options)
        dropdown.grid(row=1, column=1, sticky=tk.N)

        # DropDown menu for selecting the group type
        dropdown_group = tk.OptionMenu(self.canvas, curve_group, *edps)
        dropdown_group.grid(row=1, column=2, sticky=tk.N)

        # Tracing ID of the plot
        status = tk.Label(self.canvas, text=f"EDP Performance group: {edp_default}", bd=1, relief=tk.SUNKEN,
                          anchor=tk.E)
        status.grid(row=4, column=0, columnspan=4, sticky=tk.W+tk.E)

        # Back and Forward buttons
        def back(change_by):
            global item_default
            item_default = item_default + change_by
            show_plot()

        def forward(change_by):
            global item_default
            item_default = item_default + change_by
            show_plot()

        # Creating the back and forward button widgets
        button_back = tk.Button(self.canvas, text="<<", command=lambda: back(-1), state=tk.DISABLED)
        button_forward = tk.Button(self.canvas, text=">>", command=lambda: forward(1), state=tk.DISABLED)
        button_back.grid(row=1, column=0)
        button_forward.grid(row=1, column=3)


class VIZ:
    def __init__(self, data):
        """
        Initializing visualizations
        :param data: dict                           Includes all SLF outputs
        """
        # Color palettes for easy access
        self.color_grid = ['#840d81', '#6c4ba6', '#407bc1', '#18b5d8', '#01e9f5',
                           '#cef19d', '#a6dba7', '#77bd98', '#398684', '#094869']
        self.data = data

    def fragility_plot(self, edp, item):
        """
        Plots item fragility curves
        :param edp: str                         EDP type
        :param item: int                        Item ID number
        :return: Figure                         Matplotlib figure
        """
        # Get EDP range
        edp_range = self.data[edp]["fragilities"]["EDP"]

        # Initialize figure
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        component = self.data[edp]["fragilities"]["ITEMs"][item]

        frag_pars = self.data[edp]["component"].iloc[item-1]
        cnt = 0
        for key in component.keys():
            # Get fragility parameters, mu=mean, beta=standard deviation
            mu = frag_pars[f"DS{cnt+1}, Median Demand"]
            beta = frag_pars[f"DS{cnt+1}, Total Dispersion (Beta)"]
            # Plotting
            if mu != 0.0:
                label = r"%s: $\mu=%.3f, \beta=%.2f$" % (key, mu, beta)
                ax.plot(edp_range, component[key], label=label)
            cnt += 1

        # Labeling
        edp_type = self.data[edp]["component"]["EDP"].iloc[0]
        if edp_type in ["IDR", "IDR NS", "IDR S"]:
            xlabel = edp_type[0:3]
            xlim = [0, 0.1]
        else:
            xlabel = edp_type[0:3] + " [g]"
            xlim = [0, 4.0]
        ylim = [0, 1]

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Probability of exceeding DS')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.grid(True, which="major", axis="both", ls="--", lw=1.0)
        ax.legend(frameon=False, loc='best', fontsize=10)
        fig.tight_layout()

        return fig, ax

    def edp_dv_plots(self, edp, n_to_plot=50):
        """
        Plots EDP DV functions in terms of provided currency
        :param edp: str                         EDP type
        :param n_to_plot: int                   Number of simulation scatters to plot
        :return: Figure                         Matplotlib Figure
        """
        # Get EDP range
        edp_range = self.data[edp]["fragilities"]["EDP"]
        # Transform into % unless it is a PFA
        if max(edp_range) < 1.0:
            edp_range = edp_range * 100

        # Create labels for the type of regression function selected
        fit_pars = self.data[edp]["fit_pars"]["mean"]["popt"]
        total_replacement_cost = self.data[edp]["total_replacement_cost"]
        regression = self.data[edp]["regression"]
        if regression == "Weibull":
            regressionLabel = r"$y=MaxCost\alpha\left[1-exp\left(-\left(\frac{x}{\beta}\right)^\gamma\right)\right]$"
            fitParLabel = r"$MaxCost=%.0f, \alpha=%.2f, \beta=%.2f, \gamma=%.2f$" \
                          % (total_replacement_cost, fit_pars[0], fit_pars[1], fit_pars[2])
            errorLabel = r"$error_{max}=%.0f$" % (self.data[edp]["accuracy"]) + "%"
            regressionLabel = regressionLabel + "\n" + fitParLabel + "\n" + errorLabel

        elif regression == "Papadopoulos":
            regressionLabel = r"$y=MaxCost\left[\epsilon\frac{x^\alpha}{\beta^\alpha + x^\alpha} + " \
                              r"(1-\epsilon)\frac{x^\gamma}{\delta^\gamma + x^\gamma}\right]$"
            fitParLabel = r"$MaxCost=%.0f, \alpha=%.2f, \beta=%.2f, \gamma=%.2f, \delta=%.2f, \epsilon=%.2f$" \
                          % (total_replacement_cost, fit_pars[0], fit_pars[1], fit_pars[2], fit_pars[3], fit_pars[4])
            errorLabel = r"$error_{max}=%.0f$" % (self.data[edp]["accuracy"]) + "%"
            regressionLabel = regressionLabel + "\n" + fitParLabel + "\n" + errorLabel

        # Initialize figure
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        cnt = 0
        component = self.data[edp]
        for key in component["edp_dv_euro"].keys():
            y_fit = component["edp_dv_euro"][key] / 10.0**3
            y = component["losses"]["loss_curve"].loc[key] / 10.0**3
            if key == "mean":
                label = key
            else:
                label = f"{int(key*100)}%"
            ax.plot(edp_range, y, color=self.color_grid[cnt], label=label, alpha=0.5, marker='o', markersize=3)
            ax.plot(edp_range, y_fit, color=self.color_grid[cnt], label=label)
            cnt += 2

        # Plotting the scatters of the Monte Carlo simulations
        total_loss_storey = component["total_loss_storey"]
        # Generate a selection of random indices for plotting
        if len(total_loss_storey) > n_to_plot:
            selection = random.sample(range(len(total_loss_storey)), n_to_plot)
            loss_to_display = {}
            for sel in selection:
                loss_to_display[sel] = total_loss_storey[sel]
        else:
            loss_to_display = {}
            for sel in total_loss_storey.keys():
                loss_to_display[sel] = total_loss_storey[sel]

        for key in loss_to_display.keys():
            y_scatter = loss_to_display[key] / 10.0**3
            ax.scatter(edp_range, y_scatter, edgecolors=self.color_grid[2], marker='o', s=3, facecolors='none',
                       alpha=0.5)
        ax.scatter(edp_range, y_scatter, edgecolors=self.color_grid[2], marker='o', s=3, facecolors='none',
                   alpha=0.5, label="Simulations")

        # Labeling
        edp_type = self.data[edp]["component"]["EDP"].iloc[0]
        if edp_type in ["IDR", "IDR NS", "IDR S"]:
            xlabel = edp_type[0:3] + " [%]"
            xlim = [0, 5.0]
        else:
            xlabel = edp_type[0:3] + " [g]"
            xlim = [0, 4.0]
        ylim = [0, max(y_fit) + 50.0]

        # Annotating
        fig.text(0.1, 0.95, regressionLabel, horizontalalignment='left', verticalalignment="top")

        # Labeling
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Losses [$10^3 €/100 m^2$]")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, which="major", axis="both", ls="--", lw=1.0)
        ax.legend(frameon=False, loc='center left', fontsize=10, bbox_to_anchor=(1, 0.5))
        fig.tight_layout()

        return fig, ax


if __name__ == "__main__":
    root = Root()
    root.mainloop()
