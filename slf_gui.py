# tools libraries
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
# Path libraries
from pathlib import Path
# Data manipulation libraries
import pandas as pd
import pickle
# Visualization libraries
import matplotlib.pyplot as plt
# Importing the SLF Generator tool
from tools.slf_function_gui import SLF_function_gui


class Root(tk.Tk):
    # TODO, messagebox/warnings
    def __init__(self):

        # Initiating variables to be used later on
        self.data = None
        self.labelFrame = None
        self.MODES = None
        self.correlation_type_variable = None
        self.idr_bin_entry = None
        self.pfa_bin_entry = None
        self.n_realizations = None
        self.conversion_factor = None
        self.perform_grouping = None
        self.outputs = None

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

        # Default project name
        self.project_name = "Project1"

        title_label = tk.Label(self, text="EDP - DV Function\n Generator", font=f"{self.default_font} 12 bold")
        title_label.grid(row=0, column=0)

        # Initial window options for initiating the project or shutting down the software
        project_control = [("New Project", 39),
                           # ("Open Project", 36),
                           ("Quit", 67)]

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
        self.title("Storey Loss Function Generator")
        self.iconbitmap(self.ICON_PATH)
        self.geometry("250x250")
        self.base = tk.LabelFrame(self, text="", padx=50, pady=50)
        self.base.grid(row=0, column=0, padx=20, pady=20)

        # Default project name
        self.project_name = "Project 1"

        title_label = tk.Label(self, text="EDP - DV Function\n Generator", font=f"{self.default_font} 12 bold")
        title_label.grid(row=0, column=0)

        project_control = [("New Project", 39),
                           # ("Open Project", 36),
                           ("Quit", 67)]

        cnt = 1
        for control, ipadx in project_control:
            self.create_button(control, ipadx, cnt)
            cnt += 1

    def record_project_name(self, new_project, project):
        """
        Defines project name
        :param new_project: tkinter root
        :param project: tkinter entry
        :return: None
        """
        self.project_name = project.get()
        if self.project_name == "":
            self.project_name = "Project 1"
        new_project.destroy()
        self.destroy()
        self.start_project()

    def create_project_entry(self):
        """
        Creates entry box
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
        if text == "New Project":
            pady = 30

            def command_apply():
                new_project, project = self.create_project_entry()
                save_button = tk.Button(new_project, text="Start", command=lambda: self.record_project_name(new_project,
                                                                                                            project))
                save_button.grid(row=1, column=0, columnspan=2, ipadx=8)
                cancel_button = tk.Button(new_project, text="Cancel", command=new_project.destroy)
                cancel_button.grid(row=1, column=1, columnspan=2, padx=80, sticky=tk.E)

        elif text == "Open Project":
            pady = 5

            def command_apply():
                pass
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

        self.title("Storey Loss Function Generator")
        self.iconbitmap(self.ICON_PATH)

        # Create menu bar
        self.init_menu()

        # Create the main canvas for outputting
        canvas = tk.Canvas(self, width=800, height=800, background='white')
        canvas.grid(row=0, column=1)

        # Create the input frame
        self.frame = tk.Frame(self)
        self.frame.grid(row=0, column=0, sticky="n")

        # Show Input Title
        title_label = tk.Label(self.frame, text="EDP - DV Function Inputs", fg=self.color_grid[0],
                               font=f"{self.default_font} 14 bold")
        title_label.grid(row=0, column=0, columnspan=2, sticky="nw")

        user_input_project_name = self.project_name
        project_label = tk.Label(self.frame, text=user_input_project_name, font=f"{self.default_font} 12 bold")
        project_label.grid(row=1, column=0, columnspan=2, pady=10, sticky=tk.W)

        # Initiate a dictionary to store all necessary information for the SLF generator
        self.data = {}

        # Browse for Component Data
        self.labelFrame = tk.LabelFrame(self.frame, text="Open Component Data", fg="red")
        self.labelFrame.grid(row=2, column=0, columnspan=2, padx=20, pady=20)
        self.browse_button(self.labelFrame, "Component Data")

        # Browse for correlation tree
        self.labelFrame = tk.LabelFrame(self.frame, text="Open Correlation Tree", fg="red")
        self.labelFrame.grid(row=3, column=0, columnspan=2, padx=20, pady=20)
        self.browse_button(self.labelFrame, "Correlation Tree")

        # Correlation option selection as a radio button
        correlation_label = tk.Label(self.frame, text="Select Correlation Type", font=f"{self.default_font} 12 bold")
        correlation_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        self.MODES = [("Independent", "Independent"),
                      ("Correlated", "Correlated")]
        self.correlation_type_variable = tk.StringVar()
        self.correlation_type_variable.set("Independent")
        row_id = self.create_radio_button(5, self.correlation_type_variable)

        # Define EDP steps
        correlation_label = tk.Label(self.frame, text="Select EDP Bin", font=f"{self.default_font} 12 bold")
        correlation_label.grid(row=row_id, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))

        # Entry for IDR bin in %
        self.idr_bin_entry = self.create_entry(row_id + 1, "IDR bin", "%", 0.1)

        # Entry for PFA bin in g
        self.pfa_bin_entry = self.create_entry(row_id + 2, "PFA bin", "g", 0.25)

        # Number of realizations for Monte Carlo simulation
        monte_carlo_label = tk.Label(self.frame, text="Monte Carlo Simulations", font=f"{self.default_font} 12 bold")
        monte_carlo_label.grid(row=row_id + 3, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        self.n_realizations = self.create_entry(row_id + 4, "Number of simulations", "", 20)

        # Conversion factor
        conversion_label = tk.Label(self.frame, text="Conversion Factor", font=f"{self.default_font} 12 bold")
        conversion_label.grid(row=row_id + 5, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        self.conversion_factor = self.create_entry(row_id + 6, "Conversion factor", "", 1.0)

        # Radio button for performance grouping
        grouping_label = tk.Label(self.frame, text="Apply Performance Grouping", font=f"{self.default_font} 12 bold")
        grouping_label.grid(row=row_id + 7, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        self.MODES = [("Yes", 1),
                      ("No", 0)]
        self.perform_grouping = tk.IntVar()
        self.perform_grouping.set(0)
        row_id = self.create_radio_button(row_id + 8, self.perform_grouping)

        # Run button
        tk.Button(self.frame, text='Run', command=self.run_slf).grid(row=17, column=0, columnspan=2, sticky=tk.W,
                                                                     pady=(10, 0), padx=(20, 0), ipadx=94)
        # Store Outputs button
        tk.Button(self.frame, text='Store Outputs', command=self.store_results).grid(row=18, column=0, columnspan=2,
                                                                                     sticky=tk.W, pady=(10, 0),
                                                                                     padx=(20, 0), ipadx=60)
        # # Save file button
        # tk.Button(self.frame, text='Save', command=self.file_save).grid(row=19, column=0, columnspan=2, sticky=tk.W,
        #                                                                 pady=(10, 0), padx=(20, 0), ipadx=89)

        # Shut down the toolbox
        self.close_program()

    def file_save(self):
        files = [('All Files', '*.*'),
                 ('txt Files', '*.txt')]
        f = filedialog.asksaveasfile(mode="w", filetypes=files)

    def store_results(self):
        """
        Stores outputs and inputs into separate pickle files
        :return:
        """
        files = [('All Files', '*.*'),
                 ('Pickle Files', '*.pickle')]
        f = filedialog.asksaveasfile(mode="wb", initialfile=f"{self.project_name}_{self.correlation_type_variable.get()}",
                                     filetypes=files, defaultextension=".pickle")
        if f is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        else:
            pickle.dump(self.outputs, f)

    def run_slf(self):
        """
        Runs SLF backend function
        :return: None
        """
        project_name = self.project_name
        try:
            component_data = self.data["Component Data"]
            correlation_tree = self.data["Correlation Tree"]
            correlation_type = self.correlation_type_variable.get()
            do_grouping = True if self.perform_grouping.get() == 1 else False
            edp_bin = [float(self.idr_bin_entry.get()), float(self.pfa_bin_entry.get())]
            n_realizations = int(self.n_realizations.get())
            conversion_factor = float(self.conversion_factor.get())

            if any(x <= 0.0 for x in [n_realizations, conversion_factor] + edp_bin):
                messagebox.showwarning("EXCEPTION", "Input Must be Non-Negative, Non-Zero!")
            else:
                slf = SLF_function_gui(project_name, component_data, correlation_tree, edp_bin, correlation_type,
                                       n_realizations, conversion_factor, do_grouping, sflag=False)
                self.outputs = slf.master(sensitivityflag=False)
                # TODO, add sensitivity option on the GUI

        except:
            messagebox.showwarning("EXCEPTION", "Input Data is Missing!")

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
        Destroy program
        :return: None
        """
        tk.Button(self.frame, text='Close', command=self.main_SLF).grid(row=20, column=0, columnspan=2, sticky=tk.W,
                                                                        pady=(10, 0), padx=(20, 0), ipadx=86)

    def create_radio_button(self, cnt, variable):
        """
        Radio button to select correlation type
        :param: int                         Row of next widget
        :return: tkinter variable
        :return: int                        Row of next widget
        """
        for text, mode in self.MODES:
            radio_button = tk.Radiobutton(self.frame, text=text, variable=variable, value=mode)
            radio_button.grid(row=cnt, column=0, sticky=tk.W)
            cnt += 1
        return cnt


if __name__ == "__main__":
    root = Root()
    root.mainloop()
