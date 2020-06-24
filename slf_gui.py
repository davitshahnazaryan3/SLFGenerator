# tools libraries
import tkinter as tk
from tkinter import filedialog
# Path libraries
from pathlib import Path
# Data manipulation libraries
import pandas as pd
import numpy as np
# Visualization libraries
import matplotlib.pyplot as plt


class Root(tk.Tk):
    def __init__(self):
        # Initial directory
        self.DIR = Path.cwd()
        # Create root
        super(Root, self).__init__()
        # Default font style
        self.default_font = "Helvetica"
        self.option_add("*Font", self.default_font)

        self.title("Storey Loss Function Generator")
        self.iconbitmap("F:\Storey_loss_functions\gui\images\icon.ico")
        self.color_grid = ['#840d81', '#6c4ba6', '#407bc1', '#18b5d8', '#01e9f5',
                           '#cef19d', '#a6dba7', '#77bd98', '#398684', '#094869']

        # Create menu bar
        self.init_menu()

        # Create the main canvas for outputting
        canvas = tk.Canvas(self, width=800, height=700, background='white')
        canvas.grid(row=0, column=1)

        # Create the input frame
        self.frame = tk.Frame(self)
        self.frame.grid(row=0, column=0, sticky="n")

        # Show Input Title
        title_label = tk.Label(self.frame, text="Storey Loss Function Inputs", fg=self.color_grid[0],
                               font=f"{self.default_font} 14 bold")
        title_label.grid(row=0, column=0, columnspan=2, sticky="nw")

        # TODO, make into a user defined name through start
        # TODO, add 3 options for project, i.e. new, open, close
        # TODO, add option to close, save project
        user_input_project_name = "Project 1"
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
        self.correlation_type = tk.StringVar()
        self.correlation_type.set("Independent")
        row_id = self.correlation_radio(5, self.correlation_type)

        # Define EDP steps
        correlation_label = tk.Label(self.frame, text="Select EDP Bin", font=f"{self.default_font} 12 bold")
        correlation_label.grid(row=row_id, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))

        # Entry for IDR bin in %
        self.create_entry(row_id+1, "IDR bin", "%")

        # Entry for PFA bin in g
        self.create_entry(row_id+2, "PFA bin", "g")

        # Number of realizations for Monte Carlo simulation
        monte_carlo_label = tk.Label(self.frame, text="Monte Carlo Simulations", font=f"{self.default_font} 12 bold")
        monte_carlo_label.grid(row=row_id+3, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        self.create_entry(row_id+4, "Number of simulations", "")

        # Conversion factor
        monte_carlo_label = tk.Label(self.frame, text="Conversion Factor", font=f"{self.default_font} 12 bold")
        monte_carlo_label.grid(row=row_id+5, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        self.create_entry(row_id+6, "Conversion factor", "")

        # Radio button for performance grouping
        grouping_label = tk.Label(self.frame, text="Apply Performance Grouping", font=f"{self.default_font} 12 bold")
        grouping_label.grid(row=row_id+7, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        self.MODES = [("Yes", 1),
                      ("No", 0)]
        self.perform_grouping = tk.IntVar()
        self.perform_grouping.set(0)
        row_id = self.correlation_radio(row_id+8, self.perform_grouping)

        # Shut down the toolbox
        self.close_program()

    def create_entry(self, cnt, text, unit):
        entry = tk.Entry(self.frame, width=10)
        entry.grid(row=cnt, column=1)
        name_label = tk.Label(self.frame, text=text)
        name_label.grid(row=cnt, column=0, sticky=tk.W)
        name_label = tk.Label(self.frame, text=unit)
        name_label.grid(row=cnt, column=2, sticky=tk.E)

    def init_menu(self):
        """
        Initiates menu bar
        :return: None
        """
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        fileMenu = tk.Menu(menubar)
        # TODO, assign commands
        fileMenu.add_command(label="New Project...", font=f"{self.default_font} 10")
        fileMenu.add_command(label="Open Project...", font=f"{self.default_font} 10")
        fileMenu.add_command(label="Close Project...", font=f"{self.default_font} 10")
        fileMenu.add_command(label="Exit", command=self.onExit, font=f"{self.default_font} 10")
        menubar.add_cascade(label="File", menu=fileMenu)

    def onExit(self):
        """
        Shuts down program
        :return: None
        """
        self.quit()

    def browse_button(self, labelframe, name):
        """
        Create a button for browsing
        :param name: str                    Name of the key under which to store into the data dictionary
        :return: None
        """
        self.button = tk.Button(labelframe, text="Browse a file", command=lambda: self.fileDialog(labelframe, name),
                                width=20)
        self.button.grid(row=2, column=1)

    def fileDialog(self, labelframe, name):
        """
        Ask to open file and store file into the dictionary
        :param name: str                    Name of the key under which to store into the data dictionary
        :return: None
        """
        self.filename = filedialog.askopenfilename(initialdir=self.DIR, title="Select a File",
                                                   filetypes=(("csv files", f"*csv"), ("all files", "*.")))
        try: self.data[name] = pd.read_csv(self.filename)
        except: self.data[name] = None

        if self.data[name] is not None:
            labelframe.config(fg="green")

    def close_program(self):
        """
        Destroy program
        :return: None
        """
        tk.Button(self.frame, text='Close', command=self.destroy).grid(row=20, column=0, columnspan=2, sticky=tk.S,
                                                                       pady=(10, 0))

    def correlation_radio(self, cnt, variable):
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
