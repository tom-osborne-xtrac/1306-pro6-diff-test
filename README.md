# Differential Oscillation Test - Data Analysis Script  

This script will plot the results of a Loaded Rig test. The script uses a number of functions to handle the data output and has the ability to produce three plots. An overview plot and 1Hz and 10Hz frequency plots. A series of configurable options can be used to adjust the output of the script.
  
# Configurable Options  
The configurable options can be found at the top of the script, below the imports.  

    config = {
        "save_plot": True,          # Boolean - Will save the plot as a jpeg file
        "show_IPTrq": True,         # Boolean - Add/Remove IP Torque plot from top graph (y0)
        "show_FreqPlots": False,    # Boolean - Turn off/on plotting the 1hz & 10hz frequency plots
        "test_type": "scen123"      # options: ["bedding", "scen123"] - Adjusts axes to suit different test types
    }
