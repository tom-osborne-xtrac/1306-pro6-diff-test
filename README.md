# Differential Oscillation Test - Data Analysis Script  

This script will plot the results of a Loaded Rig test. The script uses a number of functions to handle the data output and has the ability to produce three plots. An overview plot and 1Hz and 10Hz frequency plots. A series of configurable options can be used to adjust the output of the script.
  
# Configurable Options  
The configurable options can be found at the top of the script, below the imports.  

    config = {
        "save_plot": True,          # Boolean
        "show_IPTrq": True,         # Boolean
        "show_FreqPlots": False,    # Boolean
        "test_type": "scen123"      # options: ["bedding", "scen123"]
    }
