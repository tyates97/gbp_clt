# The Central Limit Theorem in Belief Propagation
This code is designed to test the effects of the Central Limit Theorem in factor graphs implementing belief propagation for different graph topologies.


## Installation
All required libraries/packages are held in 'requirements.txt', so to get started, clone the repo and navigate to the GBP_CLT file in your terminal and run:
```bash
pip install -r requirements.txt
```
## Usage
The code is intended to run on a local web browser using [streamlit](https://streamlit.io/).

To run the main programme on Middlebury data, you will need to run the following line in your terminal from the directory:
```bash
streamlit run src/stereo_test_app.py
```

To run the main programme on synthetic data, you will need to run the following line in your terminal from the directory:
```bash
streamlit run src/synthetic_data_app.py
```

## Script descriptions
All scripts are contained in the /src directory. A brief description of the purpose of each is contained below:
- stereo_test_app = opens a local browser for analysis of the Middlebury dataset via streamlit
- synthetic_data_app = the script that opens a local browser for analysis of synthetic data via streamlit
- belief_propagation = a script containing all functions related to performing belief propagation on a factor graph
- distribution_management = a script containing all functions to create or manipulate probability distributions
- factor_graph = a script containing all functions and classes related to defining the factor graph
- graphics = a script containing all functions to create the graphics
- config = file containing all global variables
- image_processing = file containing all methods required for calculating costs and probabilities from an image
- optimisation = file containing all the functions related to fitting Gaussian curves to a distribution


## Instructions to run the script
For middlebury data, you can select the different parameters from the left hand pane. Once selected, click the "Update Parameters" button at the bottom of the pane. This will load, and once loaded, scrolling to the bottom of the page and clicking "Run Belief Propagation" will run the algorithm on the constructed factor graph, and the results will be shown underneath when finished.

For synthetic data, you can select the different parameters from the left hand pane, and the graph and visualisations will automatically update.


## License

[MIT](https://choosealicense.com/licenses/mit/)