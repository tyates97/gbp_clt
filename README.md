# The Central Limit Theorem in Belief Propagation
This code is designed to test the effects of the Central Limit Theorem in factor graphs implementing belief propagation for different graph topologies.


## Installation
All required libraries/packages are held in 'requirements.txt', so to get started, clone the repo and navigate to the GBP_CLT file in your terminal and run:
```bash
pip install -r requirements.txt
```
or if using conda, run:
```bash
conda install --yes --file requirements.txt.
```

## Usage
The code is intended to run on a local web browser using [streamlit](https://streamlit.io/).

To run the programme, you will need to run the following line in your terminal:
```bash
streamlit run src/stereo_test_app.py
```
## Script descriptions
All scripts are contained in the /src directory. A brief description of the purpose of each is contained below:
- app = the script that opens a web-browser via streamlit
- belief_propagation = a script containing all functions related to performing belief propagation on a factor graph
- distribution_management = a script containing all functions to create or manipulate probability distributions
- factor_graph = a script containing all functions and classes related to defining the factor graph
- graphics = a script containing all functions to create the graphics
- config = file containing all global variables


## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License

[MIT](https://choosealicense.com/licenses/mit/)
