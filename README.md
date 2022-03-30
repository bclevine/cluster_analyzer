# Cluster Analyzer

A script to find the redshift and richness of large batches of catalogs from the [DECaLS survey](https://www.legacysurvey.org/dr9/description/) data server. The code is multithreaded and properly logged, with a fair bit of room for customization according to your use case.

## Dependencies 
The catalog downloader requires the following packages, which come with standard Conda distributions: *logging, os, urllib, argparse, multiprocessing, time, warnings, numpy, astropy* and *pandas*.

The downloader also requires the following pip-installable packages: 
- *[tqdm](https://github.com/tqdm/tqdm)* for a progress bar. 
- *[wrapt timeout decorator](https://pypi.org/project/wrapt-timeout-decorator/)* to handle timeouts.

## How it works
Coming soon...

## Running the Script
### Download the Repository
First, download this repository onto your device. You can either manually download, or navigate to your chosen directory and run:

```
git clone https://github.com/bclevine/cluster_analyzer.git
```

### Configuration
Open `example_script.py` in your text editor of choice. You will see a config section at the top of the file. Adjust these parameters as necessary.

Then, find `RETURN_VALUES` and adjust the output of the function according to your needs.

### Run The Script
It's time to run the script. In your terminal, navigate to the directory and type the following command:

```
python3 example_script.py
```

The following flags are available:
1. *-n [number of threads]* : How many threads should be used for multithreading? More threads will usually be faster. Defaults to 25.
2. *-m [masking correction?]* : Should we include a correction for masking? Requires random DECaLS Catalogs. Defaults to True.
3. *-l [length of coordinate list]* : For testing, you may not want to download the entire catalog from your textfile. This number will cap the list at a certain index — for example, `-l 5` will download the first 5 coordinates from the list. Leave the flag blank to download the entire textfile.
4. *-t [timeout]* : How long should the maximum timeout be for any single line of sight? Defaults to 60, with units in seconds.
5. *-v [verbose]* : Use verbose output? Defaults to False. If True, the progress bar will be disabled.
6. *-s [sigma]* : Manually set sigma for finding cluster members. Defaults to 1. 

For example, the following command will use 10 threads and download the first 60 coordinates in the textfile.

```
python3 example_script.py -n 10 -l 60
```

The runs may sometimes last a long time. If you're running on a remote server and worried about losing connection, you can do:

```
nohup python3 example_script.py -n 10 -l 60 >> term_output.log &
```

This will redirect all terminal outputs to `term_output.log` and run the script in the background, so even if you disconnect, it will still continue working.

## Using the Results
The script will output a .npy file which you can open using [this Jupyter notebook](Example Analyzer.ipynb). It will also have a plaintext output in the `output.log` file, if you use that option.
