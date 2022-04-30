This code provides a test only version of our code
You can use this code to reconstruct a CAD mesh from a fine-tuned model of a ABC shape

step 0, data prepare
The fine-tuned model of one ABC shape is contained in this code.
You don't need to do anything in this step

step 1, setup your environment
conda env create -f environment.yml

Additional package needs to be installed
pip install PyMCubes
PyMesh: https://github.com/PyMesh/PyMesh/releases
or any other package it requires when you try to run

step 2, run
python3 test.py -e abc -g 0

0 is your available gpu id

step 3, check results
check results in ./output_mesh directory