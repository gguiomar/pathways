just focusing on the functions and the splitting in @/pathways_extracted/main_simple.py how could we best organize the this code into tree files

main.py - contains the pipeline defined in @/pathways_extracted/main_simple.py
training.py - only handles the training and can get called
test.py - only handles the creation of the test dataset in CONTROL 
optogenetics.py - performs the optogenetics experiments for DLS and DMS
perceptualpolicy.py - contains the organized code for the corresponding section and a grid plot with all the plots

also there are a lot of functions in @/analysis/pathway_analysis.py could we first check which functions are not being used at all in @/pathways_extracted/main_simple.py and list them and then could we find a way of splitting the analysis into more files?

after doing that, refactor the code base so that we call the analysis from the new classes

add tqdm to the testing simulations (check for jupyter notebook or not)

but first can you give me a potential new organization structure?
let's also plan to remove the unused functions
how would a new @main.py look based on these changes?
could we also add the possibility of saving the sim_data into a pckl and automatically loading if a new simulation is not run for the .py that deal with the test, optogenetics and perceptualpolicy
this should be a simple function that can get called in main.py or in 

for the perceptual policy section please cleanup the code as much as possible as there are lost functions there and repeated instatiatons of the test_data generation

use the train.py and test.py to generate these files and the perceptualpolicy.py file should just load sim_data and train_data and run the necessary functions

create a ppolicy_utils.py for the analysis functions used in this section including the plotting and generate a grid plot with all the figures present 

create a plots folder where all figures are saved