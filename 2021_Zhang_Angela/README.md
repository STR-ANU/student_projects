For the logs files please go to the following site, download, unzip and put the files in the same folder as the simulation scripts:
https://anu365-my.sharepoint.com/:f:/g/personal/u6986436_anu_edu_au/Ev_3kiughEtBiqRLp5WLjL0B4zGpoP2JnY1uDwokeNlWDg?e=Zz69yP

-----------------------------------------------------
For simulation logs:

First run in the order of SITL_processing.py, EKF_processing.py and EqF_processing.py to produce relevant pickle files for analysis (Order is important!).
Log available for analysis:
bowtie_withbias_3.BIN
line1_withbias.BIN
EqF1_withbias.BIN

'SITL_processing.py' and 'EKF_processing.py' requires the log name (eg. bowtie_withbias_3.BIN) as input in command.

'EqF_processing.py': requires the log name (eg. bowtie_withbias_3.BIN), optional argument include (typically shouldn't need to change): 
--param(default true to use same noise parameter as EKF, else type 'false' and manually change the noise values you want in the actual code)
--addnoise (default false, else type 'true' which manually adds Gaussian noise to measurements, manually change noise you want to add in xxx_noise variables in line 63-70)


Then run the other analysis scripts as required:

'RSE position.py': Plots the attitude, position and velocity error between the two filters. Requires to input log name with the following optional argument:
--param(default true to use pickle file generated using same noise parameter as EKF, else type 'false' to analyse log with user-defined parameters)

'Bias comparison.py': Plots of gyro and accel bias. Requires to input log name with the following optional argument:
--param(default true to use pickle file generated using same noise parameter as EKF, else type 'false' to analyse log with user-defined parameters)

'NEES_NIS comparison.py': Plots of NEES and NIS. Requires to input log name with the following optional arguments:
--param(default true to use pickle file generated using same noise parameter as EKF, else type 'false' to analyse log with user-defined parameters)
--addnoise(default false, type 'true' if in running 'EqF_processing.py' user have manually added Gaussian noise to the measurements.)

'Monte_Carlo.py': Runs the Monte Carlo simulations for NEES and NIS analysis. Requires to input log name with the following optional arguments:
!!Please read before you run: I have also included the pickle files used in my Monte Carlo analysis, to get the same results, need to comment out line 53-54 which calls EqF multiples times for Monte Carlo. If you run the code it will replace the pre-uploaded pickle files for the analysis.
--simnum(number of simulations to run for Monte Carlo, default is 10)
--param(default true to use pickle file generated using same noise parameter as EKF, else type 'false' to analyse log with user-defined parameters)
--addnoise(default true and should not be changed at all)

-----------------------------------------------------
For real flight log:
Run 'Real Flight Data analysis.py': Requires to input log name and type '--correct-gps' for correcting gps sample to account for time delay, optional argument:
--param(default true to use pickle file generated using same noise parameter as EKF, else type 'false' to analyse log with user-defined parameters (need to go into code to manually change to desired value))
--gps-num(default to select the first (M8) GPS measurements for EqF update, change with caution as the script compares RTK position against EqF estimated position against the source of GPS used, if changing source may lead to double ups ie. RTK position plotted twice and EqF estimated position).

'Real Flight Data analysis.py' will automatically first run the 'Real Data EqF_processing' script and then plot the position and velocity results.
