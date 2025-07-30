# Code for running the experiment
## Steps
This is minimal code to run the Veith_et_al 2025 experiments.  
Running the experiment involves two steps:  
- Run sound targeting. This measures the impulse responses in the tank, defines the sound waveforms and calculates the right loudspeaker activations to cancel echoes. Follow the instructions in `sound_targeting>README`.
- Run the experiment including video recording and closed-loop sound playback, based on prior sound targeting. See `vocloc_recorder>README`
