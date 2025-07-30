# Loop functions
## How to use
Collection of classes that implement flexible user functions for e.g. live video analysis or triggering stimuli during video and audio acquisition.
A specific class is activated if its name is defined in the `config.json`.

## Independent functions
See collection of modules in the `independent` folder  
A function that starts looping once the recording starts. It cannot set any triggers and does not receive a video stream.

## Video functions
See collection of modules in the `video` folder  
A function that receives a live video stream frame by frame (can be downsampled in space and time). The function can set a trigger to activate the triggered function and also send information to the triggered function.

## Audio functions
See collection of modules in the `audio` folder  
A function that receives audio chunkwise for further processing. The function can set a trigger to activate the triggered function and also send information to the triggered function.

## Triggered functions
See collection of modules in the `triggered` folder 
A function that is triggered once the trigger is set by the video function.
