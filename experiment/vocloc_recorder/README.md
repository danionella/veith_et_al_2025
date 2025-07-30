# Synced audio and video acquisition
## How to use
call 

    python vocloc_recorder.py config.json C:\\myrecordings\ 3600
    
   last argument is recording time in s.  

Or start the gui 

    python run.py

## Installation

    conda env create -f requirements.yml
    conda activate recorder
    
   Install the camera's python inferface. Currently supported are Basler Pylon and Spinnaker:\
   For Spinnaker cameras, go to https://flir.app.boxcn.net/v/SpinnakerSDK, download the right python whl (py3.6) and install it via
 
    pip install spinnaker_python-1.29.0.5-cp36-cp36m-win_amd64.whl
   or - if outdated - install a python wrapper provided by the manufacturer of your camera.  

   Install the frozen copy of https://github.com/portugueslab/arrayqueues into the environment.

    cd arrayqueues
    pip install -e .
  
  Set all desired variables in `config.json`.

  The video saver thread uses GPU accelerated encoding: make sure you have ffmpeg with NVIDIA nvenc support installed.

## Functionalities
The software allows synced acquisition of video and audio via NI-DAQ systems.  
Multiple processes stream audio and video to distributing processes for analysis and saving.  
One can add live analysis to the video stream and trigger a triggered function by writing a new class in one of the loops modules. These classes become a menu of functionalities which can be selected via the config file.

## Explanation of the config file 
    {
      "general": {
        "encoderPath": "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe", PATH TO ENCODER EXECUTABLE
        "logPath": ".\\log.txt", PATH TO FOLDER WHERE LOG FILE IS SAVED
        "savePath": "C:\\", DEFAULT SAVE PATH
      },
      "audio": { PARAMETER REGARDING AUDIO AND SYNCRONISATION
        "triggerDigital": true, RECORD CAMERA TRIGGER AT DIGITAL PORT
        "save": true, SAVE AUDIO
        "rate": 51200, SAMPLINGRATE OF AUDIO FILE
        "clockStr": "cDAQ2", DEVICE FOR CLOCK OF ALL TASKS
        "chStr": [ AUDIO CHANNELS TO RECORD
          "cDAQ2Mod1/ai0",
          "cDAQ2Mod1/ai1",
          "cDAQ2Mod1/ai2",
          "cDAQ2Mod1/ai3"
        ],
        "chStrTrig": "cDAQ2Mod3/port0/line1", CHANNEL WHERE THE CAMERA TRIGGER ARRIVES
        "peakAmp": 4, EXPECTED PEAK AMPLITUDE OF AUDIOFILES (V) 
        "szChunk": 100000, SIZE OF A READOUT CHUNK (NSAMPLES EVENT OF DAQ)
      },
      "audioPreview": {
        "active": false, WHETHER TO SHOW CHUNK-WISE LIVE SPECTROGRAM (MAX ACROSS AUDIO CHANNELS)
        "listen_live": false WHETHER TO PLAYBACK MEAN SIGNAL VIA SPEAKERS
        },
      "video": { PARAMETERS REGARDING THE VIDEO ENCODING
        "save": true, WHETHER TO SAVE
        "framerate": 24, FRAMERATE TO ENCODE, SHOULD MATCH TRIGGER RATE
        "qmin": 17, QUALITY PARAMETERS
        "qmax": 21,
        "bitrate": 250,
        "buffer_seconds": 3 VIDEO BUFFER. SET LOW IF YOU WANT TO TEST WHETHER YOUR FUNCTIONS RUN LIVE
      },
      "preview": { VIDEO PREVIEW
      "active": true,
      "sub_spatial": 1 TEMPORAL SUBSAMPLING OF PREVIEW STREAM
      },
      "trigger": { PARAMETERS REGARDING THE CAMERA TRIGGER
        "chStr": [ WHERE TO SEND THE TRIGGER, ALSO SUPPORTS DIGITIAL OUT
          "cDAQ2Mod3/port0/line2"
        ],
        "HIGH": 4, HIGH VOLTAGE IN V (IF ANALOG CHANNEL)
        "LOW": 0, LOW VOLTAGE IN V (IF ANALOG CHANNEL)
        "rate": 24, RATE AT WHICH TO TRIGGER IN HZ
        "duration": 1 DURATION OF TRIGGER PULSES IN ms
      },
      "camera": { PARAMETERS REGARDING THE CAMERA 
        "interface": "Pylon", WHICH CAMERA TO USE: Pylon/Spinnaker
        "exposure": 4, CAMERA EXPOSURE IN ms
        "width": 1024, ROI SIZE AND OFFSET IN px
        "height": 1024,
        "xoff": 208,
        "yoff": 4,
        "triggered": true TRIGGERED OR FREE RUNNING
      },
      "vis": { ONLY USED BY OLD CUSTOM FUNCTIONS, CAN BE DELETED AT SOME POINT
        "camWindowSize": [
          600,
          600
        ]
      },
      "audio_out": { OPTIONAL AUDIO OUTPUT CHANNELS, CAN BE AN EMPTY LIST
        "chStr": [
          "cDAQ2Mod2/ao0",
          "cDAQ2Mod2/ao1"
        ]
      },
      "indepFunc": { DEFINE A CUSTOM FUNCTION THAT RUNS IN LOOP, SEE loops>independent
        "active": false,
        "className": "MyTest",
        "allowSaving": true
      },
      "videoFunc": { DEFINE A CUSTOM VIDEO FUNCTION THAT RUNS IN LOOP, see loops>video
        "active": true,
        "className": "MyPreviewWithTrigger",
        "sub_spatial": 1,
        "sub_t": 1,
        "allowSaving": true
      },
      "triggeredFunc": { DEFINE A CUSTOM FUNCTION THAT RUNS WHEN TRIGGERED, see loops>triggered
        "active": true,
        "className": "MyTest",
        "allowSaving": true
      },
      "audioFunc": {  DEFINE A CUSTOM AUDIO FUNCTION THAT RUNS IN LOOP, see loops>audio
        "active": true,
        "className": "LiveClickCount",
        "lowcut": 3000,
        "highcut": 14000,
        "snr_threshold": 10,
        "chStr_use": ["AI0/ai0", "AI0/ai1"],
        "chMixMethod": "max",
        "bin_seconds": 60,
        "allowSaving": true
        },
      "custom_config": { ADDITIONAL PARAMETERS USED ACROSS CUSTOM FUNCTIONS
        "fn_stimset": "",
        "power_switch_on": 0,
        "serial_port": "/dev/ttyACM0",
        "power_switch_delay": 0.075,    
        "biased_playback_map": {"400": 1.79167, "800": 2.05882, "1200": 3, "2000": 6.28378, "5000": 71.42857} PLAY SOME SOUNDS MORE OFTEN
      }
    }
