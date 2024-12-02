# Live speech translation!

I did this for my gf's PhD defense, to have real time captions in Italian of what she was saying in English!

It mixes two great open source projects:
- moonshine, which is a decent open source speech to text model (way faster than whisper)
- argos translate, which is another open source project with decent and fast neural translation

## Install

- Go to the [moonshine repo](https://github.com/usefulsensors/moonshine/) and follow the installation instructions for the moonshine-onnx live transcription demo
- If you installed `uv`, run `uv pip install argostranslate` to get [Argos Translate](https://www.argosopentech.com/)

## Use
- Run the script in this repo, wait for the models to be downloaded (only the first time), and then speak! You can change source and target languages in the script.

  
