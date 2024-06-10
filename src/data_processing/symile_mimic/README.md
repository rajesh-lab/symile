# Generating Symile-M3

In this section, we provide the code to generate Symile-M3 data from scratch.

## Setup

### Google Cloud APIs

You will need the Google Cloud Translation and Text-to-Speech client libraries:

#### Translation

Follow the instructions [here](https://cloud.google.com/translate/docs/setup) to create a project that has the Cloud Translation API enabled and credentials to make authenticated calls. When you get to the section [Installing client libraries](https://cloud.google.com/translate/docs/setup#installing_client_libraries), you will only need the basic client libraries.

##### Text-to-Speech

Follow the instruction here (although can we just package it up in the environment?): https://cloud.google.com/text-to-speech/docs/libraries (probably will need to still install and initialize gcloud CLI and create credential file like in here: https://cloud.google.com/docs/authentication/provide-credentials-adc)

- Include instructions for how to install google translate/tts

#### Create splits from ImageNet

#### Create datasets

`cd` into `src/symile_data/` and set dataset parameters in `args.py`. You'll likely want to update `--n_per_language` and `--save_path`. If you're generating data for the support classification experiment, you'll want to set `--negative_samples` to `True`.

Then run:

```
(symile-env) > python generate_data.py
```

Note that you should use this script to generate train/val/test sets separately in order to ensure that each split has the same number of samples from each template.
