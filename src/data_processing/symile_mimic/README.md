### Generate data

In order to generate We also provide the code used to create the dataset from scratch

#### Google Cloud APIs

We use the Google Cloud Translation and Text-to-Speech APIs to create our dataset. You'll need the following Google client libraries to run `generate_data.py`.

##### Translate

https://cloud.google.com/translate/docs/setup

just need basic client libraries

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