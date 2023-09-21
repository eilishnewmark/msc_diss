# msc_diss
Code used to process and evaluate Google Research's Wikipedia Homograph Dataset (WHD) and LibriSpeech data for use in a neural TTS Seq2Seq frontend model, wiht the aim of improving the model's homograph disambiguation abilities.

The data was processed to add supplementary POS tags (from Festival and SpaCy) as input to the model, and also as part of a MultiTask Learning paradigm.
For this, the WHD was also cleaned so that it could be entered to Festival without Out-of-Dictionary words.
