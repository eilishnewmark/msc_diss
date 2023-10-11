# MSc Dissertation 
## Data Preparation for Input to and Evaluation of neural Seq2Seq TTS Frontend 
Code used in conjunction with an implementation of a Seq2Seq LSTM TTS frontend, to process and evaluate Google Research's Wikipedia Homograph Dataset (WHD) and LibriSpeech data, with the aim of improving the TTS frontend's homograph disambiguation abilities.

The data was processed to add supplementary POS tags (from Festival and SpaCy) as input to the model on a per-character basis, and also as part of a MultiTask Learning paradigm.
For this, the WHD was also cleaned so that it could be entered to Festival without Out-of-Dictionary words.

Data was in the form:
 <img width="850" alt="Screenshot 2023-10-11 at 11 05 42" src="https://github.com/eilishnewmark/msc_diss/assets/116748480/9a44d0df-7d5c-4dd2-8e32-a43365c7f5e1">
with added POS tags:
VBD VBD VDB VDB VDB VDB # VBP VBP VBP # DT DT DT # JJ JJ JJ JJ JJ JJ # NN NN NN NN NN NN #

## Model diagram:
![model_architecture](https://github.com/eilishnewmark/msc_diss/assets/116748480/d02f1802-416c-4e0a-bddc-ffe75879efe6)
