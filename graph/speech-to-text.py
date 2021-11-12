from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa as lb
import torch


# import tranformers
# print(transformers.__version__)

# Initialize the tokenizer
tokenizer = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')

# Initialize the model
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
# Read the sound file
# ../pod-audio.wav
waveform, rate = lb.load('../shortAudio.wav', sr = 16000)

# Tokenize the waveform
input_values = tokenizer(waveform, return_tensors='pt').input_values

# Retrieve logits from the model
logits = model(input_values).logits

# Take argmax value and decode into transcription
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)

# Print the output
print(transcription)

# predictor.predict(sentence="did uriah honestly think he could beat tim ferris in under three hours at the lincon memorial?.")


# text = ("When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously. I can tell you very senior CEOs of major American car companies would shake my hand and turn away because I wasn’t worth talking to, said Thrun, in an interview with Recode earlier this week.")

# import spacy
# nlp = spacy.load("en_core_web_sm")
# text = "did Uriah honestly think he could beat Tim Ferris in under three hours at the Lincon memorial"
# doc = nlp(text)

# # Find named entities, phrases and concepts
# for entity in doc.ents:
#     print(entity.text, entity.label_)