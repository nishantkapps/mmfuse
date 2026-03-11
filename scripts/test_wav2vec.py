import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoModelForAudioClassification
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Model
import numpy as np
import glob, os, csv

'''
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

wav_files = glob.glob(os.path.join(os.path.dirname(__file__), "../dataset/sdata/**/*.wav"), recursive=True)
out_csv = os.path.join(os.path.dirname(__file__), "wav_transcriptions.csv")
with open(out_csv, "w", newline="") as csvf:
    writer = csv.writer(csvf); 
    writer.writerow(["file", "transcription"])
    for audio_file_path in wav_files:
        speech, rate = librosa.load(audio_file_path, sr=16000); 
        input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
        with torch.no_grad(): 
            logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1); 
        transcription = processor.decode(predicted_ids[0]);
        #print(os.path.abspath(audio_file_path),"|", transcription)
        writer.writerow([os.path.abspath(audio_file_path), transcription])
'''

# 1. Define your mappings
label_names = ["Start", "Go Here", "Move Down", "Move Up", "Stop", "Move Left", "Move Right", "Perfect"]
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {label: i for i, label in enumerate(label_names)}

model_name = "facebook/wav2vec2-base-960h"
model2 = AutoModelForAudioClassification.from_pretrained(model_name, num_labels=8, id2label=id2label, label2id=label2id)
model2.eval()

torch.manual_seed(0); 
np.random.seed(0)

audio_file_path = "/home/theta/nishant/projects/mmfuse/dataset/sdata/part4/p040/p040_m1_part4.wav"
speech2, rate2 = librosa.load(audio_file_path, sr=16000);
processor2 = Wav2Vec2Processor.from_pretrained(model_name)

fextract = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
inputs2 = fextract(speech2, sampling_rate=16000, return_tensors="pt")
print(speech2.shape)
with torch.no_grad():
    logits2 = model2(**inputs2).logits
print(f"Output shape (Batch, Labels): {logits2.shape}") # Should be [1, 8]  

predicted_id2 = torch.argmax(logits2, dim=-1).item() # Get index of max logit
print(f"Audio File: {audio_file_path}")
print(f"Predicted Label ID: {predicted_id2}")
print(f"Predicted Label Name: {model2.config.id2label[predicted_id2]}") # Look up name

#print("## -------------------------------------------- ")
#print(f"Transcription: {transcription}")
