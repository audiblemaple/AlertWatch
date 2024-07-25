"""
Real-Time Speech Recognition Using Wav2Vec2 Model

This script captures audio from a microphone, detects when the speaker stops talking, and transcribes the speech using a Wav2Vec2 model. The transcription is performed in real-time.

Modules:
    - wave: For reading and writing WAV files
    - torch: PyTorch for model handling and tensor operations
    - torchaudio: For audio processing
    - pyaudio: For capturing audio from the microphone
    - soundfile: For reading sound files
    - numpy: For numerical operations
    - yaml: For loading configuration files

Classes:
    - GreedyCTCDecoder: A decoder class to convert model emissions to text using Greedy CTC decoding.

Functions:
    - is_silent(data_chunk): Returns 'True' if the data_chunk is below the 'silent' threshold.
    - compute_energy(frames): Compute the energy of the audio frames.
    - load_word_weights(yaml_file): Load word weights from a YAML file.
    - compute_response_score(transcription, word_weights): Compute the response score based on the transcription and word weights.

Usage:
    Run the script and start speaking. The script will listen to your speech, detect pauses, and transcribe the speech when you stop talking. Press Ctrl+C to stop the script.
"""

import wave
import torch
import torchaudio
import pyaudio
import soundfile as sf
from torchaudio.transforms import Resample
import numpy as np
import yaml

# Initialize the audio input
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 1500
SILENCE_CHUNKS = 30
MIN_RECORDING_TIME = 2
ENERGY_THRESHOLD = 1000
REQUIRED_SCORE = 2.0

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Load the Wav2Vec2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
model_labels = bundle.get_labels()


class GreedyCTCDecoder(torch.nn.Module):
    """
    A decoder class to convert model emissions to text using Greedy CTC decoding.

    Args:
        labels (list of str): The list of labels for the transcription.
        blank (int, optional): The index of the blank label. Defaults to 0.
    """

    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """
        Given a sequence emission over labels, get the best path string.

        Args:
            emission (torch.Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
            str: The resulting transcript.
        """
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


decoder = GreedyCTCDecoder(labels=model_labels)
print("Recording and transcribing...")


def is_silent(data_chunk):
    """
    Returns 'True' if the data_chunk is below the 'silent' threshold.

    Args:
        data_chunk (bytes): The audio data chunk.

    Returns:
        bool: True if the data chunk is silent, False otherwise.
    """
    max_val = np.max(np.frombuffer(data_chunk, dtype=np.int16))
    return max_val < SILENCE_THRESHOLD


def compute_energy(frames):
    """
    Compute the energy of the audio frames.

    Args:
        frames (list of bytes): The recorded audio frames.

    Returns:
        float: The computed energy.
    """
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    return np.sum(audio_data ** 2) / len(audio_data)


def load_word_weights(yaml_file):
    """
    Load word weights from a YAML file.

    Args:
        yaml_file (str): The path to the YAML file containing word weights.

    Returns:
        dict: A dictionary with words as keys and their corresponding weights as values.
    """
    with open(yaml_file, 'r') as file:
        words_data = yaml.safe_load(file)
    return words_data['words']


def compute_response_score(transcription, word_weights):
    """
    Compute the response score based on the transcription and word weights.

    Args:
        transcription (str): The transcribed text.
        word_weights (dict): A dictionary with words as keys and their corresponding weights as values.

    Returns:
        float: The computed response score.
    """
    words = transcription.split()
    word_score = sum(word_weights.get(word, 0) for word in words)
    return word_score


word_weights = load_word_weights('word_weights.yaml')

try:
    while True:
        print("Listening...")
        frames = []
        silent_chunks = 0

        while True:
            data = stream.read(CHUNK)
            frames.append(data)

            silent_chunks = silent_chunks + 1 if is_silent(data) else 0
            # if is_silent(data):
            #     silent_chunks += 1
            # else:
            #     silent_chunks = 0

            if silent_chunks > SILENCE_CHUNKS and len(frames) > (RATE / CHUNK * MIN_RECORDING_TIME):
                break

        energy = compute_energy(frames)
        if energy < ENERGY_THRESHOLD:
            print("Detected silence or low energy. Skipping transcription.")
            continue

        wf = wave.open("temp_audio/temp.wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        waveform, sample_rate = sf.read("temp_audio/temp.wav")
        waveform = torch.tensor(waveform, dtype=torch.float32).to(device)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=1)

        if sample_rate != bundle.sample_rate:
            waveform = Resample(sample_rate, bundle.sample_rate)(waveform.unsqueeze(0)).squeeze(0)

        with torch.inference_mode():
            model_emission, _ = model(waveform.unsqueeze(0))

        transcript = decoder(model_emission[0])
        transcript_str = str(transcript).replace("|", " ").lower()
        print("Transcription: ", transcript_str)

        score = compute_response_score(transcript_str, word_weights)
        print(f"Response score: {score:.2f}")

        if score >= REQUIRED_SCORE:
            print("Alertness test passed.")
        else:
            print("Response not sufficient. Continue monitoring.")

except KeyboardInterrupt:
    print("Stopped recording.")

stream.stop_stream()
stream.close()
audio.terminate()

# backup of a previous version
# import wave
# import torch
# import torchaudio
# import pyaudio
# import soundfile as sf
# from torchaudio.transforms import Resample
# import numpy as np
# import yaml
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
#
# # Initialize the audio input
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# CHUNK = 1024
# SILENCE_THRESHOLD = 1000
# SILENCE_CHUNKS = 30
# MIN_RECORDING_TIME = 2
# ENERGY_THRESHOLD = 950
#
# # Initialize PyAudio
# audio = pyaudio.PyAudio()
# stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
#
# # Load the Wav2Vec2 model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
# model = bundle.get_model().to(device)
# model_labels = bundle.get_labels()
#
# class GreedyCTCDecoder(torch.nn.Module):
#     def __init__(self, labels, blank=0):
#         super().__init__()
#         self.labels = labels
#         self.blank = blank
#
#     def forward(self, emission: torch.Tensor) -> str:
#         indices = torch.argmax(emission, dim=-1)
#         indices = torch.unique_consecutive(indices, dim=-1)
#         indices = [i for i in indices if i != self.blank]
#         return "".join([self.labels[i] for i in indices])
#
# decoder = GreedyCTCDecoder(labels=model_labels)
# print("Recording and transcribing...")
#
# def is_silent(data_chunk):
#     max_val = np.max(np.frombuffer(data_chunk, dtype=np.int16))
#     return max_val < SILENCE_THRESHOLD
#
# def compute_energy(frames):
#     """
#     Compute the energy of the audio frames.
#
#     Args:
#         frames (list of bytes): The recorded audio frames.
#
#     Returns:
#         float: The computed energy.
#     """
#     audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
#     return np.sum(audio_data ** 2) / len(audio_data)
#
# def load_sentences(yaml_file):
#     with open(yaml_file, 'r') as file:
#         data = yaml.safe_load(file)
#     return data['sentences']
#
# def find_best_match(transcription, sentences):
#     vectorizer = TfidfVectorizer().fit_transform([transcription] + sentences)
#     vectors = vectorizer.toarray()
#     cos_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
#     best_match_idx = np.argmax(cos_similarities)
#     best_match_sentence = sentences[best_match_idx]
#     best_match_score = cos_similarities[best_match_idx]
#     return best_match_sentence, best_match_score
#
# sentences = load_sentences('sentences.yaml')
#
# try:
#     while True:
#         print("Listening...")
#         frames = []
#         silent_chunks = 0
#
#         while True:
#             data = stream.read(CHUNK)
#             frames.append(data)
#
#             if is_silent(data):
#                 silent_chunks += 1
#             else:
#                 silent_chunks = 0
#
#             if silent_chunks > SILENCE_CHUNKS and len(frames) > (RATE / CHUNK * MIN_RECORDING_TIME):
#                 break
#
#         energy = compute_energy(frames)
#         print("Energy is: ", energy)
#         if energy < ENERGY_THRESHOLD:
#             print("Detected silence or low energy. Skipping transcription.")
#             continue
#
#         wf = wave.open("temp.wav", 'wb')
#         wf.setnchannels(CHANNELS)
#         wf.setsampwidth(audio.get_sample_size(FORMAT))
#         wf.setframerate(RATE)
#         wf.writeframes(b''.join(frames))
#         wf.close()
#
#         waveform, sample_rate = sf.read("temp.wav")
#         waveform = torch.tensor(waveform, dtype=torch.float32).to(device)
#
#         if waveform.ndim > 1:
#             waveform = waveform.mean(dim=1)
#
#         if sample_rate != bundle.sample_rate:
#             waveform = Resample(sample_rate, bundle.sample_rate)(waveform.unsqueeze(0)).squeeze(0)
#
#         with torch.inference_mode():
#             model_emission, _ = model(waveform.unsqueeze(0))
#
#         transcript = decoder(model_emission[0])
#         transcript_str = str(transcript).replace("|", " ").lower()
#         print("Transcription: ", transcript_str)
#
#         best_match_sentence, best_match_score = find_best_match(transcript_str, sentences)
#         print("Matching string not found...") if best_match_score < 0.4 else print(f"Best match: {best_match_sentence} (Score: {best_match_score:.4f})")
#
# except KeyboardInterrupt:
#     print("Stopped recording.")
#
# stream.stop_stream()
# stream.close()
# audio.terminate()
#
#
# #
# #
# # """
# # Real-Time Speech Recognition Using Wav2Vec2 Model
# #
# # This script captures audio from a microphone, detects when the speaker stops talking, and transcribes the speech using a Wav2Vec2 model. The transcription is performed in real-time.
# #
# # Modules:
# #     - wave: For reading and writing WAV files
# #     - torch: PyTorch for model handling and tensor operations
# #     - torchaudio: For audio processing
# #     - pyaudio: For capturing audio from the microphone
# #     - soundfile: For reading sound files
# #     - numpy: For numerical operations
# #
# # Classes:
# #     - GreedyCTCDecoder: A decoder class to convert model emissions to text
# #
# # Functions:
# #     - is_silent(data_chunk): Checks if the audio data chunk is below the silence threshold
# #
# # Usage:
# #     Run the script and start speaking. The script will listen to your speech, detect pauses, and transcribe the speech when you stop talking. Press Ctrl+C to stop the script.
# # """
# #
# # import wave
# # import torch
# # import torchaudio
# # import pyaudio
# # import soundfile as sf
# # from torchaudio.transforms import Resample
# # import numpy as np
# #
# # # Initialize the audio input
# # FORMAT = pyaudio.paInt16
# # CHANNELS = 1
# # RATE = 16000
# # CHUNK = 1024
# # SILENCE_THRESHOLD = 1500  # Increased silence threshold for better detection
# # SILENCE_CHUNKS = 30  # Number of chunks of silence before stopping
# # MIN_RECORDING_TIME = 2  # Minimum recording time in seconds
# # ENERGY_THRESHOLD = 1000  # Increased energy threshold to consider meaningful audio
# #
# # # Initialize PyAudio
# # audio = pyaudio.PyAudio()
# #
# # # Start the audio stream
# # stream = audio.open(format=FORMAT,
# #                     channels=CHANNELS,
# #                     rate=RATE,
# #                     input=True,
# #                     frames_per_buffer=CHUNK
# #                     )
# #
# # # Load the Wav2Vec2 model
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
# # model = bundle.get_model().to(device)
# # model_labels = bundle.get_labels()
# #
# #
# # class GreedyCTCDecoder(torch.nn.Module):
# #     """
# #     A decoder class to convert model emissions to text using Greedy CTC decoding.
# #
# #     Args:
# #         labels (list of str): The list of labels for the transcription.
# #         blank (int, optional): The index of the blank label. Defaults to 0.
# #     """
# #     def __init__(self, labels, blank=0):
# #         super().__init__()
# #         self.labels = labels
# #         self.blank = blank
# #
# #     def forward(self, emission: torch.Tensor) -> str:
# #         """
# #         Given a sequence emission over labels, get the best path string.
# #
# #         Args:
# #             emission (torch.Tensor): Logit tensors. Shape `[num_seq, num_label]`.
# #
# #         Returns:
# #             str: The resulting transcript.
# #         """
# #         indices = torch.argmax(emission, dim=-1)  # [num_seq,]
# #         indices = torch.unique_consecutive(indices, dim=-1)
# #         indices = [i for i in indices if i != self.blank]
# #         return "".join([self.labels[i] for i in indices])
# #
# #
# # decoder = GreedyCTCDecoder(labels=model_labels)
# #
# # print("Recording and transcribing...")
# #
# #
# # def is_silent(data_chunk):
# #     """
# #     Returns 'True' if the data_chunk is below the 'silent' threshold.
# #
# #     Args:
# #         data_chunk (bytes): The audio data chunk.
# #
# #     Returns:
# #         bool: True if the data chunk is silent, False otherwise.
# #     """
# #     max_val = np.max(np.frombuffer(data_chunk, dtype=np.int16))
# #     return max_val < SILENCE_THRESHOLD
# #
# #
# # def compute_energy(frames):
# #     """
# #     Compute the energy of the audio frames.
# #
# #     Args:
# #         frames (list of bytes): The recorded audio frames.
# #
# #     Returns:
# #         float: The computed energy.
# #     """
# #     audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
# #     energy = np.sum(audio_data ** 2) / len(audio_data)
# #     return energy
# #
# #
# # try:
# #     while True:
# #         print("Listening...")
# #         frames = []
# #         silent_chunks = 0
# #
# #         while True:
# #             data = stream.read(CHUNK)
# #             frames.append(data)
# #
# #             if is_silent(data):
# #                 silent_chunks += 1
# #             else:
# #                 silent_chunks = 0
# #
# #             # Only break if silence is detected after minimum recording time
# #             if silent_chunks > SILENCE_CHUNKS and len(frames) > (RATE / CHUNK * MIN_RECORDING_TIME):
# #                 break
# #
# #         # Compute the energy of the recorded audio
# #         energy = compute_energy(frames)
# #         if energy < ENERGY_THRESHOLD:
# #             print("Detected silence or low energy. Skipping transcription.")
# #             continue
# #
# #         # Save the recorded audio to a temporary file
# #         wf = wave.open("temp.wav", 'wb')
# #         wf.setnchannels(CHANNELS)
# #         wf.setsampwidth(audio.get_sample_size(FORMAT))
# #         wf.setframerate(RATE)
# #         wf.writeframes(b''.join(frames))
# #         wf.close()
# #
# #         # Load the waveform using soundfile
# #         waveform, sample_rate = sf.read("temp.wav")
# #         waveform = torch.tensor(waveform, dtype=torch.float32).to(device)
# #
# #         # If the waveform is stereo, convert it to mono by taking the mean of the channels
# #         if waveform.ndim > 1:
# #             waveform = waveform.mean(dim=1)
# #
# #         # Resample if needed
# #         if sample_rate != bundle.sample_rate:
# #             waveform = Resample(sample_rate, bundle.sample_rate)(waveform.unsqueeze(0)).squeeze(0)
# #
# #         # Get the model's output
# #         with torch.inference_mode():
# #             model_emission, _ = model(waveform.unsqueeze(0))
# #
# #         # Decode the emission to text
# #         transcript = decoder(model_emission[0])
# #         print("Transcription: ", str(transcript).replace("|", " "))
# #
# # except KeyboardInterrupt:
# #     print("Stopped recording.")
# #
# # # Clean up
# # stream.stop_stream()
# # stream.close()
# # audio.terminate()
