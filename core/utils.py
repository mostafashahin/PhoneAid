import os
import random
import datetime
import zipfile
import librosa




def generate_file_basename():    
    # Generate current timestamp with millisecond precision
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate a random number (you can choose the range, e.g., 0-9999)
    random_number = random.randint(0, 9999)

    # Combine both to form the file name
    file_name = f"file_{random_number}_{timestamp}"

    return file_name


def zip_files(file_paths, output_zip):
    # Create a ZipFile object in write mode
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for file in file_paths:
            # Add each file to the zip
            zipf.write(file, os.path.basename(file))
    return f"Output files successfully zipped into {output_zip}"



def load_speech_file(speech_file, target_sr=16000):
    # Step 1: Load the stereo WAV file (keep it stereo with mono=False)
    y, sr = librosa.load(speech_file, sr=None, mono=False)

    # Step 2: Convert stereo to mono by averaging the two channels
    y = librosa.to_mono(y)

    # Step 3: Resample the mono data to 16,000 Hz
    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    
    duration = len(y)/target_sr
    
    return y, target_sr, duration
