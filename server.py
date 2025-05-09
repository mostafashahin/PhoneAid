import sys, io
import os

import json
from core import utils
import transcriber
import Phonemize
from Levenshtein import editops

from fastapi import FastAPI, File, Form, UploadFile
import uvicorn

app = FastAPI()

engines = {}
phonemizers = {}

model_repo_dict = {'en_us_a':'mostafaashahin/SA_US_Adult',
                   'en_aus_c':'mostafaashahin/SA_Aus_Child'}

def create_phonemizer_fn(model_string):
    if model_string == 'en_us_a':
        phonemizer = Phonemize.phonemization()
        def phonemize_text(text):
            text = phonemizer.remove_special_characters(text)
            phonemes = phonemizer.cmu_phonemize(text)
            phonemes = [ph.lower() for ph in phonemes]
            return ' '.join(phonemes)
    elif model_string == 'en_aus_c':
        with open('data/en_aus_c.csv',encoding='utf-8') as f:
            pron_dict = dict([[i.strip() for i in a.split(',')] for a in f.read().splitlines()])
        def phonemize_text(text):
            return pron_dict.get(text.lower(),'')

    else:
        def phonemize_text(text):
            return ''
    return phonemize_text
    

def analyse_speech(speech_data, target_word, engine, phonemizer, p2att_file_path):
    with io.BytesIO(speech_data) as buffer:
        data, sr, dur = utils.load_speech_file(buffer)
    output =  engine.transcribe(data, attributes= 'all', phonological_matrix_file=p2att_file_path, human_readable=False)
    output_dict = json.loads(output)
    ref_phonemes = phonemizer(target_word.lower()).split()
    recog_phonems = output_dict["Phoneme"]['symbols']
    Expected_align, Recognized_align, Errors = get_error(ref_phonemes, recog_phonems)
    output_dict['Phoneme']['Assess'] = {'Expected_aligned':Expected_align,
                                        'Recognized_aligned':Recognized_align,
                                        'Errors':Errors}
    
    for att_item in output_dict['Attributes']:
        att = att_item['Name']
        rec_pat = att_item['Pattern']
        ref_pat = ['-' if f'n_{att}' in engine.p2att_map[p] else '+' for p in ref_phonemes]
        Expected_align, Recognized_align, Errors = get_error(ref_pat, rec_pat)
        att_item['Assess'] = {'Expected_aligned':Expected_align,
                              'Recognized_aligned':Recognized_align,
                              'Errors':Errors}
         
    output = json.dumps(output_dict)
    print(f'processing complete for word {target_word} with phonemes {ref_phonemes}')
    return output

def get_error(exp_list, rec_list):
    exp_list = list(exp_list)
    rec_list = list(rec_list)
    vocab = set(exp_list+rec_list)
    w2c = dict(zip(vocab,range(len(vocab))))
    
    exp_out = [[a,'H'] for a in exp_list]
    rec_out = [[a,'H'] for a in rec_list]  
    exp_enc = ''.join([chr(w2c[c]) for c in exp_list])
    rec_enc = ''.join([chr(w2c[c]) for c in rec_list])
    ops = editops(exp_enc, rec_enc)
    ops.reverse()

    for op, exp_i, rec_i in ops:
        if op == 'replace':
            exp_out[exp_i][1] = 'S'
            rec_out[rec_i][1] = 'S'
        elif op == 'insert':
            rec_out[rec_i][1] = 'I'
            exp_out.insert(exp_i,['','I'])
        elif op == 'delete':
            exp_out[exp_i][1] = 'D'
            rec_out.insert(rec_i,['','D'])

    return [p for (p,_) in exp_out], [p for (p,_) in rec_out], [t for (_,t) in rec_out]



@app.on_event("startup")
async def init_engines():
    """
    Preload models when the server starts.
    """
    global engines
    global phonemizers
    model_combinations = [("en_us", "a"), ("en_aus", "c")]
    for lang, age_cat in model_combinations:
        model_string = f'{lang}_{age_cat}'
        engine = transcriber.transcribe_SA(model_path=model_repo_dict[model_string], verbose=0)
        p2att_file_path = os.path.join('data',model_string,'p2att.csv')
        engines[model_string] = (engine, p2att_file_path)
        phonemizers[model_string] = create_phonemizer_fn(model_string)
    
    print("All models loaded.")





@app.post("/analyze/")
async def analyze(
    speech_signal: UploadFile = File(...),
    target_word: str = Form(...),
    lang: str = Form(...),
    age_cat: str = Form(...),
):
    # Read the speech signal
    speech_data = await speech_signal.read()

    # Write it to file for debugibg
    wav_fname = f"{utils.generate_file_basename()}.wav"
    json_fname = f"{utils.generate_file_basename()}.json"
    data = {
            "target_word": target_word,
            "lang": lang,
            "age_cat": age_cat
            }
    os.makedirs('tmp', exist_ok=True)
    with open(os.path.join('tmp',wav_fname), 'wb') as f:
        f.write(speech_data)

    with open(os.path.join('tmp',json_fname), 'w') as f:
        json.dump(data, f, indent=4)

    model_string = f'{lang}_{age_cat}'

    if model_string not in engines:
        return {"error": f"No model found for lang={lang}, age_cat={age_cat}"}
    if model_string not in phonemizers:
        return {"error": f"No phonemizer found for lang={lang}, age_cat={age_cat}"}

    # Get the preloaded model
    
    engine, p2att_file_path = engines.get(model_string)
    phonemizer = phonemizers.get(model_string)
    
    

    # Process the speech signal using the model
    result = analyse_speech(speech_data, target_word, engine, phonemizer, p2att_file_path)
    return result


# Run this with `uvicorn main:app --reload`
