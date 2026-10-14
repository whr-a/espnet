# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Audited ASR-WER on generated audio and AVQA choice rewards."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path


def normalize(text):
    return re.sub(r"[^\w\s]"," ",unicodedata.normalize("NFKC",text).casefold()).split()


def wer(reference,hypothesis):
    expected=normalize(reference); actual=normalize(hypothesis); row=list(range(len(actual)+1))
    for i,word in enumerate(expected,1):
        new=[i]
        for j,value in enumerate(actual,1): new.append(min(new[-1]+1,row[j]+1,row[j-1]+(word!=value)))
        row=new
    return row[-1]/max(1,len(expected))


def qa_reward(source,generation):
    text=generation["text"]
    match=re.search(r"<answer>\s*(.*?)\s*</answer>",text,re.I|re.S)
    answer=match.group(1).strip() if match else text.strip()
    choices=source.get("choices",[])
    label=re.fullmatch(r"\(?([A-Z])\)?[.)]?",answer)
    if label and ord(label.group(1))-65<len(choices): answer=choices[ord(label.group(1))-65]
    answer=re.sub(r"^(?:[A-Z]\.|\([A-Z]\))\s*","",answer)
    return float(normalize(answer)==normalize(source["reference"]))


class ASRWERReward:
    """Load a fixed independent ASR and decode actual generated codec actions."""
    def __init__(self,config):
        import torch
        from transformers import XcodecModel,AutoConfig,AutoProcessor,AutoModelForSpeechSeq2Seq
        self.torch=torch; self.config=config; self.options=config.adapter_options
        self.directory=Path(config.output_dir)/"reward_audit"; self.directory.mkdir(exist_ok=True)
        self.codec=XcodecModel(AutoConfig.from_pretrained(self.options["codec_config_path"]))
        native=torch.load(config.checkpoint,map_location="cpu",weights_only=True)
        native=native.get("module",native)
        prefix="multimodal_io_dict.discrete_audio.codec_model."
        self.codec.load_state_dict({key[len(prefix):]:value for key,value in native.items() if key.startswith(prefix)},strict=True)
        self.codec.eval().to(self.options.get("reward_device","cpu"))
        self.asr_tag=self.options["asr_model"]
        self.asr_revision=self.options.get("asr_revision")
        if not Path(self.asr_tag).is_dir() and not self.asr_revision:
            raise ValueError("ASR reward must freeze a local model or revision")
        self.processor=AutoProcessor.from_pretrained(self.asr_tag,revision=self.asr_revision)
        self.asr=AutoModelForSpeechSeq2Seq.from_pretrained(self.asr_tag,revision=self.asr_revision).eval().to(self.options.get("reward_device","cpu"))

    def __call__(self,source,generation):
        import numpy as np
        import soundfile as sf
        torch=self.torch
        matrix=torch.tensor(generation["token_ids"],dtype=torch.long)
        base=152192; size=1025; q=8
        frames=matrix[(matrix[:,0]>=base)&(matrix[:,0]<base+size)]
        if len(frames)<q:
            raise ValueError("TTS candidate has no decodable audio; preserve as runtime failure")
        codes=torch.stack([(frames[:,s]-(base+s*size+1)).clamp(0,1023) for s in range(q)],-1)
        length=len(codes)-q+1
        aligned=torch.stack([codes[s:s+length,s] for s in range(q)],0)[None].to(self.codec.device)
        with torch.inference_mode(): waveform=self.codec.decode(aligned).audio_values.squeeze().cpu().float().numpy()
        if waveform.ndim!=1 or not waveform.size or not np.isfinite(waveform).all(): raise ValueError("Invalid generated WAV")
        rate=int(self.codec.config.sample_rate); digest=hashlib.sha256(waveform.tobytes()).hexdigest()
        target=self.directory/f"{source['id']}-{digest[:16]}"
        sf.write(str(target)+".wav",waveform,rate)
        inputs=self.processor(waveform,sampling_rate=rate,return_tensors="pt")
        with torch.inference_mode(): ids=self.asr.generate(**inputs.to(self.asr.device),language="en",task="transcribe",max_new_tokens=256)
        hypothesis=self.processor.batch_decode(ids,skip_special_tokens=True)[0]
        value=wer(source["reference"],hypothesis)
        Path(str(target)+".json").write_text(json.dumps({"source_id":source["id"],"reference":source["reference"],"hypothesis":hypothesis,"wer":value,"reward":-value,"waveform_sha256":digest,"sample_rate":rate,"asr_model":self.asr_tag,"asr_revision":self.asr_revision,"codec_frames":length,"generated_tokens":generation["token_ids"]},indent=2)+"\n")
        return -value


_REWARD=None

def tts_asr_wer(source,generation):
    global _REWARD
    if _REWARD is None:
        from espnet2.rl.config import ESPnetRLConfig
        config=ESPnetRLConfig.load(source["reward_config"])
        _REWARD=ASRWERReward(config)
    return _REWARD(source,generation)
