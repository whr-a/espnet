# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Bagpiper speech model executed by SGLang's scheduler and primary sampler.

This correctness implementation recomputes the full prefix with the native HF
backbone. Each Engine request emits one primary action; secondary codec actions
are sampled here and returned through customized_info. The learner independently
scores the identical native model. TP=1, FP32, no prefix cache, no graphs.
"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3Model
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from sglang.srt.layers.logits_processor import LogitsProcessorOutput


class ESPnetBagpiperForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        if quant_config is not None:
            raise ValueError("Bagpiper first implementation requires unquantized TP=1")
        self.config=config
        self.geometry=config.espnet_bagpiper
        self.payload_path=Path(config.espnet_rl_payload_path)
        text=Qwen3Config(**self.geometry["text_config"])
        text._attn_implementation="sdpa"
        self.model=Qwen3Model(text)
        self.lm_head=nn.Linear(text.hidden_size,text.vocab_size,bias=False)
        self.stream_emb=nn.Embedding(self.geometry["num_stream"],text.hidden_size)
        audio=Qwen3OmniMoeAudioEncoderConfig(**{k:v for k,v in self.geometry["audio_config"].items() if k!="model_type"})
        audio._attn_implementation="sdpa"
        self.audio_tower=Qwen3OmniMoeAudioEncoder(audio)
        self.audio_adaptor=nn.Linear(self.geometry["adaptor_input_dim"],text.hidden_size,bias=True)
        self.eval()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def load_weights(self,weights):
        params=dict(self.named_parameters())
        loaded=set()
        mapping={"multimodal_io_dict.continuous_audio.model.audio_tower.":"audio_tower.","adaptor.continuous_audio.":"audio_adaptor."}
        for original,value in weights:
            if original.startswith("multimodal_io_dict.discrete_audio.codec_model."):
                continue  # Frozen decoder is separately strictly loaded for reward.
            name=original
            for before,after in mapping.items():
                if name.startswith(before): name=after+name[len(before):]; break
            if name not in params:
                raise ValueError(f"Unexpected Bagpiper native weight {original}")
            if params[name].shape!=value.shape:
                raise ValueError(f"Bagpiper shape mismatch {name}")
            with torch.no_grad(): params[name].copy_(value)
            loaded.add(name)
        missing=set(params)-loaded
        if missing: raise ValueError(f"Incomplete Bagpiper weights: {sorted(missing)}")
        return loaded

    def _hidden(self,payload,device):
        frames=torch.tensor(payload["sequence"],device=device,dtype=torch.long)[None]
        embeds=self.model.embed_tokens(frames)
        embeds[...,1:,:]=embeds[...,1:,:].masked_fill((frames[...,1:]==0).unsqueeze(-1),0)
        embeds=embeds.sum(2)
        native=payload.get("native_model_inputs",{})
        if native.get("continuous_audio_feats") is not None:
            feats=torch.tensor(native["continuous_audio_feats"],device=device,dtype=embeds.dtype)
            lengths=torch.tensor(native["continuous_audio_lengths"],device=device,dtype=torch.long)
            features=feats[0].T
            n_window=self.geometry["audio_config"].get("n_window",100)
            chunk=2*n_window
            full=chunk
            for _ in range(3): full=(full-1)//2+1
            rem=lengths%chunk
            for _ in range(3): rem=(rem-1)//2+1
            output_lengths=(lengths//chunk)*full+rem
            audio=self.audio_tower(features,feature_lens=lengths,aftercnn_lens=output_lengths)
            if not isinstance(audio,torch.Tensor):
                audio=getattr(audio,"last_hidden_state",None)
                if audio is None: raise TypeError("Missing Qwen audio feature Tensor")
            audio=self.audio_adaptor(audio)
            for bidx,start,length in native["continuous_audio_indices"]:
                embeds[bidx,start:start+length]=audio[:length]
        return self.model(inputs_embeds=embeds,use_cache=False).last_hidden_state[:,-1]

    def _allowed(self,raw,phase,mode,stream):
        cfg=self.geometry; tentative=int(raw.argmax())
        detect=tentative in (cfg["text_token_id"],cfg["audio_token_id"])
        audio=cfg["codec_base_offset"]<=tentative<raw.shape[-1] and not detect
        allowed=torch.zeros(raw.shape[-1],device=raw.device,dtype=torch.bool)
        if stream:
            lo=cfg["codec_base_offset"]+stream*cfg["codec_layer_size"]
            allowed[lo:lo+cfg["codec_layer_size"]]=True
            return allowed,audio
        if mode=="text_audio" and phase in ("transition","audio_stop"):
            allowed[cfg["assistant_token_id"] if phase=="transition" else cfg["eos_token_id"]]=True
        elif detect:
            allowed[cfg["text_token_id"]]=True
            if mode=="text_audio" and phase=="audio": allowed[cfg["audio_token_id"]]=True
        elif audio:
            lo=cfg["codec_base_offset"]; allowed[lo:lo+cfg["codec_layer_size"]]=True
            allowed[cfg["eos_token_id"]]=allowed[cfg["eot_token_id"]]=True
        else:
            allowed[cfg["text_token_offset"]:cfg["text_token_end"]]=True
            allowed[cfg["eos_token_id"]]=allowed[cfg["eot_token_id"]]=True
        return allowed,int(allowed.sum())>1

    @torch.no_grad()
    def forward(self,input_ids,positions,forward_batch,input_embeds=None,**kwargs):
        if forward_batch.batch_size!=1:
            raise ValueError("Correctness Bagpiper Engine accepts one request at a time")
        payload=json.loads(self.payload_path.read_text())
        expected=[frame[0] for frame in payload["sequence"]]
        if input_ids.tolist()!=expected:
            raise ValueError("SGLang/native full prompt mismatch; disable radix cache/chunking")
        hidden=self._hidden(payload,input_ids.device)
        raw=F.linear(hidden,self.lm_head.weight)[0]
        allowed,primary_active=self._allowed(raw,payload["phase"],payload["mode"],0)
        logits=raw.masked_fill(~allowed,-torch.inf)[None]
        secondary,stream_logps,masks,contracts=[],[],[],[]
        for stream in range(1,self.geometry["num_stream"]):
            support,active=self._allowed(raw,payload["phase"],payload["mode"],stream)
            value=0; lp=0.0
            if active:
                row=F.linear(hidden+self.stream_emb.weight[stream],self.lm_head.weight)[0].masked_fill(~support,-torch.inf)
                probs=torch.softmax(row.float(),-1)
                value=int(torch.multinomial(probs,1)); lp=float(torch.log_softmax(row.float(),-1)[value])
            secondary.append(value); stream_logps.append(lp); masks.append(active)
            contracts.append({"temperature":1.0,"top_k":0,"active":active})
        trace={"secondary_ids":secondary,"secondary_log_probs":stream_logps,"secondary_mask":masks,"primary_active":primary_active,"sampling_contracts":[{"temperature":1.0,"top_k":0,"active":primary_active},*contracts],"phase":payload["phase"],"native_sequence_length":len(expected)}
        return LogitsProcessorOutput(next_token_logits=logits,customized_info={"espnet_token_trace":[trace]})


EntryClass=ESPnetBagpiperForCausalLM
