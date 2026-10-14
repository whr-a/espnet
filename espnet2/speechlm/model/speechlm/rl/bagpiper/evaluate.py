# Copyright 2026 Haoran Wang
# SPDX-License-Identifier: Apache-2.0
"""Held-out Bagpiper task metrics retaining actual SGLang generation evidence."""
import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path
from espnet2.rl.config import ESPnetRLConfig
from espnet2.rl.utils import resolve_callable


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True)
    parser.add_argument("--checkpoint",required=True); parser.add_argument("--output",required=True)
    args=parser.parse_args(); cfg=ESPnetRLConfig.load(args.config)
    directory=Path(args.output); directory.mkdir(parents=True,exist_ok=False)
    checkpoint=Path(args.checkpoint).resolve()
    # The final native checkpoint is immutable. A symlink retains its identity
    # without copying another 36 GB model into the evaluation directory.
    (directory/"native.pt").symlink_to(checkpoint)
    cfg.output_dir=str(directory)
    (directory/"frozen_config.json").write_text(json.dumps(cfg.to_dict(),indent=2)+"\n")
    requests=[json.loads(row) for row in Path(cfg.eval_manifest).read_text().splitlines() if row.strip()]
    (directory/"requests.json").write_text(json.dumps(requests,indent=2)+"\n")
    env=os.environ.copy(); env["CUDA_VISIBLE_DEVICES"]=cfg.rollout_gpus
    env["PYTHONPATH"]=str(Path(cfg.sglang_source)/"python")+os.pathsep+str(Path(__file__).resolve().parents[6])+os.pathsep+str(cfg.miles_source)
    export=[cfg.rollout_python,"-m","espnet2.rl.export_worker","--config",str(directory/"frozen_config.json"),"--checkpoint",str(checkpoint),"--output",str(directory/"hf")]
    with (directory/"export.log").open("w") as handle:subprocess.run(export,env=env,check=True,stdout=handle,stderr=subprocess.STDOUT)
    rollout=[cfg.rollout_python,"-m","espnet2.rl.rollout_worker","--config",str(directory/"frozen_config.json"),"--model",str(directory/"hf"),"--requests",str(directory/"requests.json"),"--output",str(directory/"responses.json"),"--step","-1"]
    with (directory/"generation.log").open("w") as handle:subprocess.run(rollout,env=env,check=True,stdout=handle,stderr=subprocess.STDOUT)
    generations=json.loads((directory/"responses.json").read_text())["generations"]
    reward=resolve_callable(cfg.reward_function)
    sources={item["id"]:item for item in requests}
    scored=[{"source_id":generation["source_id"],"reward":reward(sources[generation["source_id"]],generation),"text":generation["text"]} for generation in generations]
    values=[item["reward"] for item in scored]
    metric="wer" if requests[0]["task"]=="tts_asr_wer" else "choice_accuracy"
    result={"metric":metric,"value":(-sum(values) if metric=="wer" else sum(values))/len(values),"sample_count":len(values),"prompt_count":len(requests),"sampling":"unit temperature, full support, all group candidates","checkpoint":str(checkpoint),"dataset":requests[0]["dataset"],"generations":scored}
    (directory/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":main()
