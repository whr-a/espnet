"""Preserve completed optimizer state after a shared pre-update failure.

Call on all live learner ranks before update_actor starts: either after the
shared prepared-batch error or through the synchronous driver's failure RPC.
This cannot recover dead ranks, broken process groups, CUDA failure or a
partial optimizer step.
"""

import json
import math
import os
from pathlib import Path

import torch.distributed as distributed


def _all_ranks(actor, value):
    if actor._world_size == 1:
        return [value]
    values = [None] * actor._world_size
    distributed.all_gather_object(
        values, value, group=actor.trainer._object_collective_group()
    )
    return values


def preserve_preupdate_failure(
    actor,
    rollout_id,
    original_error,
    *,
    failure_stage="prepare_batch_before_update",
):
    """Finish the same collective sequence on every live rank before raising.

    Checkpoint failures are secondary records. The caller must still raise the
    original prepared-batch error, and the driver must await every train RPC
    before Miles can dispose the learner actors.
    """
    trainer = actor.trainer
    step = actor.version
    step_label = f"{step:06d}" if type(step) is int else "invalid"
    rollout_label = f"{rollout_id:06d}" if type(rollout_id) is int else "invalid"
    directory = (
        Path(actor.config.output_dir)
        / "failure-checkpoints"
        / (f"step-{step_label}-before-rollout-{rollout_label}")
    )
    checkpoint = directory / "full-state.pt"
    local = {"rank": actor._rank, "error": None}
    reserved = False
    try:
        if not (
            type(step) is int
            and type(rollout_id) is int
            and step == rollout_id == trainer.completed_updates
            and trainer.rank == actor._rank
            and trainer.world_size == actor._world_size
        ):
            raise ValueError(
                "Failure checkpoint requires one completed pre-update state"
            )
        if failure_stage not in (
            "prepare_batch_before_update",
            "rollout_generation_before_update",
        ):
            raise ValueError("Unknown clean pre-update failure stage")
        if failure_stage == "rollout_generation_before_update" and (
            actor.config.execution_mode not in ("sync", "disagg_sync")
            or actor.config.training_objective != "west_grpo"
        ):
            raise ValueError("Driver failure preservation requires synchronous WEST")
        rates = [float(group["lr"]) for group in trainer.optimizer.param_groups]
        if not rates or any(not math.isfinite(rate) or rate < 0 for rate in rates):
            raise ValueError("Invalid next-update learning rates")
        epoch = trainer.scheduler.state_dict()["last_epoch"]
        if epoch != step:
            raise ValueError("Scheduler cursor differs from completed updates")
        local.update(
            step=step,
            rollout_id=rollout_id,
            next_learning_rates=rates,
            scheduler_last_epoch=epoch,
            original_error=original_error,
            failure_stage=failure_stage,
        )
        # Reserve an immutable destination once; never overwrite an earlier
        # successful checkpoint or an earlier failed preservation attempt.
        if actor._rank == 0:
            directory.mkdir(parents=True, exist_ok=False)
            reserved = True
    except Exception as error:
        local["error"] = f"{type(error).__name__}: {error}"

    admission = _all_ranks(actor, local)
    ready = (
        sorted(value["rank"] for value in admission) == list(range(actor._world_size))
        and all(value["error"] is None for value in admission)
        and all(
            (
                value["step"],
                value["rollout_id"],
                value["next_learning_rates"],
                value["scheduler_last_epoch"],
                value["original_error"],
                value["failure_stage"],
            )
            == (
                admission[0]["step"],
                admission[0]["rollout_id"],
                admission[0]["next_learning_rates"],
                admission[0]["scheduler_last_epoch"],
                admission[0]["original_error"],
                admission[0]["failure_stage"],
            )
            for value in admission
        )
    )
    saved = {"rank": actor._rank, "error": None}
    if ready:
        try:
            # The existing trainer API gathers each rank's RNG and persists
            # native weights, owner Adam moments/FP32 masters, scheduler and
            # absolute seeded data cursor. It does not step either optimizer
            # or scheduler. Do not substitute a rollout model-weight export.
            trainer.save_training_checkpoint(
                checkpoint,
                step,
                extra_state={
                    "failure_stage": failure_stage,
                    "failed_rollout_id": rollout_id,
                    "original_error": original_error,
                },
            )
        except Exception as error:
            saved["error"] = f"{type(error).__name__}: {error}"
    else:
        saved["error"] = "Ranks did not admit a consistent clean pre-update checkpoint"
    statuses = _all_ranks(actor, saved)
    succeeded = ready and all(value["error"] is None for value in statuses)
    result = {
        "status": "Saved" if succeeded else "NotSaved",
        "original_error": original_error,
        "failure_stage": failure_stage,
        "failed_rollout_id": rollout_id,
        "completed_updates": step,
        "next_update": step,
        "next_optimizer_update_number": step + 1 if type(step) is int else None,
        "checkpoint": str(checkpoint),
        "rank_admission": admission,
        "rank_save_status": statuses,
        "optimizer_or_scheduler_stepped": False,
        "resume_requires_independent_checkpoint_validation": True,
    }
    receipt = {"rank": actor._rank, "error": None}
    if actor._rank == 0 and reserved:
        try:
            if checkpoint.exists():
                result["checkpoint_size_bytes"] = checkpoint.stat().st_size
            temporary = directory / "preservation.json.tmp"
            with temporary.open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(directory / "preservation.json")
        except Exception as error:
            receipt["error"] = f"{type(error).__name__}: {error}"
    # Every learner finishes checkpoint status and receipt status before its
    # train RPC raises. The driver's all-RPC await is the cleanup boundary.
    result["rank_receipt_status"] = _all_ranks(actor, receipt)
    return result
