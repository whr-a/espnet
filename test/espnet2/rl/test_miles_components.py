"""Rank RPC collection must await peers before propagating a failure."""

import asyncio
from types import SimpleNamespace

import pytest

from espnet2.rl.miles_components import NativeTrainerGroup


@pytest.mark.parametrize("failing_rank", [None, 0, 1])
def test_rank_collection_preserves_order_and_awaits_delayed_peer(failing_rank):
    async def execute():
        failure_seen = asyncio.Event()
        peer_finished = asyncio.Event()
        failure = RuntimeError("Synthetic rank RPC failure")

        async def reply(rank, value, *, offset):
            if rank == failing_rank:
                failure_seen.set()
                raise failure
            if failing_rank is not None:
                await failure_seen.wait()
                await asyncio.sleep(0)
            peer_finished.set()
            return rank + value + offset

        workers = [
            SimpleNamespace(
                operation=SimpleNamespace(
                    remote=lambda value, offset, rank=rank: reply(
                        rank, value, offset=offset
                    )
                )
            )
            for rank in range(2)
        ]
        group = NativeTrainerGroup(workers)
        if failing_rank is None:
            assert await group.call("operation", 4, offset=3) == [7, 8]
        else:
            with pytest.raises(RuntimeError) as caught:
                await group.call("operation", 4, offset=3)
            assert caught.value is failure
            assert peer_finished.is_set()

    asyncio.run(execute())
