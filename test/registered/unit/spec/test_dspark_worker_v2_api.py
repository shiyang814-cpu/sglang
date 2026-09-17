import inspect

from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    DSparkWorkerV2,
)


def test_forward_batch_generation_accepts_pp_proxy_tensors():
    parameters = inspect.signature(
        DSparkWorkerV2.forward_batch_generation
    ).parameters
    assert "pp_proxy_tensors" in parameters
