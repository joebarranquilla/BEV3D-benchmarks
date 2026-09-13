from pathlib import Path


def test_bevformer_test_script_does_not_block_single_gpu_execution():
    test_script = Path("third_party/BEVFormer/tools/test.py")
    content = test_script.read_text(encoding="utf-8")

    assert "assert False" not in content
    assert "single_gpu_test" in content
