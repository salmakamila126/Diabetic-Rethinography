"""Fixed-shape synchronous CPU FP32 backend candidates."""

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import torch

BACKENDS = ("eager", "torchscript", "onnxruntime", "openvino")


@dataclass
class Backend:
    infer: object
    artifact: Path
    settings: dict


def capabilities():
    requirements = {"eager": (), "torchscript": (), "onnxruntime": ("onnx", "onnxruntime"),
                    "openvino": ("onnx", "openvino")}
    return {name: {"status": "CANDIDATE" if not missing else "DEPENDENCY_MISSING",
                   "missing_packages": missing}
            for name, packages in requirements.items()
            for missing in [[p for p in packages if find_spec(p) is None]]}


def export_onnx(model, inputs, output):
    import onnx

    names = ["pixels"] if len(inputs) == 1 else ["global_pixels", "patch_pixels"]
    path = output / "model.onnx"
    torch.onnx.export(model, inputs, path, input_names=names, output_names=["logits"],
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(str(path))
    return path, names


def build(name, model, inputs, output, *, threads, shared_onnx=None):
    if name not in BACKENDS or threads < 1:
        raise ValueError("Invalid CPU backend or thread count")
    missing = capabilities()[name]["missing_packages"]
    if missing:
        raise ImportError(", ".join(missing))
    if model.training or any(p.device.type != "cpu" or p.dtype != torch.float32
                             for p in model.parameters() if p.is_floating_point()):
        raise ValueError("Backend exports require an eval-mode CPU FP32 model")
    if any(x.device.type != "cpu" or x.dtype != torch.float32 for x in inputs):
        raise ValueError("Backend inputs must be CPU FP32")
    if name == "eager":
        path = output / "inference_state_dict.pt"
        torch.save(model.state_dict(), path)
        return Backend(model, path, {"runtime": "PyTorch eager", "threads": torch.get_num_threads()})
    if name == "torchscript":
        with torch.no_grad():
            traced = torch.jit.freeze(torch.jit.trace(model, inputs, check_inputs=[inputs]))
        path = output / "model.torchscript.pt"
        traced.save(str(path))
        return load(name, model, path, [], threads=threads)

    path, names = shared_onnx if shared_onnx is not None else export_onnx(model, inputs, output)
    if name == "openvino":
        import openvino as ov

        ir_path = output / "model.xml"
        ov.save_model(ov.Core().read_model(str(path)), str(ir_path), compress_to_fp16=False)
        path = ir_path
    return load(name, model, path, names, threads=threads)


def load(name, model, path, names, *, threads):
    """Open verified export artifacts without tracing or rebuilding them."""
    if name == "eager":
        return Backend(model, path, {"runtime": "PyTorch eager", "threads": torch.get_num_threads()})
    if name == "torchscript":
        reloaded = torch.jit.load(str(path), map_location="cpu").eval()
        return Backend(reloaded, path, {"runtime": "PyTorch TorchScript traced/frozen",
                                      "threads": torch.get_num_threads()})
    if name == "onnxruntime":
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads, opts.inter_op_num_threads = threads, 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise ValueError("Requested ORT CPU-only execution was not honored")

        def run(*tensors):
            feeds = {key: value.numpy() for key, value in zip(names, tensors, strict=True)}
            return session.run(["logits"], feeds)[0]

        return Backend(run, path, {"runtime": "ONNX Runtime", "provider": "CPUExecutionProvider",
                                   "opset": 17, "intra_op_threads": threads, "inter_op_threads": 1,
                                   "mode": "sequential", "graph_optimization": "all"})

    import openvino as ov

    if name != "openvino":
        raise ValueError(f"Unexpected backend: {name}")
    compiled = ov.Core().compile_model(str(path), "CPU", {"PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_NUM_THREADS": threads, "NUM_STREAMS": 1, "INFERENCE_PRECISION_HINT": "f32"})
    request = compiled.create_infer_request()
    ordered_ports = [compiled.input(name) for name in names]
    out = compiled.output("logits")
    if str(compiled.get_property("INFERENCE_PRECISION_HINT")) not in ("<Type: 'float32'>", "f32", "float32"):
        raise ValueError("Requested OpenVINO FP32 precision was not honored")

    def run(*tensors):
        inputs_by_port = {port: value.numpy() for port, value in zip(ordered_ports, tensors, strict=True)}
        return request.infer(inputs_by_port, share_outputs=False)[out]

    return Backend(run, path, {"runtime": "OpenVINO", "device": "CPU", "opset": 17,
                                  "precision_hint": "f32", "threads": threads,
                                  "streams": 1, "performance_hint": "LATENCY"})
