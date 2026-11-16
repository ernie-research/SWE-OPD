import argparse
import concurrent.futures
import json
import os
import pickle
import re
import shutil
import time


import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp
from transformers import AutoConfig
from typing_extensions import override

from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf, remove_padding


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


pickle.Unpickler = UnpicklerWrapper


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k or "_state" in k:
                continue
            print(f"find {k} in torch_dist ckpt")
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def get_expert_param(args, name, param):
    if ".experts." not in name:
        yield name, param
        return

    num_experts = args.num_experts
    match = re.search(r"mlp.experts\.(.+)\.weight(\d+)", name)
    if not match:
        assert param.shape[0] == num_experts
        for expert_id in range(num_experts):
            expert_name = name.replace(".experts.experts.", ".experts.") + str(expert_id)
            expert_param = param[expert_id]
            yield expert_name, expert_param
    else:
        yield name, param


def get_layer_param(args, name, param):
    if ".layers." not in name:
        yield name, param
        return

    num_layers = args.num_layers
    match = re.search(r"\.layers\.(\d+)\.", name)
    if not match:
        assert param.shape[0] == num_layers
        for layer_id in range(num_layers):
            layer_name = name.replace(".layers.", f".layers.{layer_id}.")
            layer_param = param[layer_id]
            yield from get_expert_param(args, layer_name, layer_param)
    else:
        yield from get_expert_param(args, name, param)


def get_named_params(args, state_dict):
    for name, param in state_dict.items():
        name = f"module.module.{name}"
        yield from get_layer_param(args, name, param)


def save_tensors(
    args,
    model_name,
    state_dict,
    output_dir,
    chunk_size,
    vocab_size=None,
    origin_hf_dir=None,
    num_shards=None,
    save_workers=1,
):
    print(f"start saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    total_size = 0
    converted_tensors = []
    converted_names = set()
    for name, param in get_named_params(args, state_dict):
        if vocab_size:
            param = remove_padding(name, param, vocab_size)
        converted_named_tensors = convert_to_hf(args, model_name, name, param)
        for converted_name, converted_param in converted_named_tensors:
            converted_names.add(converted_name)
            tensor_size = converted_param.numel() * converted_param.element_size()
            converted_tensors.append((converted_name, converted_param, tensor_size))
            total_size += tensor_size

    if origin_hf_dir is not None:
        safetensors_files = [f for f in os.listdir(origin_hf_dir) if f.endswith(".safetensors")]
        for filename in safetensors_files:
            with safetensors.safe_open(os.path.join(origin_hf_dir, filename), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k not in converted_names:
                        converted_name = k
                        print(f"add {k} from origin hf checkpoint")
                        converted_param = f.get_tensor(k)
                        converted_names.add(k)
                        tensor_size = converted_param.numel() * converted_param.element_size()
                        converted_tensors.append((converted_name, converted_param, tensor_size))
                        total_size += tensor_size

    if num_shards is not None:
        if num_shards <= 0:
            raise ValueError(f"num_shards must be positive, got {num_shards}")
        if num_shards > len(converted_tensors):
            raise ValueError(
                f"cannot create {num_shards} non-empty shards from only "
                f"{len(converted_tensors)} tensors"
            )

        # Assign larger tensors first to the currently lightest shard. This
        # produces exactly num_shards non-empty, size-balanced files without
        # splitting any tensor. Tensor placement has no effect on HF semantics.
        modeltensors = [{} for _ in range(num_shards)]
        shard_sizes = [0] * num_shards
        for converted_name, converted_param, tensor_size in sorted(
            converted_tensors, key=lambda item: item[2], reverse=True
        ):
            shard_id = min(range(num_shards), key=shard_sizes.__getitem__)
            modeltensors[shard_id][converted_name] = converted_param
            shard_sizes[shard_id] += tensor_size
        assert all(modeltensors), "exact shard assignment unexpectedly produced an empty shard"
        print(
            f"assigned {len(converted_tensors)} tensors to exactly {num_shards} shards; "
            f"size range=[{min(shard_sizes)}, {max(shard_sizes)}] bytes"
        )
    else:
        modeltensors = [{}]
        current_size = 0
        for converted_name, converted_param, tensor_size in converted_tensors:
            if modeltensors[-1] and tensor_size + current_size > chunk_size:
                modeltensors.append({})
                current_size = 0
            modeltensors[-1][converted_name] = converted_param
            current_size += tensor_size

    metadata = {"metadata": {"total_size": total_size}, "weight_map": {}}

    num_files = len(modeltensors)
    for i, tensors in enumerate(modeltensors):
        filename = f"model-{i:05d}-of-{num_files:05d}.safetensors"
        for key in tensors.keys():
            metadata["weight_map"][key] = filename
    index_filepath = os.path.join(output_dir, "model.safetensors.index.json")
    json.dump(metadata, open(index_filepath, "w"), indent=2)
    print(f"{index_filepath} saved.")

    def save_shard(item):
        i, tensors = item
        filename = f"model-{i:05d}-of-{num_files:05d}.safetensors"
        t = time.time()
        filepath = os.path.join(output_dir, filename)
        safetensors.torch.save_file(tensors, filepath)
        return filename, time.time() - t

    if save_workers <= 0:
        raise ValueError(f"save_workers must be positive, got {save_workers}")
    if save_workers == 1:
        for item in enumerate(modeltensors):
            filename, elapsed = save_shard(item)
            print(f"{filename} saved in {elapsed:.2f} sec.", flush=True)
    else:
        workers = min(save_workers, num_files)
        print(f"saving {num_files} shards with {workers} parallel workers", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(save_shard, item) for item in enumerate(modeltensors)]
            for future in concurrent.futures.as_completed(futures):
                filename, elapsed = future.result()
                print(f"{filename} saved in {elapsed:.2f} sec.", flush=True)


def copy_assets(origin_hf_dir, output_dir):
    for filename in os.listdir(origin_hf_dir):
        if filename == "model.safetensors.index.json" or filename.endswith(".safetensors"):
            continue
        origin_filename = os.path.join(origin_hf_dir, filename)
        if not os.path.isfile(origin_filename):
            print(f"Skip {filename}, not a file.")
            continue
        src, dst = origin_filename, os.path.join(output_dir, filename)
        print(f"copy from {src} to {dst}")
        shutil.copy(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        default=None,
        help="use the origin hf dir to copy files like tokenizer, config.json, etc.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    parser.add_argument(
        "-a", "--add-missing-from-origin-hf", action="store_true", help="Add missing weights from origin hf checkpoint"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5 * 1024**3,
        help="Chunk size for saving tensors, default is 2GB.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Write exactly this many non-empty, size-balanced safetensors shards. Overrides --chunk-size.",
    )
    parser.add_argument(
        "--save-workers",
        type=int,
        default=1,
        help="Number of safetensors shards to write concurrently (default: 1).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocab size for removing padding, if applicable. If not provided, no padding will be removed.",
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    if args.model_name is None and args.origin_hf_dir is None:
        raise ValueError(
            "Either --model-name or --origin-hf-dir must be provided, so that we can know the name of the params."
        )

    if args.model_name is None:
        hf_config = AutoConfig.from_pretrained(args.origin_hf_dir, trust_remote_code=True)
        args.model_name = type(hf_config).__name__.lower()

    state_dict = {}
    print(f"loading model from {args.input_dir}")
    t = time.time()
    megatron_args = torch.load(os.path.join(args.input_dir, "common.pt"), weights_only=False)["args"]
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(args.input_dir),
        planner=EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    print(f"model loaded in {time.time()-t:.2f} sec.")

    save_tensors(
        megatron_args,
        args.model_name,
        state_dict,
        args.output_dir,
        args.chunk_size,
        args.vocab_size,
        args.origin_hf_dir if args.add_missing_from_origin_hf else None,
        args.num_shards,
        args.save_workers,
    )

    if args.origin_hf_dir:
        copy_assets(args.origin_hf_dir, args.output_dir)
