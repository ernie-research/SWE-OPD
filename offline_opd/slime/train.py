import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action


def _rollout_data_prefetch_enabled(args) -> bool:
    """Validate and return whether one-step offline data prefetch is enabled."""
    if not getattr(args, "prefetch_rollout_data", False):
        return False

    if args.load_debug_rollout_data is None:
        raise ValueError(
            "--prefetch-rollout-data is only supported with --load-debug-rollout-data. "
            "Prefetching online rollout would generate the next batch with stale actor weights."
        )

    incompatible = []
    if args.offload_rollout:
        incompatible.append("--offload-rollout")
    if args.rollout_global_dataset:
        incompatible.append("rollout global dataset")
    if args.eval_interval is not None:
        incompatible.append("--eval-interval")
    if incompatible:
        raise ValueError(
            "--prefetch-rollout-data is incompatible with " + ", ".join(incompatible)
        )

    return True


def _load_rollout_data_and_prefetch_next(
    rollout_manager,
    rollout_id: int,
    num_rollout: int,
    pending_generate,
    *,
    prefetch: bool,
):
    """Load one rollout and optionally keep exactly one subsequent load in flight."""
    if pending_generate is None:
        generate_ref = rollout_manager.generate.remote(rollout_id)
    else:
        pending_rollout_id, generate_ref = pending_generate
        if pending_rollout_id != rollout_id:
            raise RuntimeError(
                f"prefetched rollout id mismatch: expected {rollout_id}, got {pending_rollout_id}"
            )

    rollout_data_ref = ray.get(generate_ref)

    pending_generate = None
    next_rollout_id = rollout_id + 1
    if prefetch and next_rollout_id < num_rollout:
        pending_generate = (
            next_rollout_id,
            rollout_manager.generate.remote(next_rollout_id),
        )

    return rollout_data_ref, pending_generate


def train(args):
    configure_logger()
    release_train = args.release_train
    prefetch_rollout_data = _rollout_data_prefetch_enabled(args)

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout and not release_train:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    # train loop. In offline replay mode, submit the next disk load immediately
    # after the current one is available so it overlaps actor training.
    pending_generate = None
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id))

        rollout_data_ref, pending_generate = _load_rollout_data_and_prefetch_next(
            rollout_manager,
            rollout_id,
            args.num_rollout,
            pending_generate,
            prefetch=prefetch_rollout_data,
        )

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            force_sync = release_train or rollout_id == args.num_rollout - 1
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=force_sync)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        offload_train(actor_trains)
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
