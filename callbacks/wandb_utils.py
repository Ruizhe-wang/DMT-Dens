def safe_wandb_step(trainer, run) -> int:
    """Return a W&B log step that never goes backwards.

    Lightning can advance the WandbLogger experiment step before callback hooks
    manually call ``run.log``. In that case logging with ``trainer.global_step``
    produces W&B warnings and the payload is ignored.
    """

    trainer_step = int(getattr(trainer, "global_step", 0) or 0)
    run_step = getattr(run, "step", None)
    if run_step is None:
        return trainer_step
    try:
        return max(trainer_step, int(run_step))
    except (TypeError, ValueError):
        return trainer_step
