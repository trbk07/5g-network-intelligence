"""Interpret stopping metadata without fitting or changing historical checkpoints."""


def stopping_status(training,config):
    """Distinguish a patience stop from epoch-cap and wall-time exhaustion."""
    best,completed=training["best_epoch"],training["epochs_completed"]
    maximum,patience=config["max_epochs"],config["early_stopping_patience"]
    if (any(type(x) is not int for x in (best,completed,maximum,patience))
            or not 0<=best<completed<=maximum or patience<1):
        raise ValueError("Valid completed training/stopping metadata required")
    stale=completed-1-best
    early=stale>=patience
    epoch_cap=completed==maximum
    time_budget=bool(training.get("budget_exhausted",False))
    reason=("early_stopping" if early else "time_budget" if time_budget else
            "epoch_cap" if epoch_cap else "incomplete")
    return {"termination_reason":reason,"early_stopping_completed":early,
            "epoch_cap_reached":epoch_cap,"stale_epochs":stale}
