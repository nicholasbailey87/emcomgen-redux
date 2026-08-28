"""
`train.build_lr_schedule` and the `[scheduler]` surface `parse_config` validates
it against.

Two invariants, and the second is the reason the first is safe:

    * **The warm-up ascends from zero to the base rate, whatever else is set.**
    * **The shape that follows opens at the base rate**, so the handover is
      continuous.

Both were false until 2026-08-28. `PASS` applies one floor across the whole run
-- `min_lr + (base_lr - min_lr) * multiplier` with
`min_lr = base_lr * cool_point_multiplier` -- so passing the configured floor
straight to it opened the warm-up at `floor * base_lr` rather than at zero, and
at the `cool_point_multiplier = 1.0` every rung inherited it pinned every step
to `base_lr` and the schedule did nothing whatsoever. `d5c47f5` set that floor
deliberately alongside `warm_up_epochs = 0`; `b298da5` re-enabled the warm-up
without touching it, and the ten-epoch ramp it advertised ran on no rung.

Not for want of being written down -- docs/training.md had carried "`cool_point_
multiplier = 1.0` makes `lr_schedule_shape` a complete no-op" the whole time,
while DEFAULT.toml's comment beside the key described the ramp as though it ran.
A prose warning in one file did not stop the config in another from claiming the
opposite, which is the argument for asserting it here instead.

The second invariant is held by the config surface rather than by arithmetic.
`lr_schedule_shape` takes an intention -- `flat` or `cosine` -- not a
`gradboard.cycles.FN_LIBRARY` curve, because two of those curves (`ascent`,
`triangle`) open at their *trough*: a warm-up in front of one discarded its rate
at the handover and re-climbed it over the remaining epochs, with nothing in the
config saying so. The sentinels name only curves that open at their peak, so the
discontinuity is unreachable rather than merely unused. `test_the_shape_table_
only_admits_shapes_that_open_at_their_peak` is what keeps that true if the table
grows.

Nothing about any of this raised. The runs completed and the metrics looked
ordinary while the config said one thing and the optimiser did another, so these
tests assert the learning rate a real `PASS` reports at real step counts rather
than the multiplier a cycle returns in isolation.
"""

import _bootstrap  # noqa: F401

import pytest
import torch
from gradboard.cycles import FN_LIBRARY
from gradboard.scheduler import PASS

import parse_config
import train

from _bootstrap import all_rungs, rung

RUNGS = all_rungs()

BASE_LR = 1e-4
TRAINING_EXAMPLES = 5000
BATCH_SIZE = 32

# Comfortably below 1.0 and not a round fraction of it, so a rate that landed on
#     the floor by some other route would not look like a pass.
FLOOR = 1 / 60


def _config(epochs, warm_up_epochs, shape="flat", floor=None):
    """A `[scheduler]` block standing on its own, with no rung behind it."""
    scheduler = {
        "epochs": epochs,
        "warm_up_epochs": warm_up_epochs,
        "lr_schedule_shape": shape,
        "range_test": False,
    }
    if floor is not None:
        scheduler["cool_point_multiplier"] = floor
    return {"scheduler": scheduler}


def _rates(config, at_epochs):
    """
    The learning rate a real `PASS` puts on a parameter group at each of
    `at_epochs`, as a fraction of the base rate.

    Driven through `PASS` rather than through the schedule object because the
    defect being guarded against lived in `update_learning_rates`, not in the
    cycles: the multiplier moved correctly throughout and was multiplied by
    zero.
    """
    schedule = train.build_lr_schedule(config, TRAINING_EXAMPLES, BATCH_SIZE)
    optimiser = torch.optim.AdamW(
        [{"params": [torch.zeros(1, requires_grad=True)], "lr": BASE_LR}]
    )
    scheduler = PASS(
        schedule,
        torch.nn.Linear(1, 1),
        optimiser,
        range_test=False,
        cool_point_multiplier=0.0,
    )

    steps_per_epoch = len(schedule) / config["scheduler"]["epochs"]

    out = []
    for epoch in at_epochs:
        scheduler.step_count = min(
            int(round(epoch * steps_per_epoch)), len(schedule) - 1
        )
        scheduler.update_learning_rates()
        out.append(optimiser.param_groups[0]["lr"] / BASE_LR)
    return out


def _shapes():
    """Every sentinel, with a floor where the sentinel needs one."""
    return [
        (name, FLOOR if takes_floor else None)
        for name, (_, takes_floor) in parse_config.LR_SCHEDULE_SHAPES.items()
    ]


# --------------------------------------------------------------------------
# The warm-up
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape,floor", _shapes())
def test_the_warm_up_starts_at_zero(shape, floor):
    """The claim the coupled floor broke, asserted for every shape."""
    opening, = _rates(_config(100, 10, shape, floor), [0])
    assert opening == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("shape,floor", _shapes())
def test_the_warm_up_reaches_the_base_rate_by_its_last_epoch(shape, floor):
    """
    And it must arrive. A ramp that opened at zero and never got there would
        satisfy the test above while starving the run.
    """
    late, = _rates(_config(100, 10, shape, floor), [9.99])
    assert late == pytest.approx(1.0, abs=0.02)


@pytest.mark.parametrize("shape,floor", _shapes())
def test_the_warm_up_is_monotone_over_its_own_epochs(shape, floor):
    rates = _rates(_config(100, 10, shape, floor), [0, 2, 4, 6, 8, 9.99])
    assert rates == sorted(rates)


def test_a_run_shorter_than_its_warm_up_is_all_ramp():
    """`train.py` takes `min(warm_up_epochs, epochs)`, so this is not an error."""
    opening, end = _rates(_config(5, 10), [0, 4.99])
    assert opening == pytest.approx(0.0, abs=1e-9)
    assert end == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("shape,floor", _shapes())
def test_no_warm_up_opens_at_the_base_rate(shape, floor):
    """
    `warm_up_epochs = 0` builds one cycle, and every admissible shape opens at
        its peak -- so a run without a warm-up starts at the configured rates
        rather than climbing to them. Rung 10 relies on this.
    """
    opening, = _rates(_config(100, 0, shape, floor), [0])
    assert opening == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------
# The handover, and the shape after it
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape,floor", _shapes())
def test_the_handover_is_continuous(shape, floor):
    """
    The second invariant. The warm-up ends at base and the next cycle opens at
        base, so nothing is discarded at the seam. This is what the sentinel
        surface buys: with `ascent` -- reachable before, and the shape every rung
        was configured with -- the rate fell from 0.999 of base to the floor in
        one step and did not recover for eighty epochs.
    """
    before, after = _rates(_config(100, 10, shape, floor), [9.99, 10.01])
    assert before == pytest.approx(1.0, abs=0.02)
    assert after == pytest.approx(1.0, abs=1e-3)


def test_every_descending_shape_opens_at_its_peak():
    """
    The invariant above, asserted against the table rather than through it, so
        that adding a sentinel for a trough-opening curve fails here rather than
        quietly reintroducing the discontinuity on some future rung.

    Scoped to the shapes that take a floor, because those are the ones whose
        curve is actually traced. A shape that takes no floor is built with
        `low == high` and is constant whatever its generating function does --
        which is why `flat` can map to `ascent`, a curve that opens at its
        trough, and still hand over continuously. If `flat` ever grew a floor
        this test would start failing, which is the point.
    """
    descending = {
        name: curve
        for name, (curve, takes_floor) in parse_config.LR_SCHEDULE_SHAPES.items()
        if takes_floor
    }

    assert descending, "no shape descends; the floor has nothing to govern"

    for name, curve in descending.items():
        assert FN_LIBRARY[curve](0, 100) == pytest.approx(1.0), (
            f"`{name}` maps to `{curve}`, which opens at its trough -- a warm-up "
            "in front of it would be discarded at the handover"
        )


def test_a_shape_that_takes_no_floor_is_pinned_flat_whatever_its_curve():
    """
    The exemption above, stated as a property rather than left implicit: a
        floorless shape is built with `low == high == 1.0`, so its curve is
        unreachable. This is what makes `flat`'s mapping to `ascent` a detail
        rather than a latent discontinuity.
    """
    for name, (_, takes_floor) in parse_config.LR_SCHEDULE_SHAPES.items():
        if takes_floor:
            continue

        schedule = train.build_lr_schedule(
            _config(100, 0, name), TRAINING_EXAMPLES, BATCH_SIZE
        )
        cycle, = schedule.cycles
        assert cycle.low == cycle.high == 1.0, name


def test_flat_holds_the_base_rate_for_the_whole_run():
    """
    `flat` means no cooling: the post-warm-up cycle sits at base throughout,
        which is what every trace in this repo was actually run under.
    """
    rates = _rates(_config(100, 10), [11, 30, 60, 99])
    assert rates == pytest.approx([1.0] * 4, abs=1e-9)


def test_cosine_descends_to_the_floor_and_stops_there():
    """
    A decay must bottom out at `cool_point_multiplier`, not at zero. That is the
        floor's only remaining job.
    """
    rates = _rates(_config(100, 10, "cosine", FLOOR), [10.01, 55, 99])
    assert rates[0] == pytest.approx(1.0, abs=1e-3)
    assert rates[2] == pytest.approx(FLOOR, abs=0.02)
    assert rates[0] > rates[1] > rates[2]


def test_cosine_is_the_falling_half_and_does_not_come_back_up():
    """
    `FN_LIBRARY`'s own `cosine` is a full period and ends where it began, which
        would leave a run at its opening rate. The sentinel maps to
        `half_cosine` for that reason, and this is the assertion that says so in
        terms of the rate rather than the mapping.
    """
    rates = _rates(_config(100, 10, "cosine", FLOOR), [20, 40, 60, 80, 99])
    assert rates == sorted(rates, reverse=True)


# --------------------------------------------------------------------------
# What the config is allowed to say
# --------------------------------------------------------------------------


def _validate(**scheduler):
    """`validate_config` over a whole config, with `[scheduler]` overridden."""
    config = parse_config.get_config(rung(RUNGS[0]))
    config["scheduler"].update(scheduler)
    for key, value in list(config["scheduler"].items()):
        if value is None:
            del config["scheduler"][key]
    return parse_config.validate_config(config)


@pytest.mark.parametrize("shape", ["ascent", "triangle", "half_cosine", "linear", ""])
def test_a_gradboard_curve_name_is_not_an_accepted_shape(shape):
    """
    Including the curves that would in fact work. The surface is intentions, and
        a config naming `half_cosine` is a config written against the old
        contract -- which also named `ascent` and meant something else.
    """
    with pytest.raises(parse_config.InvalidConfig):
        _validate(lr_schedule_shape=shape, cool_point_multiplier=None)


def test_a_floor_beside_a_flat_schedule_is_rejected():
    """
    Rejected rather than ignored. A scheduler key that looks set and is never
        read is exactly how the warm-up came to run on no rung.
    """
    with pytest.raises(parse_config.InvalidConfig):
        _validate(lr_schedule_shape="flat", cool_point_multiplier=0.5)


def test_a_cosine_without_a_floor_is_rejected():
    with pytest.raises(parse_config.InvalidConfig):
        _validate(lr_schedule_shape="cosine", cool_point_multiplier=None)


@pytest.mark.parametrize("floor", [-0.1, 1.0, 1.5, "0.5", True])
def test_a_floor_outside_zero_to_one_is_rejected(floor):
    """
    1.0 is excluded deliberately: it is a flat schedule spelled as a decay, and
        spelling it that way is what made the old surface unreadable.
    """
    with pytest.raises(parse_config.InvalidConfig):
        _validate(lr_schedule_shape="cosine", cool_point_multiplier=floor)


@pytest.mark.parametrize("warm_up", [-1, 1.5, "10", None])
def test_a_warm_up_that_is_not_a_non_negative_integer_is_rejected(warm_up):
    with pytest.raises(parse_config.InvalidConfig):
        _validate(warm_up_epochs=warm_up, cool_point_multiplier=None)


def test_a_flat_schedule_with_no_floor_is_accepted():
    """The default every rung inherits."""
    _validate(lr_schedule_shape="flat", warm_up_epochs=10, cool_point_multiplier=None)


def test_a_cosine_with_a_floor_is_accepted():
    _validate(
        lr_schedule_shape="cosine", warm_up_epochs=10, cool_point_multiplier=FLOOR
    )


# --------------------------------------------------------------------------
# The real ladder
# --------------------------------------------------------------------------


@pytest.mark.parametrize("config_file", RUNGS)
def test_every_rung_opens_where_its_warm_up_says(config_file):
    config = parse_config.get_config(rung(config_file))
    opening, = _rates(config, [0])

    if config["scheduler"]["warm_up_epochs"]:
        assert opening == pytest.approx(0.0, abs=1e-9)
    else:
        assert opening == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("config_file", RUNGS)
def test_no_rung_reaches_the_scheduler_with_a_schedule_pinned_flat(config_file):
    """
    The regression itself. A rung that asks for a warm-up must see its rate
        move; before this change every one of them ran constant because the
        floor arrived at `PASS` instead of at the cycle.
    """
    config = parse_config.get_config(rung(config_file))
    epochs = config["scheduler"]["epochs"]
    rates = _rates(config, [0, epochs * 0.05, epochs * 0.5, epochs - 0.01])

    if config["scheduler"]["warm_up_epochs"]:
        assert len(set(round(r, 6) for r in rates)) > 1, (
            "the schedule is constant despite a configured warm-up"
        )
