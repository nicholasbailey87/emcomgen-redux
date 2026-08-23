"""
Puts `code/` on `sys.path`. Imported by `conftest.py` for pytest runs, and by
each test module directly so `python tests/test_<name>.py` still works.
"""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code")
)

from parse_config import get_config  # noqa: E402


CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "experiments",
    "ablation",
    "configs",
)


def rung(config_file):
    return os.path.join(CONFIG_DIR, config_file)


def config_section(section, config_file=None, **overrides):
    """Omit `config_file` for `DEFAULT.toml` alone."""
    settings = dict(get_config(config_file)[section])
    settings.update(overrides)
    return settings


# The listener is two modules now -- a language model and a discriminator, each
#     named in `[receiver]` and configured from its own table. Most of what the
#     tests below assert is a property of the *pair* (the score cannot see the
#     referents' magnitude, no stage can read their ordering, the residual
#     stream does not grow), so they compose the two the way `Receiver` does and
#     hand the result referent embeddings and message embeddings directly.
#     `Receiver` itself takes images, which these tests have none of.

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from models import receiver as _receiver  # noqa: E402


class Listener(nn.Module):
    """
    `Receiver` from the referent embeddings inwards: one dropout, then the two
        slots, in that order.

    `dropout` defaults to 0.0 rather than to DEFAULT's 0.1 because almost every
        property under test is deterministic, and a test that has to remember
        to call `.eval()` is a test that will one day forget.
    """

    def __init__(self, language_model, discriminator, dropout=0.0):
        super().__init__()
        self.language_model = language_model
        self.discriminator = discriminator
        self.input_dropout = nn.Dropout(p=dropout)

        # Forwarded so a test can size its inputs without knowing which slot
        #     holds which width.
        self.referent_embedding_size = discriminator.referent_embedding_size
        self.token_embedding_size = language_model.token_embedding_size
        if hasattr(language_model, "message_length"):
            self.message_length = language_model.message_length

    def forward(self, referents, messages):
        referents = self.input_dropout(referents)
        return self.discriminator(
            referents, self.language_model(messages, referents)
        )


def build_listener(
    language_model,
    discriminator,
    referent_dim,
    config_file=None,
    dropout=0.0,
    seed=0,
    language_model_overrides=None,
    discriminator_overrides=None,
):
    """
    Compose one of the four legal slot pairings from a config.

    Args:
        language_model: class name in `code/models/receiver.py`
        discriminator: class name in `code/models/receiver.py`
        referent_dim: what the vision model would have produced
        config_file: a rung, or None for `DEFAULT.toml` alone
        dropout: `[receiver] dropout`, defaulting to off
        seed: set immediately before each slot is built, so two builds of the
            same pairing are identical
    """
    language_model_class = getattr(_receiver, language_model)
    discriminator_class = getattr(_receiver, discriminator)

    language_model_settings = config_section(
        "receiver_language_model", config_file, **(language_model_overrides or {})
    )
    discriminator_settings = config_section(
        "receiver_discriminator", config_file, **(discriminator_overrides or {})
    )

    torch.manual_seed(seed)
    built_language_model = language_model_class(
        referent_dim, **language_model_settings
    )

    torch.manual_seed(seed)
    built_discriminator = discriminator_class(
        referent_dim,
        # Sized from the language model, exactly as `build_models` does it.
        built_language_model.output_size,
        **discriminator_settings,
    )

    return Listener(built_language_model, built_discriminator, dropout=dropout)
